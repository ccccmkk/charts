"""포트폴리오 최적화 — 종목 하나가 아니라 바구니를 다룬다.

Riskfolio-Lib / PyPortfolioOpt가 하는 일 중 실전에서 실제로 쓰는 것만
numpy·scipy로 직접 구현했다. 무거운 의존성 없이 돌고 결과는 동일하다.

  · equal        동일비중 — 이기기 의외로 어려운 기준선
  · inverse_vol  변동성 역수 — 계산 한 줄, 성능은 준수
  · risk_parity  위험기여도 균등
  · min_var      최소분산
  · max_sharpe   샤프 최대화
  · hrp          계층적 위험 패리티 (López de Prado) — 공분산 역행렬을 쓰지
                 않아서 종목수가 많아도 안정적이다
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["optimize", "backtest_weights", "METHODS"]

METHODS = ["equal", "inverse_vol", "risk_parity", "min_var", "max_sharpe", "hrp"]


def _cov(returns: pd.DataFrame) -> np.ndarray:
    return returns.cov().to_numpy()


def _clean(returns: pd.DataFrame) -> pd.DataFrame:
    r = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    return r.dropna()


# ── 개별 방법 ──────────────────────────────────────────────────

def _equal(r: pd.DataFrame) -> np.ndarray:
    return np.repeat(1 / r.shape[1], r.shape[1])


def _inverse_vol(r: pd.DataFrame) -> np.ndarray:
    v = r.std(ddof=1).to_numpy()
    v = np.where(v <= 0, np.nan, v)
    w = 1 / v
    w = np.nan_to_num(w)
    return w / w.sum() if w.sum() > 0 else _equal(r)


def _risk_parity(r: pd.DataFrame, iters: int = 500) -> np.ndarray:
    """위험기여도가 같아지도록 반복 조정 (곱셈 업데이트)."""
    S = _cov(r)
    n = S.shape[0]
    w = np.repeat(1 / n, n)
    target = 1 / n
    for _ in range(iters):
        mrc = S @ w                      # 한계 위험기여
        port_var = float(w @ S @ w)
        if port_var <= 0:
            break
        rc = w * mrc / port_var          # 위험기여 비중
        w = w * (target / np.maximum(rc, 1e-12)) ** 0.1
        w = np.clip(w, 0, None)
        s = w.sum()
        if s <= 0:
            return _equal(r)
        w /= s
    return w


def _solve(r: pd.DataFrame, objective, max_weight: float = 1.0) -> np.ndarray:
    """long-only, 합=1 제약 하 최소화. scipy가 없으면 역변동성으로 대체.

    max_weight는 한 종목 상한. 최대샤프는 제약이 없으면 거의 항상 한 종목에
    100%를 몰아주는 코너해로 수렴하므로(위 검증에서 실제로 그랬다) 실전에서는
    상한을 걸어야 쓸 수 있다.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        return _inverse_vol(r)

    n = r.shape[1]
    cap = max(float(max_weight), 1.0 / n)   # 상한이 너무 낮으면 해가 없다
    w0 = np.repeat(1 / n, n)
    res = minimize(
        objective, w0, method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not res.success or not np.isfinite(res.x).all():
        return _inverse_vol(r)
    w = np.clip(res.x, 0, None)
    return w / w.sum() if w.sum() > 0 else _equal(r)


def _min_var(r: pd.DataFrame, max_weight: float = 1.0) -> np.ndarray:
    S = _cov(r)
    return _solve(r, lambda w: float(w @ S @ w), max_weight)


def _max_sharpe(r: pd.DataFrame, periods: int = 252, max_weight: float = 1.0) -> np.ndarray:
    S = _cov(r)
    mu = r.mean().to_numpy()

    def neg_sharpe(w):
        vol = np.sqrt(max(float(w @ S @ w), 1e-18))
        return -(float(mu @ w) * periods) / (vol * np.sqrt(periods))

    return _solve(r, neg_sharpe, max_weight)


def _hrp(r: pd.DataFrame) -> np.ndarray:
    """계층적 위험 패리티 — 상관 거리로 군집화 후 재귀 이분할."""
    try:
        from scipy.cluster.hierarchy import linkage, to_tree
    except ImportError:
        return _inverse_vol(r)

    S = r.cov()
    corr = r.corr().fillna(0).to_numpy()
    n = corr.shape[0]
    if n < 2:
        return np.array([1.0])

    dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    np.fill_diagonal(dist, 0.0)
    # condensed 거리 벡터
    iu = np.triu_indices(n, 1)
    link = linkage(dist[iu], method="single")

    # 준대각화 — 군집 순서대로 종목 재배열
    root = to_tree(link)
    order: list[int] = []

    def _walk(node):
        if node.is_leaf():
            order.append(node.id)
            return
        _walk(node.get_left())
        _walk(node.get_right())

    _walk(root)

    cov = S.to_numpy()
    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            mid = len(cl) // 2
            left, right = cl[:mid], cl[mid:]

            def _cluster_var(items):
                sub = cov[np.ix_(items, items)]
                d = np.diag(sub)
                iv = 1 / np.where(d <= 0, np.nan, d)
                iv = np.nan_to_num(iv)
                iv = iv / iv.sum() if iv.sum() > 0 else np.repeat(1 / len(items), len(items))
                return float(iv @ sub @ iv)

            vl, vr = _cluster_var(left), _cluster_var(right)
            alpha = 1 - vl / (vl + vr) if (vl + vr) > 0 else 0.5
            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= 1 - alpha
            nxt += [left, right]
        clusters = nxt

    return w / w.sum() if w.sum() > 0 else _equal(r)


_DISPATCH = {
    "equal": _equal, "inverse_vol": _inverse_vol, "risk_parity": _risk_parity,
    "min_var": _min_var, "max_sharpe": _max_sharpe, "hrp": _hrp,
}


# ── 공개 API ───────────────────────────────────────────────────

def optimize(
    prices: pd.DataFrame, method: str = "hrp", periods: int = 252,
    max_weight: float = 0.35,
) -> pd.Series:
    """종가 매트릭스 → 비중. 합은 1, 공매도 없음.

    max_weight: 한 종목 상한 (min_var / max_sharpe에만 적용). 기본 35%.
                1.0으로 두면 제약 없는 이론해가 나오는데, 대개 한 종목에
                전부 몰리므로 실사용은 권하지 않는다.
    """
    if method not in _DISPATCH:
        raise ValueError(f"모르는 방법: {method} (가능: {', '.join(METHODS)})")
    r = _clean(prices.pct_change())
    if r.empty or r.shape[1] == 0:
        return pd.Series(dtype=float)
    if r.shape[1] == 1:
        return pd.Series([1.0], index=r.columns)

    if method == "max_sharpe":
        w = _max_sharpe(r, periods, max_weight)
    elif method == "min_var":
        w = _min_var(r, max_weight)
    else:
        w = _DISPATCH[method](r)
    return pd.Series(w, index=r.columns).sort_values(ascending=False)


def backtest_weights(
    prices: pd.DataFrame, weights: pd.Series, rebalance: str = "ME",
    initial: float = 10_000_000.0,
) -> pd.Series:
    """고정 비중을 주기적으로 리밸런싱했을 때의 자산곡선."""
    cols = [c for c in weights.index if c in prices.columns]
    px = prices[cols].dropna()
    if px.empty:
        return pd.Series(dtype=float)
    w = weights[cols].to_numpy(float)
    w = w / w.sum()

    rets = px.pct_change().fillna(0.0)
    marks = rets.resample(rebalance).last().index
    equity, val, cur = [], initial, w.copy()

    for dt, row in rets.iterrows():
        cur = cur * (1 + row.to_numpy())
        s = cur.sum()
        val *= s if s > 0 else 1.0
        cur = cur / s if s > 0 else w.copy()
        if dt in marks:
            cur = w.copy()            # 리밸런싱: 목표 비중으로 복귀
        equity.append(val)
    return pd.Series(equity, index=rets.index, name="equity")


def compare(prices: pd.DataFrame, periods: int = 252,
            max_weight: float = 0.35) -> pd.DataFrame:
    """모든 방법을 한 표로 비교한다."""
    from . import metrics as M

    rows = []
    for m in METHODS:
        try:
            w = optimize(prices, m, periods, max_weight)
            eq = backtest_weights(prices, w)
            if eq.empty:
                continue
            p = M.perf_stats(eq, "D")
            rows.append({
                "method": m, "cagr": p.cagr, "sharpe": p.sharpe,
                "sortino": p.sortino, "max_dd": p.max_drawdown,
                "calmar": p.calmar,
                "top_holding": f"{w.index[0]} {w.iloc[0]:.1%}",
                "n_effective": float(1 / (w**2).sum()),   # 유효 종목수
            })
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
