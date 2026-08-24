"""파라미터 최적화와 워크포워드 검증.

브라우저에서 못 하던 작업이 여기 있다. 수백~수천 조합을 전수 탐색하면
반드시 "그럴듯한 최고 성적"이 나오는데, 그 대부분은 과최적화다.
그래서 이 모듈은 항상 두 가지를 같이 낸다:

  · in-sample 최고 성적            (믿으면 안 되는 숫자)
  · walk-forward out-of-sample 성적 (실제로 참고할 숫자)

Deflated Sharpe로 "N번 뒤져서 얻은 샤프"를 시행횟수만큼 할인해 보여준다.
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

from . import backtest as B, indicators as I, metrics as M, signals as S
from .config import BacktestConfig

__all__ = ["sweep", "walk_forward", "DEFAULT_GRID"]


DEFAULT_GRID = {
    "rsi_oversold": [25, 30, 35],
    "vol_spike": [1.5, 1.8, 2.2],
    "adx_trend": [15, 20, 25],
}


def _evaluate(df: pd.DataFrame, ind: pd.DataFrame, params: dict,
              cfg: BacktestConfig, apply_tax: bool) -> tuple[dict, B.BacktestResult]:
    sig = S.generate(ind, S.SignalParams(**params))
    res = B.run(df, sig, cfg, apply_tax=apply_tax)
    perf, ts = res.performance, res.stats
    row = {
        **params,
        "total_return": perf.total_return, "cagr": perf.cagr,
        "sharpe": perf.sharpe, "sortino": perf.sortino, "calmar": perf.calmar,
        "max_dd": perf.max_drawdown, "psr": perf.psr,
        "n_trades": ts["n_trades"], "win_rate": ts["win_rate"],
        "profit_factor": ts["profit_factor"], "expectancy": ts["expectancy"],
    }
    return row, res


def sweep(
    df: pd.DataFrame,
    grid: dict | None = None,
    config: BacktestConfig | None = None,
    metric: str = "sharpe",
    min_trades: int = 5,
    apply_tax: bool = False,
) -> pd.DataFrame:
    """파라미터 전수 탐색. 거래수가 너무 적은 조합은 걸러낸다."""
    grid = grid or DEFAULT_GRID
    cfg = config or BacktestConfig()
    ind = I.compute_all(df)

    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        try:
            row, _ = _evaluate(df, ind, params, cfg, apply_tax)
        except Exception:
            continue
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["valid"] = out["n_trades"] >= min_trades
    out = out.sort_values([metric], ascending=False).reset_index(drop=True)

    # 시행횟수만큼 할인한 샤프 — 이 값이 낮으면 1등 조합도 운일 가능성이 크다
    n_trials = len(out)
    sr_std = float(out["sharpe"].std(ddof=1)) if len(out) > 1 else None
    best = out.iloc[0]
    if best["n_trades"] >= min_trades:
        params = {k: best[k] for k in keys}
        _, res = _evaluate(df, ind, params, cfg, apply_tax)
        r = M.returns_from_equity(res.equity)
        out.attrs["deflated_sharpe"] = M.deflated_sharpe(r, n_trials, sr_std)
    out.attrs["n_trials"] = n_trials
    return out


def walk_forward(
    df: pd.DataFrame,
    grid: dict | None = None,
    config: BacktestConfig | None = None,
    n_splits: int = 4,
    train_ratio: float = 0.7,
    metric: str = "sharpe",
    min_trades: int = 3,
    apply_tax: bool = False,
) -> dict:
    """구간을 나눠 학습구간에서 고른 파라미터를 다음 구간에 그대로 적용한다.

    각 구간의 out-of-sample 성적을 이어붙인 것이 실전에 가장 가까운 추정치다.
    in-sample 대비 성적이 크게 무너지면 그 전략은 과최적화된 것이다.
    """
    grid = grid or DEFAULT_GRID
    cfg = config or BacktestConfig()
    keys = list(grid)
    n = len(df)
    if n < 250:
        raise ValueError(f"워크포워드에는 최소 250봉이 필요하다 (현재 {n}봉)")

    fold = n // n_splits
    folds = []
    oos_returns: list[pd.Series] = []

    for k in range(n_splits):
        lo = k * fold
        hi = n if k == n_splits - 1 else (k + 1) * fold
        seg = df.iloc[lo:hi]
        cut = int(len(seg) * train_ratio)
        if cut < 60 or len(seg) - cut < 30:
            continue
        train, test = seg.iloc[:cut], seg.iloc[cut:]

        # 학습구간에서 최적 조합 선택
        tr_ind = I.compute_all(train)
        best_row, best_val, best_params = None, -np.inf, None
        for combo in itertools.product(*(grid[k2] for k2 in keys)):
            params = dict(zip(keys, combo))
            try:
                row, _ = _evaluate(train, tr_ind, params, cfg, apply_tax)
            except Exception:
                continue
            if row["n_trades"] < min_trades:
                continue
            if row[metric] > best_val:
                best_row, best_val, best_params = row, row[metric], params
        if best_params is None:
            continue

        # 그대로 검증구간에 적용 (재탐색 없음 = out-of-sample)
        te_ind = I.compute_all(test)
        oos_row, oos_res = _evaluate(test, te_ind, best_params, cfg, apply_tax)
        oos_returns.append(M.returns_from_equity(oos_res.equity))

        folds.append({
            "fold": k + 1,
            "train_period": f"{train.index[0].date()}~{train.index[-1].date()}",
            "test_period": f"{test.index[0].date()}~{test.index[-1].date()}",
            "params": best_params,
            "is_sharpe": best_row["sharpe"], "oos_sharpe": oos_row["sharpe"],
            "is_return": best_row["total_return"], "oos_return": oos_row["total_return"],
            "oos_trades": oos_row["n_trades"], "oos_max_dd": oos_row["max_dd"],
        })

    if not folds:
        return {"folds": [], "verdict": "구간이 부족해 검증하지 못했다"}

    fdf = pd.DataFrame(folds)
    is_sr, oos_sr = fdf["is_sharpe"].mean(), fdf["oos_sharpe"].mean()
    # 저하율 — 1에 가까울수록 과최적화가 심하다
    decay = 1 - (oos_sr / is_sr) if is_sr > 0 else 1.0

    if oos_sr <= 0:
        verdict = "❌ 실패 — out-of-sample 샤프가 0 이하다. 실전 투입 금지."
    elif decay > 0.7:
        verdict = f"⚠ 과최적화 의심 — 성능이 {decay:.0%} 무너졌다."
    elif decay > 0.4:
        verdict = f"△ 보통 — {decay:.0%} 저하. 파라미터 민감도가 높다."
    else:
        verdict = f"✅ 견고 — 저하율 {decay:.0%}. 구간이 바뀌어도 유지된다."

    combined = pd.concat(oos_returns) if oos_returns else pd.Series(dtype=float)
    return {
        "folds": folds,
        "table": fdf,
        "is_sharpe_mean": float(is_sr),
        "oos_sharpe_mean": float(oos_sr),
        "decay": float(decay),
        "oos_psr": M.probabilistic_sharpe(combined) if len(combined) > 10 else float("nan"),
        "verdict": verdict,
    }
