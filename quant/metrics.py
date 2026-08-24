"""성과지표 — QuantStats 계열 지표 + 몬테카를로 + 켈리 + 과최적화 진단.

의존성은 numpy/pandas뿐이다. quantstats가 설치돼 있으면 tearsheet 생성에만
선택적으로 쓴다(`full_report`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import pandas as pd

from .config import ANNUALIZATION

__all__ = [
    "Performance", "returns_from_equity", "drawdown_series", "max_drawdown",
    "perf_stats", "trade_stats", "kelly_fraction", "monte_carlo",
    "probabilistic_sharpe", "deflated_sharpe", "summary_table",
]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def ann_factor(interval: str = "D", crypto: bool = False) -> float:
    if crypto:
        return {"D": 365, "W": 52, "M": 12}.get(interval, 365)
    return ANNUALIZATION.get(interval, 252)


# ── 기본 변환 ──────────────────────────────────────────────────

def returns_from_equity(equity: pd.Series | Sequence[float]) -> pd.Series:
    eq = pd.Series(equity).astype(float)
    return eq.pct_change().dropna()


def drawdown_series(equity: pd.Series) -> pd.Series:
    eq = pd.Series(equity).astype(float)
    return eq / eq.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """최대낙폭(양수 %). 25.3이면 -25.3%를 뜻한다."""
    dd = drawdown_series(equity)
    return float(max(0.0, -dd.min() * 100)) if len(dd) else 0.0


def _dd_duration(equity: pd.Series) -> int:
    """고점을 회복하기까지 걸린 최장 봉 수."""
    eq = pd.Series(equity).astype(float).to_numpy()
    peak, peak_i, worst = -np.inf, 0, 0
    for i, v in enumerate(eq):
        if v >= peak:
            peak, peak_i = v, i
        elif i - peak_i > worst:
            worst = i - peak_i
    return int(worst)


# ── 과최적화 진단 ──────────────────────────────────────────────

def probabilistic_sharpe(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """PSR — 관측된 샤프가 기준치를 진짜로 넘을 확률(0~1).

    Bailey & López de Prado. 수익률의 왜도/첨도까지 반영하기 때문에
    "샤프 2.0인데 꼬리가 두꺼운 전략"을 걸러낸다. 표본이 적으면 값이 떨어진다.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 10 or not np.isfinite(r.std(ddof=1)) or r.std(ddof=1) < 1e-12:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)                       # 연율화 전 샤프
    skew = float(r.skew())
    kurt = float(r.kurtosis()) + 3.0                    # pandas는 초과첨도를 준다
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    z = (sr - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(_norm_cdf(z))


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sr_std: float | None = None) -> float:
    """DSR — 파라미터를 N번 탐색했다는 사실을 반영해 샤프를 할인한다.

    파라미터 스윕을 1000번 돌려 가장 좋은 걸 고르면 순전히 운으로도 높은
    샤프가 나온다. n_trials가 클수록 요구 기준선이 올라간다.
    """
    r = pd.Series(returns).dropna()
    if n_trials < 2 or len(r) < 10:
        return probabilistic_sharpe(r)
    v = trial_sr_std if trial_sr_std and trial_sr_std > 0 else float(r.std(ddof=1)) or 1.0
    gamma = 0.5772156649015329                          # 오일러-마스케로니
    e1 = math.sqrt(2 * math.log(n_trials))
    # 기대 최대 샤프 (극단값 이론 근사)
    sr0 = v * ((1 - gamma) * e1 + gamma * math.sqrt(2 * math.log(n_trials / math.e)))
    return probabilistic_sharpe(r, benchmark_sr=sr0)


# ── 핵심 지표 ──────────────────────────────────────────────────

@dataclass
class Performance:
    bars: int
    years: float
    total_return: float      # %
    cagr: float              # %
    volatility: float        # % 연율화
    sharpe: float
    sortino: float
    calmar: float
    omega: float
    max_drawdown: float      # % (양수)
    dd_duration: int         # 봉
    ulcer: float
    var95: float             # % (1봉 기준)
    cvar95: float            # %
    skew: float
    kurtosis: float
    tail_ratio: float
    exposure: float          # %
    psr: float               # 0~1
    benchmark_return: float  # % Buy & Hold
    alpha: float             # %p

    def to_dict(self) -> dict:
        return asdict(self)


def perf_stats(
    equity: pd.Series,
    interval: str = "D",
    benchmark: pd.Series | None = None,
    exposure: pd.Series | None = None,
    crypto: bool = False,
) -> Performance:
    """자산곡선 하나에서 위험조정 지표 일체를 뽑는다."""
    eq = pd.Series(equity).astype(float).dropna()
    ann = ann_factor(interval, crypto)
    r = returns_from_equity(eq)
    n = len(r)

    bars = max(len(eq) - 1, 1)
    years = bars / ann
    growth = float(eq.iloc[-1] / eq.iloc[0]) if len(eq) and eq.iloc[0] > 0 else 1.0
    total = (growth - 1) * 100
    cagr = (growth ** (1 / years) - 1) * 100 if years > 0 and growth > 0 else 0.0

    mu = float(r.mean()) if n else 0.0
    sd = float(r.std(ddof=1)) if n > 1 else 0.0
    vol = sd * math.sqrt(ann) * 100
    downside = r[r < 0]
    dsd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0

    # 변동성이 사실상 0이면(합성 데이터·미체결 구간) 샤프가 무한대로 튄다
    eps = 1e-12
    sharpe = (mu * ann) / (sd * math.sqrt(ann)) if sd > eps else 0.0
    sortino = (mu * ann) / (dsd * math.sqrt(ann)) if dsd > eps else 0.0
    mdd = max_drawdown(eq)
    calmar = cagr / mdd if mdd > 0 else 0.0

    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    omega = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0

    dd = drawdown_series(eq)
    ulcer = float(np.sqrt((dd.to_numpy() ** 2).mean()) * 100) if len(dd) else 0.0

    var95 = float(np.percentile(r, 5) * 100) if n else 0.0
    tail = r[r <= np.percentile(r, 5)] if n else r
    cvar95 = float(tail.mean() * 100) if len(tail) else 0.0

    if n:
        p95, p05 = np.percentile(r, 95), np.percentile(r, 5)
        tail_ratio = float(abs(p95 / p05)) if p05 != 0 else 0.0
    else:
        tail_ratio = 0.0

    expo = float(pd.Series(exposure).mean() * 100) if exposure is not None and len(exposure) else 100.0

    if benchmark is not None and len(benchmark) > 1:
        b = pd.Series(benchmark).astype(float).dropna()
        bench = (float(b.iloc[-1] / b.iloc[0]) - 1) * 100
    else:
        bench = 0.0

    return Performance(
        bars=bars, years=years, total_return=total, cagr=cagr, volatility=vol,
        sharpe=sharpe, sortino=sortino, calmar=calmar, omega=omega,
        max_drawdown=mdd, dd_duration=_dd_duration(eq), ulcer=ulcer,
        var95=var95, cvar95=cvar95,
        skew=float(r.skew()) if n > 2 else 0.0,
        kurtosis=float(r.kurtosis()) if n > 3 else 0.0,
        tail_ratio=tail_ratio, exposure=expo,
        psr=probabilistic_sharpe(r), benchmark_return=bench, alpha=total - bench,
    )


# ── 거래 단위 통계 ─────────────────────────────────────────────

def trade_stats(trades: pd.DataFrame) -> dict:
    """체결 단위 통계. trades는 pnl_pct / pnl / bars 컬럼을 가진다."""
    if trades is None or len(trades) == 0:
        return {"n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "payoff": 0.0,
                "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "best": 0.0,
                "worst": 0.0, "avg_bars": 0.0, "max_win_streak": 0,
                "max_loss_streak": 0, "kelly": 0.0}

    t = trades.copy()
    pnl = t["pnl_pct"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_p, gross_l = float(wins.sum()), float(-losses.sum())

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0
    payoff = avg_win / avg_loss if avg_loss > 0 else 0.0
    wr = len(wins) / len(t)

    streak = (pnl > 0).astype(int)
    max_w = max_l = cur_w = cur_l = 0
    for x in streak:
        if x:
            cur_w += 1; cur_l = 0; max_w = max(max_w, cur_w)
        else:
            cur_l += 1; cur_w = 0; max_l = max(max_l, cur_l)

    return {
        "n_trades": int(len(t)),
        "win_rate": wr * 100,
        "profit_factor": gross_p / gross_l if gross_l > 0 else (float("inf") if gross_p > 0 else 0.0),
        "payoff": payoff,
        "expectancy": float(pnl.mean()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best": float(pnl.max()),
        "worst": float(pnl.min()),
        "avg_bars": float(t["bars"].mean()) if "bars" in t else 0.0,
        "max_win_streak": max_w,
        "max_loss_streak": max_l,
        "kelly": kelly_fraction(wr, payoff) * 100,
    }


def kelly_fraction(win_rate: float, payoff: float) -> float:
    """켈리 기준 f* = W - (1-W)/R.

    음수면 기댓값이 마이너스라 베팅하지 않는 것이 최적이다.
    실전에서는 추정오차와 변동성 때문에 Half Kelly 이하를 쓴다.
    """
    if payoff <= 0:
        return 0.0
    return win_rate - (1 - win_rate) / payoff


# ── 몬테카를로 ─────────────────────────────────────────────────

def monte_carlo(
    trade_returns: Sequence[float], runs: int = 5000, seed: int = 42
) -> dict | None:
    """거래 순서를 부트스트랩 재추출해 결과 분포를 만든다.

    "그 수익률이 실력인가 순서 운인가"를 가르는 도구. 실제 성과가 중앙값보다
    한참 위라면 운이 좋았던 배열일 가능성이 크다.
    """
    r = np.asarray([x / 100.0 for x in trade_returns], dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return None

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(runs, len(r)))
    paths = np.cumprod(1.0 + r[idx], axis=1)

    finals = (paths[:, -1] - 1.0) * 100
    peaks = np.maximum.accumulate(paths, axis=1)
    dds = ((peaks - paths) / peaks).max(axis=1) * 100

    q = lambda a, p: float(np.percentile(a, p))
    return {
        "runs": runs,
        "p5": q(finals, 5), "p25": q(finals, 25), "p50": q(finals, 50),
        "p75": q(finals, 75), "p95": q(finals, 95),
        "median_dd": q(dds, 50), "worst_dd": q(dds, 95),
        "prob_loss": float((finals < 0).mean() * 100),
        "prob_ruin": float((paths.min(axis=1) < 0.5).mean() * 100),  # 반토막 확률
    }


# ── 출력 ──────────────────────────────────────────────────────

def summary_table(perf: Performance, tstats: dict, mc: dict | None = None) -> str:
    """터미널용 한 화면 요약."""
    L = []
    add = L.append
    add("┌─ 위험조정 수익 " + "─" * 46)
    add(f"│ CAGR {perf.cagr:>8.2f}%   Sharpe {perf.sharpe:>6.2f}   Sortino {perf.sortino:>6.2f}")
    add(f"│ 변동성{perf.volatility:>8.2f}%   Calmar {perf.calmar:>6.2f}   Omega   {perf.omega:>6.2f}")
    add(f"│ MDD  {perf.max_drawdown:>8.2f}%   기간 {perf.dd_duration:>5d}봉   Ulcer   {perf.ulcer:>6.2f}")
    add(f"│ VaR95{perf.var95:>8.2f}%   CVaR95{perf.cvar95:>7.2f}%   꼬리비율{perf.tail_ratio:>6.2f}")
    add(f"│ 왜도 {perf.skew:>8.2f}    첨도 {perf.kurtosis:>7.2f}    노출도 {perf.exposure:>6.1f}%")
    add(f"│ PSR  {perf.psr:>8.1%}  ← 샤프가 0보다 클 확률 (표본·꼬리 반영)")
    add("├─ 벤치마크 대비 " + "─" * 46)
    add(f"│ 전략 {perf.total_return:>8.2f}%   단순보유 {perf.benchmark_return:>8.2f}%   알파 {perf.alpha:>8.2f}%p")
    add("├─ 거래 통계 " + "─" * 50)
    t = tstats
    pf = t["profit_factor"]
    pf_s = "∞" if pf == float("inf") else f"{pf:.2f}"
    add(f"│ 거래 {t['n_trades']:>5d}회   승률 {t['win_rate']:>6.1f}%   PF {pf_s:>6}   손익비 {t['payoff']:>5.2f}")
    add(f"│ 기대값{t['expectancy']:>7.2f}%   평균수익 {t['avg_win']:>6.2f}%   평균손실 {-t['avg_loss']:>7.2f}%")
    add(f"│ 최고 {t['best']:>7.2f}%   최악 {t['worst']:>9.2f}%   연승/연패 {t['max_win_streak']}/{t['max_loss_streak']}")
    add(f"│ 켈리 {t['kelly']:>7.1f}%   (Half Kelly {t['kelly']/2:>5.1f}%)   평균보유 {t['avg_bars']:>4.1f}봉")
    if mc:
        add("├─ 몬테카를로 " + "─" * 49)
        add(f"│ {mc['runs']}회 재추출 → 비관5% {mc['p5']:>7.1f}%  중앙 {mc['p50']:>7.1f}%  낙관95% {mc['p95']:>7.1f}%")
        add(f"│ 손실확률 {mc['prob_loss']:>5.1f}%   반토막확률 {mc['prob_ruin']:>5.1f}%   예상MDD {mc['median_dd']:>5.1f}%")
    add("└" + "─" * 62)
    return "\n".join(L)
