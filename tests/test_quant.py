"""quant 패키지 검증 — 손으로 계산 가능한 케이스로 수치를 못 박아 둔다.

실행:  python -m pytest tests/ -v
네트워크가 필요 없도록 전부 합성 데이터를 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import backtest as B, indicators as I, metrics as M, portfolio as PF, signals as S
from quant.config import BacktestConfig, Costs, NO_COSTS


# ── fixtures ───────────────────────────────────────────────────

@pytest.fixture
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 600
    idx = pd.bdate_range("2021-01-04", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    hi = c * (1 + np.abs(rng.normal(0, 0.008, n)))
    lo = c * (1 - np.abs(rng.normal(0, 0.008, n)))
    op = np.r_[c[0], c[:-1]]
    hi = np.maximum.reduce([hi, op, c])
    lo = np.minimum.reduce([lo, op, c])
    return pd.DataFrame(
        {"open": op, "high": hi, "low": lo, "close": c,
         "volume": rng.lognormal(13, 0.5, n)}, index=idx)


# ── metrics: 손검산 ────────────────────────────────────────────

def test_cagr_matches_closed_form():
    """매 봉 정확히 +0.1%, 252봉 = 1년 → CAGR = 1.001^252 - 1."""
    eq = pd.Series(10_000 * np.power(1.001, np.arange(253)))
    p = M.perf_stats(eq, "D")
    assert p.cagr == pytest.approx((1.001 ** 252 - 1) * 100, abs=0.05)
    assert p.max_drawdown == 0.0


def test_max_drawdown_hand_computed():
    """100 → 120 → 60 → 90 이면 최대낙폭은 (120-60)/120 = 50%."""
    assert M.max_drawdown(pd.Series([100, 120, 60, 90])) == pytest.approx(50.0)
    assert M._dd_duration(pd.Series([100, 120, 60, 90])) == 2


def test_zero_volatility_does_not_explode_sharpe():
    """변동성이 0이면 샤프는 무한대가 아니라 0이어야 한다."""
    eq = pd.Series(np.full(300, 10_000.0))
    assert M.perf_stats(eq, "D").sharpe == 0.0


def test_kelly_formula():
    """f* = W - (1-W)/R. 승률 60%, 손익비 2 → 0.4"""
    assert M.kelly_fraction(0.6, 2.0) == pytest.approx(0.4)
    assert M.kelly_fraction(0.3, 1.0) < 0          # 기대값 음수 → 베팅 금지
    assert M.kelly_fraction(0.9, 0.0) == 0.0       # 손익비 0 → 방어


def test_trade_stats_hand_computed():
    tr = pd.DataFrame({"pnl_pct": [10] * 6 + [-5] * 4,
                       "pnl": [10] * 6 + [-5] * 4, "bars": [5] * 10})
    t = M.trade_stats(tr)
    assert t["win_rate"] == pytest.approx(60.0)
    assert t["profit_factor"] == pytest.approx(60 / 20)
    assert t["payoff"] == pytest.approx(2.0)
    assert t["expectancy"] == pytest.approx(4.0)
    assert t["kelly"] == pytest.approx(40.0)
    assert (t["max_win_streak"], t["max_loss_streak"]) == (6, 4)


def test_trade_stats_empty():
    t = M.trade_stats(pd.DataFrame())
    assert t["n_trades"] == 0 and t["kelly"] == 0.0


def test_monte_carlo_is_reproducible_and_ordered():
    pnl = [10] * 6 + [-5] * 4
    a, b = M.monte_carlo(pnl), M.monte_carlo(pnl)
    assert a == b, "같은 입력에 같은 결과가 나와야 한다"
    assert a["p5"] <= a["p50"] <= a["p95"]
    assert 0 <= a["prob_loss"] <= 100
    assert M.monte_carlo([1, 2, 3]) is None, "표본 5개 미만이면 None"


def test_deflated_sharpe_discounts_with_more_trials():
    """탐색 횟수가 늘수록 샤프 신뢰도는 반드시 내려간다."""
    r = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 300))
    psr = M.probabilistic_sharpe(r)
    assert M.deflated_sharpe(r, 1000) <= M.deflated_sharpe(r, 10) <= psr


# ── indicators ─────────────────────────────────────────────────

def test_indicator_ranges(ohlcv):
    ind = I.compute_all(ohlcv)
    assert ind["rsi"].dropna().between(0, 100).all()
    assert ind["adx"].dropna().between(0, 100).all()
    bb = ind[["bb_up", "bb_lo"]].dropna()
    assert (bb["bb_up"] >= bb["bb_lo"]).all()
    assert set(ind["st_dir"].unique()) <= {1, -1}


def test_wilder_seed_matches_sma():
    """와일더 평활의 첫 값은 앞 n개의 단순평균이어야 한다."""
    s = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8])
    w = I.wilder(s, 4)
    assert np.isnan(w.iloc[2])
    assert w.iloc[3] == pytest.approx(2.5)                    # (1+2+3+4)/4
    assert w.iloc[4] == pytest.approx((2.5 * 3 + 5) / 4)


def test_indicators_no_lookahead(ohlcv):
    """미래 봉을 잘라내도 과거 지표값이 바뀌면 안 된다(미래참조 검출)."""
    full = I.compute_all(ohlcv)
    trimmed = I.compute_all(ohlcv.iloc[:-50])
    for col in ["rsi", "macd", "atr", "adx", "sma50", "bb_up", "st_dir"]:
        a = full[col].iloc[:-50].dropna()
        b = trimmed[col].reindex(a.index).dropna()
        common = a.index.intersection(b.index)
        assert np.allclose(a.loc[common], b.loc[common], equal_nan=True), f"{col} 미래참조"


# ── backtest ───────────────────────────────────────────────────

def _flat_market(n=40, price=100.0):
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"open": price, "high": price, "low": price,
                         "close": price, "volume": 1000.0}, index=idx)


def test_take_profit_exit_is_exact():
    """+15% 익절이 정확히 목표가에 체결되는지 (비용 0 기준)."""
    df = _flat_market(30)
    df.iloc[5:, :4] = 100.0
    df.iloc[10:, :4] = 120.0                        # 10봉째부터 +20%
    sig = pd.DataFrame({"buy": False, "sell": False, "buy_reason": "",
                        "sell_reason": "", "buy_strength": 0, "sell_strength": 0},
                       index=df.index)
    sig.iloc[1, sig.columns.get_loc("buy")] = True  # 1봉 시그널 → 2봉 시가 진입

    cfg = BacktestConfig(initial_capital=1_000_000, costs=NO_COSTS,
                         stop_loss=None, take_profit=0.15, max_holding_bars=None)
    res = B.run(df, sig, cfg)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["entry_price"] == pytest.approx(100.0)
    assert t["exit_price"] == pytest.approx(115.0)   # 100 * 1.15
    assert t["pnl_pct"] == pytest.approx(15.0)
    assert t["exit_reason"] == "익절"


def test_stop_loss_wins_ties():
    """같은 봉에 손절·익절이 모두 걸리면 보수적으로 손절이 먼저다."""
    df = _flat_market(20)
    df.iloc[10, df.columns.get_loc("high")] = 130.0   # 익절 도달
    df.iloc[10, df.columns.get_loc("low")] = 90.0     # 손절도 도달
    sig = pd.DataFrame({"buy": False, "sell": False, "buy_reason": "",
                        "sell_reason": "", "buy_strength": 0, "sell_strength": 0},
                       index=df.index)
    sig.iloc[1, sig.columns.get_loc("buy")] = True
    res = B.run(df, sig, BacktestConfig(initial_capital=1e6, costs=NO_COSTS,
                                        stop_loss=0.05, take_profit=0.15,
                                        max_holding_bars=None))
    assert res.trades.iloc[0]["exit_reason"] == "손절"


def test_costs_reduce_returns(ohlcv):
    """수수료·슬리피지를 넣으면 수익률은 반드시 같거나 낮아진다."""
    ind = I.compute_all(ohlcv)
    sig = S.generate(ind)
    free = B.run(ohlcv, sig, BacktestConfig(costs=NO_COSTS))
    paid = B.run(ohlcv, sig, BacktestConfig(costs=Costs(0.001, 0.002, 0.0)))
    if len(free.trades):
        assert paid.performance.total_return <= free.performance.total_return


def test_no_lookahead_entry_uses_next_open():
    """진입가는 시그널 봉의 종가가 아니라 다음 봉 시가여야 한다."""
    df = _flat_market(20)
    df.iloc[:, :4] = 100.0
    df.iloc[3, df.columns.get_loc("close")] = 100.0
    df.iloc[4, df.columns.get_loc("open")] = 111.0   # 다음 봉 시가만 다르게
    sig = pd.DataFrame({"buy": False, "sell": False, "buy_reason": "",
                        "sell_reason": "", "buy_strength": 0, "sell_strength": 0},
                       index=df.index)
    sig.iloc[3, sig.columns.get_loc("buy")] = True
    res = B.run(df, sig, BacktestConfig(costs=NO_COSTS, stop_loss=None,
                                        take_profit=None, max_holding_bars=None))
    assert res.trades.iloc[0]["entry_price"] == pytest.approx(111.0)


def test_equity_length_and_exposure(ohlcv):
    ind = I.compute_all(ohlcv)
    res = B.run(ohlcv, S.generate(ind), BacktestConfig())
    assert len(res.equity) == len(ohlcv)
    assert res.equity.notna().all()
    assert 0.0 <= res.exposure.mean() <= 1.0


def test_no_signals_holds_cash(ohlcv):
    empty = pd.DataFrame({"buy": False, "sell": False, "buy_reason": "",
                          "sell_reason": "", "buy_strength": 0, "sell_strength": 0},
                         index=ohlcv.index)
    res = B.run(ohlcv, empty, BacktestConfig(initial_capital=5_000_000))
    assert len(res.trades) == 0
    assert res.equity.nunique() == 1
    assert res.equity.iloc[-1] == pytest.approx(5_000_000)


# ── signals ────────────────────────────────────────────────────

def test_signals_shape_and_exclusivity(ohlcv):
    sig = S.generate(I.compute_all(ohlcv))
    assert len(sig) == len(ohlcv)
    assert not (sig["buy"] & sig["sell"]).any(), "같은 봉에 매수·매도가 동시에 서면 안 된다"
    assert sig["buy"].dtype == bool


def test_score_returns_bounded_dict(ohlcv):
    r = S.score(I.compute_all(ohlcv))
    assert isinstance(r["score"], (int, np.integer)) and r["score"] >= 0
    assert isinstance(r["reasons"], list)


# ── portfolio ──────────────────────────────────────────────────

@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n, k = 700, 6
    idx = pd.bdate_range("2021-01-04", periods=n)
    base = rng.normal(0, 0.012, (n, 2))
    cols = {}
    for i in range(k):
        g = base[:, 0] if i < 3 else base[:, 1]
        cols[f"S{i}"] = 100 * np.exp(np.cumsum(0.7 * g + 0.3 * rng.normal(0.0003, 0.014, n)))
    return pd.DataFrame(cols, index=idx)


@pytest.mark.parametrize("method", PF.METHODS)
def test_weights_are_valid(prices, method):
    w = PF.optimize(prices, method)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all(), "공매도 비중이 나오면 안 된다"
    assert len(w) == prices.shape[1]


@pytest.mark.parametrize("method", ["min_var", "max_sharpe"])
def test_max_weight_cap_respected(prices, method):
    w = PF.optimize(prices, method, max_weight=0.30)
    assert w.max() <= 0.3001


def test_single_asset_portfolio(prices):
    w = PF.optimize(prices[["S0"]], "hrp")
    assert w.iloc[0] == pytest.approx(1.0)


def test_rebalance_equity_is_positive(prices):
    w = PF.optimize(prices, "hrp")
    eq = PF.backtest_weights(prices, w)
    assert len(eq) == len(prices)
    assert (eq > 0).all()
