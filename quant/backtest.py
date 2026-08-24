"""백테스트 엔진 — 롱온리, 다음 봉 시가 체결, 거래비용 반영.

index.html의 signalBacktest와 다른 점 두 가지:
  1) 체결을 시그널 발생 봉의 종가가 아니라 **다음 봉 시가**로 한다.
     종가 체결은 "종가를 보고 종가에 산다"는 뜻이라 미래참조이고,
     실전 수익률을 체계적으로 부풀린다.
  2) 수수료·슬리피지·거래세를 뺀다. 회전율이 높은 전략일수록 차이가 크다.

동시에 손절/익절이 걸린 봉은 보수적으로 **손절 먼저** 체결한 것으로 본다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BacktestConfig, Costs
from . import metrics as M

__all__ = ["BacktestResult", "run"]


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    exposure: pd.Series
    benchmark: pd.Series
    config: BacktestConfig
    _perf: object = field(default=None, repr=False)

    @property
    def performance(self):
        if self._perf is None:
            self._perf = M.perf_stats(
                self.equity, interval=self.config.interval,
                benchmark=self.benchmark, exposure=self.exposure,
            )
        return self._perf

    @property
    def stats(self) -> dict:
        return M.trade_stats(self.trades)

    def monte_carlo(self, runs: int = 5000) -> dict | None:
        if len(self.trades) == 0:
            return None
        return M.monte_carlo(self.trades["pnl_pct"].to_numpy(), runs=runs)

    def summary(self) -> str:
        head = (
            f"{self.equity.index[0].date()} ~ {self.equity.index[-1].date()}  "
            f"{len(self.equity)}봉  초기자금 {self.config.initial_capital:,.0f}\n"
        )
        return head + M.summary_table(self.performance, self.stats, self.monte_carlo())


def run(
    df: pd.DataFrame,
    sig: pd.DataFrame,
    config: BacktestConfig | None = None,
    apply_tax: bool = False,
) -> BacktestResult:
    """OHLCV + 시그널 → 자산곡선과 체결 내역.

    df  : open/high/low/close/volume
    sig : buy/sell (bool)  — signals.generate() 출력
    """
    cfg = config or BacktestConfig()
    costs: Costs = cfg.costs
    idx = df.index
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)

    buy = sig["buy"].reindex(idx).fillna(False).to_numpy(bool)
    sell = sig["sell"].reindex(idx).fillna(False).to_numpy(bool)

    cash = float(cfg.initial_capital)
    shares = 0.0
    entry_price = entry_i = 0
    entry_date = None
    entry_reason = ""

    equity = np.empty(n)
    in_market = np.zeros(n, dtype=bool)
    trades: list[dict] = []

    buy_reason = sig["buy_reason"].reindex(idx).fillna("").to_numpy()
    sell_reason = sig["sell_reason"].reindex(idx).fillna("").to_numpy()

    def _close(i: int, price: float, why: str):
        nonlocal cash, shares, entry_price, entry_i, entry_date, entry_reason
        proceeds = shares * price * (1 - costs.exit_cost(apply_tax))
        invested = shares * entry_price * (1 + costs.entry_cost())
        trades.append({
            "entry_date": entry_date, "exit_date": idx[i],
            "entry_price": entry_price, "exit_price": price,
            "pnl": proceeds - invested,
            "pnl_pct": (proceeds / invested - 1) * 100 if invested else 0.0,
            "bars": i - entry_i,
            "entry_reason": entry_reason, "exit_reason": why,
        })
        cash = proceeds
        shares = 0.0

    for i in range(n):
        # ── 1) 보유 중이면 먼저 청산 조건을 본다 (손절 → 익절 → 시그널 → 보유한도)
        if shares > 0:
            stop = entry_price * (1 - cfg.stop_loss) if cfg.stop_loss else None
            target = entry_price * (1 + cfg.take_profit) if cfg.take_profit else None

            if stop is not None and l[i] <= stop:
                _close(i, stop, "손절")
            elif target is not None and h[i] >= target:
                _close(i, target, "익절")
            elif sell[i]:
                # 시그널은 이 봉 종가에 확정 → 다음 봉 시가에 청산
                if i + 1 < n:
                    _close(i + 1, o[i + 1], sell_reason[i] or "매도시그널")
                else:
                    _close(i, c[i], sell_reason[i] or "매도시그널(최종봉)")
            elif cfg.max_holding_bars and (i - entry_i) >= cfg.max_holding_bars:
                _close(i, c[i], "보유한도")

        # ── 2) 비어 있으면 진입 (직전 봉 시그널 → 이번 봉 시가)
        if shares == 0 and i > 0 and buy[i - 1] and i < n - 1:
            fill = o[i] * (1 + costs.entry_cost())
            if fill > 0:
                shares = cash / fill
                entry_price = o[i]
                entry_i, entry_date = i, idx[i]
                entry_reason = buy_reason[i - 1] or "매수시그널"
                cash = 0.0

        equity[i] = cash + shares * c[i]
        in_market[i] = shares > 0

    # 미청산 포지션은 마지막 종가로 정리
    if shares > 0:
        _close(n - 1, c[-1], "기간만료")
        equity[-1] = cash

    tdf = pd.DataFrame(trades)
    if len(tdf) == 0:
        tdf = pd.DataFrame(columns=["entry_date", "exit_date", "entry_price", "exit_price",
                                    "pnl", "pnl_pct", "bars", "entry_reason", "exit_reason"])

    return BacktestResult(
        equity=pd.Series(equity, index=idx, name="equity"),
        trades=tdf,
        exposure=pd.Series(in_market, index=idx, name="in_market"),
        benchmark=df["close"],
        config=cfg,
    )
