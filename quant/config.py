"""전역 설정 — 연율화 계수, 거래비용, 스캐너 유니버스."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 봉 주기별 연율화 계수 (Sharpe/CAGR 계산에 사용)
ANNUALIZATION = {"D": 252, "W": 52, "M": 12, "60m": 252 * 6.5, "H": 252 * 6.5}
CRYPTO_ANNUALIZATION = {"D": 365, "W": 52, "M": 12}

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


@dataclass(frozen=True)
class Costs:
    """왕복 거래비용. 백테스트 결과를 현실에 붙이는 가장 중요한 파라미터."""

    commission: float = 0.0005   # 편도 수수료 5bp
    slippage: float = 0.0010     # 편도 슬리피지 10bp
    tax: float = 0.0018          # 국내 매도 시 증권거래세 (코스피/코스닥 0.18%)

    def entry_cost(self) -> float:
        return self.commission + self.slippage

    def exit_cost(self, apply_tax: bool = False) -> float:
        return self.commission + self.slippage + (self.tax if apply_tax else 0.0)


# 비용 0 — 순수 시그널 성능을 볼 때만 사용
NO_COSTS = Costs(commission=0.0, slippage=0.0, tax=0.0)
US_COSTS = Costs(commission=0.0005, slippage=0.0010, tax=0.0)
KR_COSTS = Costs(commission=0.00015, slippage=0.0010, tax=0.0018)


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000_000.0
    costs: Costs = field(default_factory=Costs)
    interval: str = "D"
    # 시그널이 없어도 손절/익절은 강제 — 무한 보유 방지
    stop_loss: float | None = 0.05      # -5%
    take_profit: float | None = 0.15    # +15%
    max_holding_bars: int | None = 60   # 최대 보유 봉


# 스캐너 기본 유니버스 (index.html SCR_UNIVERSE와 동일 컨셉)
UNIVERSE = {
    "us": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "NFLX",
        "CRM", "ADBE", "ORCL", "QCOM", "INTC", "MU", "PLTR", "SMCI", "ARM", "COIN",
        "JPM", "BAC", "GS", "MS", "V", "MA", "XOM", "CVX", "UNH", "LLY",
        "COST", "WMT", "HD", "MCD", "NKE", "SBUX", "DIS", "BA", "CAT", "GE",
        "SPY", "QQQ", "IWM", "DIA", "SOXL", "TQQQ", "ARKK", "XLE", "XLF", "XLK",
    ],
    "kr": [
        "005930", "000660", "373220", "207940", "005380", "005490", "051910", "006400",
        "035420", "035720", "068270", "105560", "055550", "086790", "316140", "000270",
        "012330", "028260", "010130", "011200", "009150", "018260", "032830", "003550",
        "034730", "015760", "017670", "030200", "096770", "010950", "042660", "064350",
        "247540", "086520", "091990", "196170", "145020", "285130", "112040", "263750",
    ],
}
