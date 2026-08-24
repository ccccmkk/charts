"""charts-quant — 브라우저 차트 앱(index.html)의 퀀트 엔진을 Python으로 옮긴 패키지.

빠른 사용법
-----------
    from quant import data, indicators, signals, backtest
    from quant.config import BacktestConfig, US_COSTS

    df  = data.load("AAPL", start="2022-01-01")
    ind = indicators.compute_all(df)
    sig = signals.generate(ind)
    res = backtest.run(df, sig, BacktestConfig(costs=US_COSTS))
    print(res.summary())

CLI
---
    python -m quant backtest AAPL
    python -m quant optimize 005930 --walk-forward
    python -m quant scan --universe kr
    python -m quant portfolio AAPL MSFT NVDA --method hrp
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import backtest, config, data, indicators, metrics, optimize, portfolio, screener, signals  # noqa: E402,F401

__all__ = [
    "backtest", "config", "data", "indicators", "metrics",
    "optimize", "portfolio", "screener", "signals", "__version__",
]
