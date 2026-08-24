"""유니버스 스캐너 — index.html의 runScan을 서버사이드로 옮긴 것.

브라우저는 CORS와 탭 하나의 동시 연결 제한 때문에 수십 종목이 한계였다.
여기서는 스레드로 병렬 수집하면서 종목마다 백테스트까지 돌린다.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import backtest as B, data as D, indicators as I, signals as S
from .config import UNIVERSE, BacktestConfig, KR_COSTS, US_COSTS

__all__ = ["scan"]


def _one(ticker: str, start, interval: str, run_backtest: bool) -> dict | None:
    try:
        df = D.load(ticker, start=start, interval=interval)
        if len(df) < 60:
            return None
        ind = I.compute_all(df)
        row = S.score(ind)
        row["ticker"] = ticker

        if run_backtest:
            is_kr = D.detect_market(ticker) == "kr"
            cfg = BacktestConfig(costs=KR_COSTS if is_kr else US_COSTS, interval=interval)
            res = B.run(df, S.generate(ind), cfg, apply_tax=is_kr)
            ts, perf = res.stats, res.performance
            row.update({
                "n_trades": ts["n_trades"], "win_rate": ts["win_rate"],
                "profit_factor": ts["profit_factor"], "expectancy": ts["expectancy"],
                "sharpe": perf.sharpe, "cagr": perf.cagr, "max_dd": perf.max_drawdown,
                "alpha": perf.alpha,
            })
        return row
    except Exception:
        return None


def scan(
    universe: str | list[str] = "us",
    start: str | None = None,
    interval: str = "D",
    run_backtest: bool = True,
    workers: int = 8,
    min_score: int = 0,
    progress: bool = True,
) -> pd.DataFrame:
    """유니버스를 훑어 점수순으로 정렬한 표를 만든다.

    universe: 'us' | 'kr' | 'all' | 티커 리스트
    """
    if isinstance(universe, str):
        tickers = (UNIVERSE["us"] + UNIVERSE["kr"]) if universe == "all" else UNIVERSE[universe]
    else:
        tickers = list(universe)

    start = start or (pd.Timestamp.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    rows, done = [], 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, t, start, interval, run_backtest): t for t in tickers}
        for fut in as_completed(futs):
            done += 1
            if progress and done % 10 == 0:
                print(f"  스캔 {done}/{len(tickers)}", flush=True)
            r = fut.result()
            if r and r["score"] >= min_score:
                rows.append(r)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["reasons"] = df["reasons"].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    front = [c for c in ["ticker", "score", "close", "rsi", "adx", "reasons"] if c in df]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest].sort_values("score", ascending=False).reset_index(drop=True)
