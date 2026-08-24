"""데이터 로더 — 국내(FinanceDataReader/pykrx)와 해외(yfinance)를 한 인터페이스로 통합.

모든 로더는 동일한 스키마를 반환한다:
    DatetimeIndex + [open, high, low, close, volume]  (float, 결측 제거)
"""
from __future__ import annotations

import re
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR

_COLS = ["open", "high", "low", "close", "volume"]
_KR_TICKER = re.compile(r"^\d{6}$")
_CRYPTO = re.compile(r"^[A-Z]{2,10}[-/](USD|USDT|KRW)$")


class DataError(RuntimeError):
    """데이터를 어떤 소스에서도 받지 못했을 때."""


def detect_market(ticker: str) -> str:
    """티커 모양으로 시장을 추정한다. '005930' → kr, 'AAPL' → us, 'BTC-USD' → crypto."""
    t = ticker.strip().upper()
    if _CRYPTO.match(t):
        return "crypto"
    if _KR_TICKER.match(t.split(".")[0]):
        return "kr"
    if t.endswith((".KS", ".KQ")):
        return "kr"
    return "us"


def _normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """소스별로 제각각인 컬럼명을 공통 스키마로 정리한다."""
    if df is None or len(df) == 0:
        raise DataError(f"{source}: 빈 응답")

    # yfinance는 단일 종목에도 MultiIndex 컬럼을 줄 때가 있다
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        # ('Close','AAPL') 형태와 ('AAPL','Close') 형태 모두 대응
        if any(str(c).lower() in ("open", "high", "low", "close", "volume") for c in lvl0):
            df = df.droplevel(-1, axis=1)
        else:
            df = df.droplevel(0, axis=1)

    rename = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_")
        if key in ("open", "시가"):
            rename[c] = "open"
        elif key in ("high", "고가"):
            rename[c] = "high"
        elif key in ("low", "저가"):
            rename[c] = "low"
        elif key in ("close", "adj_close", "종가"):
            rename[c] = "close"
        elif key in ("volume", "거래량"):
            rename[c] = "volume"
    df = df.rename(columns=rename)

    # 중복 컬럼(close/adj_close가 함께 매핑된 경우) 제거 — 뒤쪽 우선
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    missing = [c for c in _COLS if c not in df.columns]
    if missing:
        raise DataError(f"{source}: 컬럼 누락 {missing} (실제: {list(df.columns)})")

    out = df[_COLS].apply(pd.to_numeric, errors="coerce")
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # 거래 정지일(가격 0/NaN) 제거
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[out["close"] > 0]
    out["volume"] = out["volume"].fillna(0.0)
    out.index.name = "date"
    return out.astype(float)


# ── 소스별 로더 ────────────────────────────────────────────────

def _from_yfinance(ticker: str, start, end, interval: str) -> pd.DataFrame:
    import yfinance as yf

    iv = {"D": "1d", "W": "1wk", "M": "1mo"}.get(interval, interval)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(
            ticker, start=start, end=end, interval=iv,
            auto_adjust=True, progress=False, threads=False,
        )
    return _normalize(df, "yfinance")


def _from_fdr(ticker: str, start, end, interval: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    df = fdr.DataReader(ticker, start, end)
    out = _normalize(df, "FinanceDataReader")
    return _resample(out, interval)


def _from_pykrx(ticker: str, start, end, interval: str) -> pd.DataFrame:
    """pykrx 1.2.x부터 KRX 계정이 필요하다.

    환경변수 KRX_ID / KRX_PW를 설정해야 동작한다. 없으면 빈 응답이 오고
    load()가 다음 소스로 넘어간다.
    """
    from pykrx import stock

    code = ticker.split(".")[0]
    fmt = "%Y%m%d"
    df = stock.get_market_ohlcv(
        pd.Timestamp(start).strftime(fmt), pd.Timestamp(end).strftime(fmt), code
    )
    out = _normalize(df, "pykrx")
    return _resample(out, interval)


def _resample(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """일봉을 주봉/월봉으로 집계 (index.html의 aggregate()와 동일 규칙)."""
    if interval in ("D", "1d", None):
        return df
    rule = {"W": "W-FRI", "M": "ME"}.get(interval, interval)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna(subset=["close"])


# ── 캐시 ──────────────────────────────────────────────────────

def _cache_path(ticker: str, interval: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", ticker)
    return CACHE_DIR / f"{safe}_{interval}.csv"


def _read_cache(path: Path, max_age_hours: float) -> pd.DataFrame | None:
    if not path.exists():
        return None
    age = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).total_seconds()
    if age > max_age_hours * 3600:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df if len(df) else None
    except Exception:
        return None


def load_csv(path: str | Path, interval: str = "D") -> pd.DataFrame:
    """로컬 CSV/부분 파일에서 OHLCV를 읽는다.

    네트워크가 막힌 환경이나 증권사에서 내려받은 데이터를 쓸 때의 경로다.
    첫 컬럼이 날짜여야 하고, 컬럼명은 한글(시가/고가/저가/종가/거래량)과
    영문(open/high/low/close/volume) 모두 인식한다.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _resample(_normalize(df, f"csv:{path}"), interval)


# ── 공개 API ───────────────────────────────────────────────────

def load(
    ticker: str,
    start: str | date | None = None,
    end: str | date | None = None,
    interval: str = "D",
    source: str = "auto",
    use_cache: bool = True,
    cache_hours: float = 12.0,
) -> pd.DataFrame:
    """OHLCV를 불러온다. 소스가 실패하면 다음 소스로 자동 폴백한다.

    >>> df = load("AAPL", start="2023-01-01")
    >>> df = load("005930", start="2023-01-01")   # 삼성전자
    """
    ticker = ticker.strip()

    # 티커 자리에 CSV 경로를 주면 로컬 파일에서 읽는다 (오프라인/증권사 데이터)
    if ticker.lower().endswith(".csv") and Path(ticker).exists():
        df = load_csv(ticker, interval)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize() + timedelta(days=1)
    start = pd.Timestamp(start) if start else end - timedelta(days=365 * 3)

    cache = _cache_path(ticker, interval)
    if use_cache:
        cached = _read_cache(cache, cache_hours)
        if cached is not None:
            sliced = cached.loc[(cached.index >= start) & (cached.index <= end)]
            if len(sliced) > 20:
                return sliced

    market = detect_market(ticker) if source == "auto" else source
    if market == "kr":
        chain = [_from_fdr, _from_pykrx, _from_yfinance]
    elif market == "crypto":
        chain = [_from_yfinance, _from_fdr]
    else:
        chain = [_from_yfinance, _from_fdr]

    errors = []
    for fn in chain:
        try:
            df = fn(ticker, start, end, interval)
            if len(df) < 20:
                errors.append(f"{fn.__name__}: {len(df)}봉뿐")
                continue
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache)
            return df
        except Exception as exc:  # 소스 하나가 죽어도 전체는 계속
            errors.append(f"{fn.__name__}: {type(exc).__name__} {exc}")

    raise DataError(f"'{ticker}' 로드 실패\n  " + "\n  ".join(errors))


def load_many(
    tickers: list[str], start=None, end=None, interval: str = "D", **kw
) -> dict[str, pd.DataFrame]:
    """여러 종목을 한 번에. 실패한 종목은 조용히 건너뛴다."""
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            # CSV 경로로 넘어온 경우 파일명만 라벨로 쓴다
            label = Path(t).stem if t.lower().endswith(".csv") else t
            out[label] = load(t, start, end, interval, **kw)
        except Exception:
            continue
    return out


def close_matrix(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{티커: OHLCV} → 종가 매트릭스 (포트폴리오 최적화 입력)."""
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame({t: df["close"] for t, df in frames.items()}).sort_index()
