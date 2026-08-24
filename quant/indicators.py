"""기술적 지표 — index.html의 JS 구현을 pandas로 이식했다.

같은 입력에 같은 값이 나오도록 세부 규칙(와일더 평활, 모집단 표준편차 등)을
JS 쪽과 일치시켰다. 검증은 tests/test_parity.py 참고.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma", "ema", "wilder", "bollinger", "rsi", "macd", "stoch", "atr",
    "supertrend", "obv", "adx", "anchored_vwap", "squeeze", "divergence",
    "liquidity_sweep", "order_flow_delta", "compute_all",
]


# ── 기본 이동평균 ──────────────────────────────────────────────

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """와일더 평활 — 첫 n개의 단순평균으로 시드 후 (prev*(n-1)+x)/n.

    pandas의 ewm(alpha=1/n)은 시드가 달라서 초기값이 어긋난다. JS 구현과
    맞추기 위해 시드를 명시적으로 넣는다.
    """
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) < n:
        return pd.Series(out, index=s.index)
    seed = np.nanmean(v[:n])
    out[n - 1] = seed
    prev = seed
    for i in range(n, len(v)):
        x = v[i]
        if np.isnan(x):
            out[i] = prev
            continue
        prev = (prev * (n - 1) + x) / n
        out[i] = prev
    return pd.Series(out, index=s.index)


# ── 오실레이터 / 밴드 ──────────────────────────────────────────

def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)          # JS와 동일하게 모집단 표준편차
    return pd.DataFrame({"bb_mid": mid, "bb_up": mid + k * sd, "bb_lo": mid - k * sd})


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    ag = wilder(gain.fillna(0.0), n)
    al = wilder(loss.fillna(0.0), n)
    rs = ag / al.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(al != 0, 100.0).where(ag.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def stoch(high, low, close, k: int = 14, d: int = 3) -> pd.DataFrame:
    hh = high.rolling(k).max()
    ll = low.rolling(k).min()
    kk = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return pd.DataFrame({"stoch_k": kk, "stoch_d": kk.rolling(d).mean()})


# ── 변동성 / 추세 ──────────────────────────────────────────────

def true_range(high, low, close) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def atr(high, low, close, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def supertrend(high, low, close, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    a = atr(high, low, close, period).fillna(0.0).to_numpy()
    h, l, c = (x.to_numpy(dtype=float) for x in (high, low, close))
    n = len(c)
    hl2 = (h + l) / 2
    bu, bl = hl2 + mult * a, hl2 - mult * a
    up = np.empty(n); dn = np.empty(n); direction = np.empty(n, dtype=int); st = np.empty(n)

    up[0], dn[0], direction[0], st[0] = bu[0], bl[0], 1, bl[0]
    for i in range(1, n):
        p_up, p_dn = up[i - 1], dn[i - 1]
        up[i] = bl[i] if bl[i] > p_dn else (max(bl[i], p_dn) if c[i - 1] > p_dn else bl[i])
        dn[i] = bu[i] if bu[i] < p_up else (min(bu[i], p_up) if c[i - 1] < p_up else bu[i])
        if direction[i - 1] == 1:
            direction[i] = -1 if c[i] < up[i] else 1
        else:
            direction[i] = 1 if c[i] > dn[i] else -1
        st[i] = up[i] if direction[i] == 1 else dn[i]
    return pd.DataFrame(
        {"st": st, "st_dir": direction, "st_up": up, "st_dn": dn}, index=close.index
    )


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume.fillna(0.0)).cumsum() + float(volume.iloc[0] or 0.0)


def adx(high, low, close, n: int = 14) -> pd.DataFrame:
    up_move = high.diff()
    dn_move = -low.diff()
    pdm = up_move.where((up_move > dn_move) & (up_move > 0), 0.0).fillna(0.0)
    ndm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0.0).fillna(0.0)
    tr = true_range(high, low, close)

    # 와일더 누적합 (평균이 아니라 합을 굴린다 — JS 구현과 동일)
    def _wsum(s: pd.Series) -> pd.Series:
        v = s.to_numpy(dtype=float)
        out = np.full(len(v), np.nan)
        if len(v) <= n:
            return pd.Series(out, index=s.index)
        acc = float(np.nansum(v[1:n + 1]))
        out[n] = acc
        for i in range(n + 1, len(v)):
            acc = acc - acc / n + v[i]
            out[i] = acc
        return pd.Series(out, index=s.index)

    str_, spdm, sndm = _wsum(tr), _wsum(pdm), _wsum(ndm)
    di_p = 100 * spdm / str_.replace(0, np.nan)
    di_n = 100 * sndm / str_.replace(0, np.nan)
    dx = 100 * (di_p - di_n).abs() / (di_p + di_n).replace(0, np.nan)
    return pd.DataFrame({"adx": wilder(dx.fillna(0.0), n), "di_plus": di_p, "di_minus": di_n})


# ── 스마트머니 계열 ────────────────────────────────────────────

def anchored_vwap(high, low, close, volume, anchor: int = 0) -> pd.Series:
    """앵커 시점부터 누적한 거래량가중평균가."""
    tp = (high + low + close) / 3
    v = volume.fillna(0.0)
    out = pd.Series(np.nan, index=close.index)
    cvp = (tp * v).iloc[anchor:].cumsum()
    cv = v.iloc[anchor:].cumsum()
    out.iloc[anchor:] = np.where(cv > 0, cvp / cv.replace(0, np.nan), tp.iloc[anchor:])
    return out


def squeeze(close, high, low, kc_mult: float = 1.5, kc_n: int = 20) -> pd.DataFrame:
    """TTM 스퀴즈 — 볼린저밴드가 켈트너채널 안으로 들어가면 압축(변동성 수축)."""
    bb = bollinger(close, 20, 2.0)
    em = ema(close, kc_n)
    at = atr(high, low, close, kc_n)
    kc_u, kc_l = em + kc_mult * at, em - kc_mult * at
    valid = (bb["bb_up"].notna() & kc_u.notna()).to_numpy()
    raw = ((bb["bb_up"] <= kc_u) & (bb["bb_lo"] >= kc_l)).to_numpy()
    on_arr = np.where(valid, raw, False)

    hh = high.rolling(kc_n).max()
    ll = low.rolling(kc_n).min()
    mom = close - ((hh + ll) / 2 + em) / 2

    # 압축이 풀리는 순간(직전 True → 현재 False)이 실제 트리거
    prev_on = np.r_[False, on_arr[:-1]]
    prev_valid = np.r_[False, valid[:-1]]
    fired = prev_on & ~on_arr & prev_valid & valid
    return pd.DataFrame(
        {
            "sqz_on": pd.Series(on_arr, index=close.index).where(valid).astype("boolean"),
            "sqz_mom": mom,
            "sqz_fired": pd.Series(fired, index=close.index),
        }
    )


def _pivots(v: np.ndarray, low_pivot: bool) -> np.ndarray:
    """3봉 기준 지역 극점 인덱스."""
    if low_pivot:
        m = (v[1:-1] <= v[:-2]) & (v[1:-1] <= v[2:])
    else:
        m = (v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:])
    return np.nonzero(m)[0] + 1


def divergence(close: pd.Series, indicator: pd.Series, lookback: int = 40) -> pd.DataFrame:
    """가격과 지표의 방향이 어긋나는 지점 (강세/약세 다이버전스)."""
    c = close.to_numpy(dtype=float)
    ind = indicator.to_numpy(dtype=float)
    bull = np.zeros(len(c), dtype=bool)
    bear = np.zeros(len(c), dtype=bool)

    lows = _pivots(c, True)
    highs = _pivots(c, False)
    for piv, arr, flag, cmp_price, cmp_ind in (
        (lows, bull, True, np.less, np.greater),
        (highs, bear, False, np.greater, np.less),
    ):
        for a, b in zip(piv[1:], piv[:-1]):
            if a - b > lookback:
                continue
            if np.isnan(ind[a]) or np.isnan(ind[b]):
                continue
            # 가격은 낮아졌는데(고점 갱신인데) 지표는 반대로 → 다이버전스
            if cmp_price(c[a], c[b]) and cmp_ind(ind[a], ind[b]):
                arr[a] = True
    return pd.DataFrame({"div_bull": bull, "div_bear": bear}, index=close.index)


def liquidity_sweep(high, low, close, open_, lookback: int = 20) -> pd.DataFrame:
    """유동성 스윕 — 직전 고/저점을 잠깐 깨고 되돌아오는 스탑헌팅 패턴."""
    ph = high.rolling(lookback).max().shift(1)
    pl = low.rolling(lookback).min().shift(1)
    rng = (high - low).replace(0, np.nan)

    bull_wick = (pl - low) / rng
    bear_wick = (high - ph) / rng
    bull = (low < pl) & (close > pl) & (close > open_) & (bull_wick > 0.25)
    bear = (high > ph) & (close < ph) & (close < open_) & (bear_wick > 0.25)
    return pd.DataFrame(
        {
            "sweep_bull": bull.to_numpy(dtype=bool),
            "sweep_bear": bear.to_numpy(dtype=bool),
        },
        index=close.index,
    )


def order_flow_delta(open_, high, low, close, volume) -> pd.DataFrame:
    """OHLCV만으로 매수/매도 압력을 근사한다.

    종가가 봉의 어디에 위치하는지(-1~+1)에 거래량을 곱해 델타를 만든다.
    틱 데이터 없이 쓰는 근사치이므로 절대값보다 방향과 누적 추세가 의미 있다.
    """
    rng = (high - low).replace(0, np.nan)
    pos = ((close - low) / rng * 2 - 1).fillna(0.0)
    delta = pos * volume.fillna(0.0)
    return pd.DataFrame({"ofd": delta, "ofd_cum": delta.cumsum()})


# ── 일괄 계산 ──────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV 하나로 모든 지표를 붙인 와이드 프레임을 만든다."""
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    parts = [
        df,
        pd.DataFrame({"sma20": sma(c, 20), "sma50": sma(c, 50), "sma200": sma(c, 200),
                      "ema20": ema(c, 20), "rsi": rsi(c), "atr": atr(h, l, c),
                      "obv": obv(c, v), "avwap": anchored_vwap(h, l, c, v)}),
        bollinger(c),
        macd(c),
        stoch(h, l, c),
        supertrend(h, l, c),
        adx(h, l, c),
        squeeze(c, h, l),
        liquidity_sweep(h, l, c, o),
        order_flow_delta(o, h, l, c, v),
    ]
    out = pd.concat(parts, axis=1)
    out = pd.concat([out, divergence(c, out["rsi"])], axis=1)
    out["vol_ratio"] = v / sma(v, 20)
    return out
