"""시그널 생성 — index.html의 detectSig / scoreStock 규칙을 벡터화해 옮겼다.

모든 규칙은 "그 봉이 닫힌 뒤"에 판정된다. 실제 진입은 backtest에서
다음 봉 시가로 처리하므로 미래참조(look-ahead)가 생기지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as I

__all__ = ["SignalParams", "generate", "score"]


@dataclass
class SignalParams:
    """스윕/최적화 대상 파라미터. 기본값은 index.html과 동일하다."""

    rsi_oversold: float = 30.0
    rsi_overbought: float = 65.0
    vol_spike: float = 1.8          # 거래량 배수 하한
    vol_strong: float = 2.5
    adx_trend: float = 20.0         # 이 위면 추세장으로 본다
    trend_tol: float = 0.03         # MA50 대비 허용 오차
    avwap_band: float = 0.015       # AVWAP 근접 판정 폭


def generate(ind: pd.DataFrame, p: SignalParams | None = None) -> pd.DataFrame:
    """지표 프레임 → 매수/매도 시그널 프레임.

    반환 컬럼: buy, sell, buy_reason, sell_reason, buy_strength, sell_strength
    """
    p = p or SignalParams()
    c, o, h, l = ind["close"], ind["open"], ind["high"], ind["low"]
    r = ind["rsi"]
    vr = ind["vol_ratio"].fillna(1.0)
    s50, s200 = ind["sma50"], ind["sma200"]
    bu, bl = ind["bb_up"], ind["bb_lo"]
    macd_l, macd_s = ind["macd"], ind["macd_signal"]
    adx_v = ind["adx"]
    av = ind["avwap"]

    up_bar = c > o
    dn_bar = c < o
    trending = adx_v > p.adx_trend
    up_trend = c > s50 * (1 - p.trend_tol)
    dn_trend = c < s50 * (1 + p.trend_tol)
    macd_cross_up = (macd_l.shift(1) < macd_s.shift(1)) & (macd_l > macd_s)
    macd_cross_dn = (macd_l.shift(1) > macd_s.shift(1)) & (macd_l < macd_s)

    # ── 매수 규칙 ──
    buy_rules = {
        "RSI 극단 과매도 반등": (r < p.rsi_oversold) & up_bar & (vr > p.vol_spike),
        "골든크로스": (s50.shift(1) <= s200.shift(1)) & (s50 > s200),
        "MACD 상향 + 추세정렬": macd_cross_up & (r < 55) & (vr > p.vol_spike) & up_trend & trending,
        "BB 하단 강세반등": (l < bl * 1.01) & up_bar & (vr > p.vol_strong),
        "RSI+MACD 복합매수": (r < 35) & macd_cross_up & up_bar & (vr > 1.5),
        "AVWAP 지지반등": ((c - av).abs() / av < p.avwap_band) & (c > av) & up_bar & (vr > 2.0),
        "AVWAP 상향돌파": (c > av) & (c.shift(1) <= av.shift(1)) & (vr > 2.0),
        "스퀴즈 상방분출": ind["sqz_fired"] & (ind["sqz_mom"] > 0),
        "RSI 강세 다이버전스": ind["div_bull"] & (r < 50),
        "유동성 스윕 반등": ind["sweep_bull"] & (vr > 1.5),
    }
    # ── 매도 규칙 ──
    sell_rules = {
        "데스크로스": (s50.shift(1) >= s200.shift(1)) & (s50 < s200),
        "MACD 하향 + 추세정렬": macd_cross_dn & (r > 55) & (vr > p.vol_spike) & dn_trend & trending,
        "BB 상단 거래량거부": (h > bu) & dn_bar & (vr > p.vol_strong),
        "RSI+MACD 복합매도": (r > p.rsi_overbought) & (r < r.shift(1) - 3) & (macd_l < macd_s) & dn_trend & trending,
        "AVWAP 저항반락": ((c - av).abs() / av < p.avwap_band) & (c < av) & dn_bar & (vr > 2.0),
        "MA200 하향이탈": (c.shift(1) >= s200) & (c < s200),
        "대량거래 급락": (vr > p.vol_strong) & dn_bar & ((o - c) / o > 0.03),
        "RSI 약세 다이버전스": ind["div_bear"] & (r > 60),
    }

    def _collapse(rules: dict[str, pd.Series]) -> tuple[pd.Series, pd.Series, pd.Series]:
        mat = pd.DataFrame({k: v.fillna(False).to_numpy(dtype=bool) for k, v in rules.items()},
                           index=ind.index)
        hit = mat.any(axis=1)
        count = mat.sum(axis=1)
        reason = mat.apply(lambda row: " + ".join(mat.columns[row.to_numpy()]), axis=1)
        return hit, reason, count

    buy, buy_reason, buy_n = _collapse(buy_rules)
    sell, sell_reason, sell_n = _collapse(sell_rules)

    # 같은 봉에 매수/매도가 동시에 뜨면 규칙이 더 많이 걸린 쪽을 택한다
    conflict = buy & sell
    buy = buy & ~(conflict & (sell_n > buy_n))
    sell = sell & ~(conflict & (buy_n >= sell_n))

    return pd.DataFrame(
        {
            "buy": buy, "sell": sell,
            "buy_reason": buy_reason.where(buy, ""),
            "sell_reason": sell_reason.where(sell, ""),
            "buy_strength": buy_n.where(buy, 0).astype(int),
            "sell_strength": sell_n.where(sell, 0).astype(int),
        },
        index=ind.index,
    )


def score(ind: pd.DataFrame) -> dict:
    """스크리너 점수 — index.html scoreStock 이식. 마지막 봉 기준."""
    if len(ind) < 30:
        return {"score": 0, "reasons": []}

    last = ind.iloc[-1]
    pts, why = 0, []

    if bool(ind["sqz_fired"].iloc[-1]):
        pts += 10
        why.append("🚀스퀴즈" + ("↑" if last["sqz_mom"] > 0 else "↓"))
    elif last["sqz_on"] is True:
        pts += 5
        why.append("⚡압축중")

    if pd.notna(last["avwap"]) and last["avwap"] > 0:
        away = (last["close"] - last["avwap"]) / last["avwap"] * 100
        if 0 < away < 8:
            pts += 3; why.append(f"AVWAP+{away:.1f}%")
        elif away >= 8:
            pts += 1; why.append(f"AVWAP+{away:.1f}%")
        elif -3 < away <= 0:
            pts += 2; why.append(f"AVWAP{away:.1f}%")

    if ind["ofd"].tail(5).sum() > 0:
        pts += 2; why.append("OFD+")
    if len(ind) > 6 and ind["close"].iloc[-1] < ind["close"].iloc[-6] \
            and ind["ofd_cum"].iloc[-1] > ind["ofd_cum"].iloc[-6]:
        pts += 4; why.append("OFD다이버전스")

    if bool(ind["div_bull"].tail(15).any()):
        pts += 5; why.append("RSI강세다이버전스")
    rv = last["rsi"]
    if pd.notna(rv):
        if rv < 35:
            pts += 3; why.append(f"RSI{rv:.0f}과매도")
        elif 50 < rv < 68:
            pts += 1; why.append(f"RSI{rv:.0f}")

    if bool(ind["sweep_bull"].tail(10).any()):
        pts += 3; why.append("스윕↑")

    vr = last["vol_ratio"]
    if pd.notna(vr) and vr > 2 and last["close"] > last["open"]:
        pts += 2; why.append(f"거래량{vr:.1f}x")

    return {
        "score": pts,
        "reasons": why,
        "rsi": float(rv) if pd.notna(rv) else None,
        "adx": float(last["adx"]) if pd.notna(last["adx"]) else None,
        "close": float(last["close"]),
    }
