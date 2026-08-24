"""머신러닝 예측 — 그래디언트 부스팅으로 'N봉 뒤 상승 확률'을 추정한다.

설계 원칙
---------
1. 가격을 맞히려 하지 않는다. 방향의 **확률**만 낸다.
   가격 회귀는 R²가 0.99가 나와도 전부 "어제 종가"를 되뇌는 것뿐이다.

2. 피처는 전부 스케일 프리로 만든다. 종가 68,000을 그대로 넣으면
   트리가 "68,000 이상이면 상승" 같은 규칙을 외운다. 비율·z스코어만 쓴다.

3. 라벨은 **다음 봉 시가부터** 계산한다. 종가에 판단해서 종가에 체결한다는
   가정은 미래참조다. backtest.py의 체결 규칙과 일치시킨다.

4. 검증은 purged walk-forward. 시계열을 무작위로 나누면 미래가 학습에
   새어들어가 AUC가 가짜로 0.7까지 오른다. 학습·검증 경계에서 라벨이
   겹치는 구간(horizon 봉)을 잘라낸다(purging).

5. 항상 기준선과 비교한다. AUC 0.52는 "예측된다"가 아니라 "동전던지기"다.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest as B, indicators as I, metrics as M, signals as S
from .config import BacktestConfig

__all__ = ["MLConfig", "make_features", "make_target", "walk_forward_predict",
           "evaluate", "signals_from_proba", "feature_importance"]


@dataclass
class MLConfig:
    horizon: int = 5            # 며칠 뒤를 볼 것인가
    threshold: float = 0.0      # 이 수익률(%)을 넘으면 '상승'으로 라벨
    n_splits: int = 5           # 워크포워드 구간 수
    min_train: int = 250        # 최소 학습 봉 수
    prob_cut: float = 0.55      # 이 확률 이상이면 매수 시그널
    seed: int = 42
    params: dict = field(default_factory=dict)


# ── 피처 ───────────────────────────────────────────────────────

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV → 스케일 프리 피처. 전부 봉 t까지의 정보만 쓴다."""
    ind = I.compute_all(df)
    c, h, l, v = ind["close"], ind["high"], ind["low"], ind["volume"]
    eps = 1e-12

    f = pd.DataFrame(index=df.index)

    # 수익률 (여러 시계)
    for k in (1, 2, 3, 5, 10, 20, 60):
        f[f"ret{k}"] = c.pct_change(k)

    # 변동성 구조
    atr = ind["atr"]
    f["atr_pct"] = atr / c
    f["hl_pct"] = (h - l) / c
    f["ret_std20"] = c.pct_change().rolling(20).std()
    f["ret_std60"] = c.pct_change().rolling(60).std()
    f["vol_regime"] = f["ret_std20"] / (f["ret_std60"] + eps)

    # 이동평균 대비 위치 (가격 자체가 아니라 괴리율)
    for n in (20, 50, 200):
        f[f"px_sma{n}"] = c / ind[f"sma{n}"] - 1
    f["sma20_50"] = ind["sma20"] / ind["sma50"] - 1
    f["sma50_200"] = ind["sma50"] / ind["sma200"] - 1

    # 오실레이터 (이미 0~100이라 그대로 스케일 정규화만)
    f["rsi"] = ind["rsi"] / 100
    f["rsi_chg"] = ind["rsi"].diff(3) / 100
    f["stoch_k"] = ind["stoch_k"] / 100
    f["adx"] = ind["adx"] / 100
    f["di_diff"] = (ind["di_plus"] - ind["di_minus"]) / 100

    # MACD·볼린저는 가격으로 나눠 비율화
    f["macd_hist"] = ind["macd_hist"] / c
    f["macd_diff"] = (ind["macd"] - ind["macd_signal"]) / c
    bb_w = (ind["bb_up"] - ind["bb_lo"]).replace(0, np.nan)
    f["bb_pos"] = (c - ind["bb_lo"]) / bb_w
    f["bb_width"] = bb_w / ind["bb_mid"]

    # 거래량·수급
    f["vol_ratio"] = ind["vol_ratio"]
    f["vol_z"] = (v - v.rolling(60).mean()) / (v.rolling(60).std() + eps)
    ofd = ind["ofd"]
    f["ofd_z"] = (ofd - ofd.rolling(20).mean()) / (ofd.rolling(20).std() + eps)
    f["ofd_sum5"] = ofd.rolling(5).sum() / (v.rolling(5).sum() + eps)

    # AVWAP·SuperTrend·스퀴즈
    f["px_avwap"] = c / ind["avwap"] - 1
    f["st_dir"] = ind["st_dir"]
    f["sqz_on"] = ind["sqz_on"].astype("float64").fillna(0.0)
    f["sqz_mom"] = ind["sqz_mom"] / c

    # 이벤트성 피처 (최근 발생 여부)
    f["sweep_bull5"] = ind["sweep_bull"].rolling(5).sum()
    f["sweep_bear5"] = ind["sweep_bear"].rolling(5).sum()
    f["div_bull10"] = ind["div_bull"].rolling(10).sum()
    f["div_bear10"] = ind["div_bear"].rolling(10).sum()

    # 달력 효과
    f["dow"] = df.index.dayofweek
    f["month"] = df.index.month

    return f.replace([np.inf, -np.inf], np.nan)


def make_target(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0) -> pd.Series:
    """봉 t에서 판단 → t+1 시가 진입 → t+1+horizon 시가 청산 수익률의 부호.

    종가 기준으로 라벨을 만들면 "종가를 보고 종가에 산다"가 되어 미래참조다.
    """
    o = df["open"]
    entry = o.shift(-1)                       # 다음 봉 시가에 진입
    exit_ = o.shift(-(1 + horizon))           # horizon 봉 뒤 시가에 청산
    fwd = (exit_ / entry - 1) * 100
    return (fwd > threshold).astype("float64").where(fwd.notna())


# ── 모델 ───────────────────────────────────────────────────────

def _make_model(cfg: MLConfig, n_train: int):
    """LightGBM이 있으면 쓰고, 없으면 sklearn으로 대체한다.

    표본이 1천 개 남짓이라 트리를 얕게, 규제를 세게 건다. 기본값 그대로
    쓰면 학습셋을 통째로 외운다.
    """
    p = dict(
        n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=4,
        min_child_samples=max(20, n_train // 40),
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, random_state=cfg.seed, verbose=-1,
    )
    p.update(cfg.params)
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**p)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=p["n_estimators"], learning_rate=p["learning_rate"],
            max_leaf_nodes=p["num_leaves"], max_depth=p["max_depth"],
            min_samples_leaf=p["min_child_samples"],
            l2_regularization=p["reg_lambda"], random_state=cfg.seed,
        )


def walk_forward_predict(df: pd.DataFrame, cfg: MLConfig | None = None) -> pd.DataFrame:
    """Purged walk-forward. 각 구간은 그 이전 데이터만으로 학습한다.

    반환: index=날짜, [proba, y_true, fold]  — 전부 out-of-sample
    """
    cfg = cfg or MLConfig()
    X = make_features(df)
    y = make_target(df, cfg.horizon, cfg.threshold)

    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    n = len(X)
    if n < cfg.min_train + 60:
        raise ValueError(f"표본 부족: 유효 {n}봉 (최소 {cfg.min_train + 60}봉 필요)")

    fold_size = (n - cfg.min_train) // cfg.n_splits
    if fold_size < 20:
        raise ValueError(f"구간이 너무 작다 ({fold_size}봉). n_splits를 줄여라.")

    rows = []
    for k in range(cfg.n_splits):
        test_lo = cfg.min_train + k * fold_size
        test_hi = n if k == cfg.n_splits - 1 else test_lo + fold_size
        # purging — 학습 구간 끝의 라벨이 검증 구간과 겹치므로 잘라낸다
        train_hi = max(0, test_lo - cfg.horizon - 1)
        if train_hi < 100:
            continue

        Xtr, ytr = X.iloc[:train_hi], y.iloc[:train_hi]
        Xte, yte = X.iloc[test_lo:test_hi], y.iloc[test_lo:test_hi]
        if ytr.nunique() < 2 or len(Xte) == 0:
            continue

        model = _make_model(cfg, len(Xtr))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xtr, ytr)
            proba = model.predict_proba(Xte)[:, 1]

        rows.append(pd.DataFrame(
            {"proba": proba, "y_true": yte.to_numpy(), "fold": k + 1}, index=Xte.index))

    if not rows:
        raise ValueError("유효한 학습 구간을 만들지 못했다")
    return pd.concat(rows)


# ── 평가 ───────────────────────────────────────────────────────

def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC. 0.5면 동전던지기."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def evaluate(df: pd.DataFrame, cfg: MLConfig | None = None) -> dict:
    """OOS 예측력을 기준선과 나란히 놓는다.

    핵심은 모델 점수가 아니라 **기준선을 넘느냐**다.
    """
    cfg = cfg or MLConfig()
    oos = walk_forward_predict(df, cfg)
    y = oos["y_true"].to_numpy()
    p = oos["proba"].to_numpy()

    base_rate = float(y.mean())                       # 그냥 '항상 상승'의 적중률
    acc = float(((p > 0.5).astype(float) == y).mean())
    auc = _auc(y, p)

    # 확률이 높다고 실제로 더 잘 맞는가 (구간별 적중률)
    bins = pd.cut(oos["proba"], [0, .45, .5, .55, .6, 1.0], include_lowest=True)
    calib = oos.groupby(bins, observed=True).agg(
        n=("y_true", "size"), 실제상승률=("y_true", "mean"))
    calib["실제상승률"] = (calib["실제상승률"] * 100).round(1)

    return {
        "n_oos": int(len(oos)), "folds": int(oos["fold"].nunique()),
        "auc": auc, "accuracy": acc * 100, "base_rate": base_rate * 100,
        "edge": (acc - max(base_rate, 1 - base_rate)) * 100,
        "calibration": calib, "oos": oos,
    }


def signals_from_proba(df: pd.DataFrame, oos: pd.DataFrame, cfg: MLConfig) -> pd.DataFrame:
    """확률 → 매수/매도 시그널. backtest.run()에 그대로 넣을 수 있는 모양."""
    buy = pd.Series(False, index=df.index)
    sell = pd.Series(False, index=df.index)
    hi = oos["proba"] > cfg.prob_cut
    lo = oos["proba"] < (1 - cfg.prob_cut)
    buy.loc[hi[hi].index] = True
    sell.loc[lo[lo].index] = True
    return pd.DataFrame({
        "buy": buy, "sell": sell,
        "buy_reason": buy.map({True: f"ML 확률>{cfg.prob_cut:.2f}", False: ""}),
        "sell_reason": sell.map({True: f"ML 확률<{1-cfg.prob_cut:.2f}", False: ""}),
        "buy_strength": buy.astype(int), "sell_strength": sell.astype(int),
    })


def feature_importance(df: pd.DataFrame, cfg: MLConfig | None = None, top: int = 15) -> pd.Series:
    """전체 구간으로 한 번 학습해 어떤 피처를 보는지 확인한다.

    이건 해석용이지 성능 근거가 아니다(in-sample이다).
    """
    cfg = cfg or MLConfig()
    X = make_features(df)
    y = make_target(df, cfg.horizon, cfg.threshold)
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    model = _make_model(cfg, len(X))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return pd.Series(dtype=float)
    return pd.Series(imp, index=X.columns).sort_values(ascending=False).head(top)


def compare(df: pd.DataFrame, cfg: MLConfig | None = None,
            config: BacktestConfig | None = None, apply_tax: bool = False) -> dict:
    """ML 전략 vs 규칙 기반 시그널 vs 단순보유 — 같은 백테스트 엔진으로."""
    cfg = cfg or MLConfig()
    bcfg = config or BacktestConfig()
    ev = evaluate(df, cfg)

    ml_sig = signals_from_proba(df, ev["oos"], cfg)
    # ML은 OOS 구간만 유효하므로 규칙 기반도 같은 구간에서 비교한다
    span = df.loc[ev["oos"].index[0]:ev["oos"].index[-1]]
    ml_res = B.run(span, ml_sig.reindex(span.index).fillna(False), bcfg, apply_tax=apply_tax)

    ind = I.compute_all(span)
    rule_res = B.run(span, S.generate(ind), bcfg, apply_tax=apply_tax)

    bh = (float(span["close"].iloc[-1] / span["close"].iloc[0]) - 1) * 100
    return {"eval": ev, "ml": ml_res, "rule": rule_res, "buy_hold": bh, "span": span}
