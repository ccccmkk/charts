# charts

브라우저 기술적분석 차트 앱 + Python 퀀트 엔진.

| | 무엇 | 어디서 |
|---|---|---|
| **웹 앱** | 차트·시그널·스크리너·백테스트 애니메이션 | `index.html` (단일 파일, 정적 배포) |
| **퀀트 엔진** | 파라미터 최적화·워크포워드·포트폴리오·몬테카를로 | `quant/` (Python 패키지) |

웹 앱은 "보는" 도구, Python 패키지는 "검증하는" 도구다. 지표와 시그널 규칙은
양쪽이 동일하게 맞춰져 있다.

---

## 1. 웹 앱 (`index.html`)

의존성 없이 파일 하나만 열면 된다. Plotly만 CDN에서 받는다.

주요 기능: 캔들·볼린저·MACD·RSI·스토캐스틱·ATR·SuperTrend·OBV·ADX,
Anchored VWAP, TTM 스퀴즈, RSI 다이버전스, 유동성 스윕, 오더플로우 델타,
시그널 자동 탐지, 종목 스캐너, 시그널 기반 백테스트 애니메이션, Gemini 연동.

### 퀀트 리포트 패널

백테스트 애니메이션 모달의 **"퀀트 리포트 펼치기"** 버튼에서 볼 수 있다.

- 위험조정 수익: CAGR, Sharpe, Sortino, Calmar, 변동성, MDD와 그 지속기간, 시장노출도
- 거래 통계: Profit Factor, 손익비, 기대값, 최다 연승/연패, 평균 보유기간
- Buy & Hold 대비 알파 — 그냥 들고 있는 것보다 나은지
- 켈리 기준 베팅비중 (Full / Half)
- 몬테카를로 2000회 — 거래 순서를 재추출해 5% / 중앙 / 95% 구간과 손실 확률

---

## 2. Python 퀀트 엔진 (`quant/`)

브라우저에서 못 하는 일을 한다: 수백 조합 파라미터 탐색, 워크포워드
out-of-sample 검증, 다종목 포트폴리오 최적화, 과최적화 진단.

### 설치

```bash
pip install -r requirements.txt
# 또는 최소 구성 (지표·백테스트·성과지표만)
pip install numpy pandas
```

`numpy`와 `pandas`만 있으면 핵심 기능은 전부 돌아간다. 나머지는 선택이다.

### 빠른 사용

```python
from quant import data, indicators, signals, backtest
from quant.config import BacktestConfig, US_COSTS

df  = data.load("AAPL", start="2022-01-01")   # 국내는 data.load("005930")
ind = indicators.compute_all(df)              # 지표 38종
sig = signals.generate(ind)                   # 매수/매도 시그널
res = backtest.run(df, sig, BacktestConfig(costs=US_COSTS))

print(res.summary())
```

### CLI

```bash
python -m quant backtest AAPL --start 2022-01-01 --show-trades
python -m quant backtest 005930 --interval W
python -m quant optimize AAPL --walk-forward       # ← 실전 판단은 이걸로
python -m quant scan --universe kr --top 15
python -m quant portfolio AAPL MSFT NVDA GOOGL --method hrp --compare
```

네트워크가 막힌 환경이나 증권사에서 받은 데이터는 CSV로 넣는다:

```bash
python -m quant backtest --csv data/samsung.csv
python -m quant portfolio a.csv b.csv c.csv --compare
```

### 모듈

| 모듈 | 역할 |
|---|---|
| `data` | FinanceDataReader / pykrx / yfinance / CSV 통합 로더, 디스크 캐시, 자동 폴백 |
| `indicators` | 지표 38종. index.html의 JS 구현을 pandas로 이식 |
| `signals` | 매수/매도 규칙 18종 + 스크리너 점수 |
| `backtest` | 롱온리 엔진. **다음 봉 시가 체결**, 수수료·슬리피지·거래세 반영 |
| `metrics` | 성과지표 일체 + 몬테카를로 + 켈리 + PSR/DSR |
| `optimize` | 파라미터 스윕, 워크포워드 검증 |
| `portfolio` | equal / inverse_vol / risk_parity / min_var / max_sharpe / **HRP** |
| `screener` | 유니버스 병렬 스캔 |

---

## 3. 웹 앱과 다르게 만든 것

Python 엔진은 `index.html`의 백테스트를 그대로 옮기지 않았다. 두 군데를 고쳤다.

**1) 체결 시점 — 종가 → 다음 봉 시가**

웹 앱은 시그널이 뜬 봉의 **종가**에 체결한다. 종가는 그 봉이 닫혀야 알 수 있는
값이므로, "종가를 보고 종가에 산다"는 미래참조(look-ahead)다. 실전에서 재현할 수
없고 수익률을 체계적으로 부풀린다. Python 엔진은 다음 봉 시가에 체결한다.

**2) 거래비용 반영**

수수료·슬리피지·증권거래세를 뺀다. 회전율이 높은 전략일수록 차이가 커진다.

```python
from quant.config import KR_COSTS, US_COSTS, NO_COSTS
# KR: 수수료 0.015% + 슬리피지 0.1% + 매도세 0.18%
```

그래서 같은 종목·같은 시그널이라도 **Python 쪽 수익률이 더 낮게 나오는 것이 정상**이다.
낮은 쪽이 실전에 가깝다.

---

## 4. 과최적화 진단

이 패키지에서 가장 중요한 부분이다. 파라미터를 수백 번 뒤지면 반드시 좋아 보이는
조합이 나오지만 대부분은 우연이다. 세 가지 안전장치가 있다.

**PSR (Probabilistic Sharpe Ratio)** — 관측된 샤프가 진짜로 0을 넘을 확률.
수익률의 왜도·첨도까지 반영하므로 "샤프는 2인데 꼬리가 두꺼운" 전략을 걸러낸다.

**DSR (Deflated Sharpe Ratio)** — N번 탐색했다는 사실만큼 샤프를 할인한다.
`optimize` 결과에 자동으로 붙는다. 50% 미만이면 1등 조합도 운일 가능성이 크다.

**워크포워드** — 구간을 나눠 앞 구간에서 고른 파라미터를 다음 구간에 그대로 적용한다.
재탐색하지 않으므로 이 성적이 실전에 가장 가깝다.

```
$ python -m quant optimize SOME.csv --walk-forward

 fold           test_period          params              is_sharpe  oos_sharpe
    1 2019-12-10~2020-05-05 rsi_oversold=25 ...               0.47       -0.29
    2 2021-04-13~2021-09-07 rsi_oversold=25 ...               0.57        0.49
    3 2022-08-16~2023-01-10 rsi_oversold=25 ...               1.44       -0.46
    4 2023-12-19~2024-05-14 rsi_oversold=25 ...              -0.16       -0.50

  in-sample 평균 샤프 : 0.58
  out-of-sample 평균  : -0.19
  성능 저하율         : 133%

  판정: ❌ 실패 — out-of-sample 샤프가 0 이하다. 실전 투입 금지.
```

in-sample에서는 +13% 수익에 Profit Factor 1.27로 멀쩡해 보였던 전략이다.
**스윕 결과만 보고 판단하면 안 되는 이유가 이것이다.**

---

## 5. 데이터 소스 참고

| 소스 | 대상 | 비고 |
|---|---|---|
| FinanceDataReader | 국내 + 해외 | 국내 주가는 이쪽이 정확하다 |
| yfinance | 해외 | 재무제표·배당·옵션체인까지 |
| pykrx | 국내 (KRX 원본) | **1.2.x부터 KRX 계정 필요** — `KRX_ID` / `KRX_PW` 환경변수 |
| CSV | 무엇이든 | `--csv` 또는 `data.load_csv()` |

`data.load()`는 시장을 자동 판별해 소스를 고르고, 실패하면 다음 소스로 넘어간다.
결과는 `.cache/`에 12시간 캐시된다.

---

## 6. 테스트

```bash
python -m pytest tests/ -v
```

29개 테스트가 전부 합성 데이터로 돌아 네트워크가 필요 없다. 손으로 계산 가능한
값을 못 박아 두었다 — CAGR 폐형식, MDD 손검산, 켈리 공식, 익절 체결가, 손절 우선,
미래참조 검출(과거 지표가 미래 봉에 영향받지 않는지), 비용 단조성, 비중 합 = 1.

---

## 주의

백테스트 결과는 과거 데이터에 대한 시뮬레이션이며 미래 수익을 보장하지 않는다.
이 저장소는 투자 조언이 아니다.
