"""커맨드라인 진입점 — python -m quant <명령>."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import backtest as B, data as D, indicators as I, metrics as M
from . import optimize as O, portfolio as PF, screener as SC, signals as S
from .config import BacktestConfig, KR_COSTS, NO_COSTS, US_COSTS


def _costs_for(ticker: str, mode: str):
    if mode == "none":
        return NO_COSTS, False
    is_kr = D.detect_market(ticker) == "kr"
    return (KR_COSTS if is_kr else US_COSTS), is_kr


def _load(args) -> pd.DataFrame:
    if getattr(args, "csv", None):
        df = D.load_csv(args.csv, interval=args.interval)
        if args.start:
            df = df[df.index >= pd.Timestamp(args.start)]
        if args.end:
            df = df[df.index <= pd.Timestamp(args.end)]
        return df
    return D.load(args.ticker, start=args.start, end=args.end, interval=args.interval)


# ── 명령별 구현 ────────────────────────────────────────────────

def cmd_backtest(args) -> int:
    df = _load(args)
    ind = I.compute_all(df)
    sig = S.generate(ind)
    costs, tax = _costs_for(args.ticker, args.costs)
    cfg = BacktestConfig(
        initial_capital=args.capital, costs=costs, interval=args.interval,
        stop_loss=args.stop, take_profit=args.target, max_holding_bars=args.max_bars,
    )
    res = B.run(df, sig, cfg, apply_tax=tax)

    print(f"\n■ {args.ticker}  ({args.interval}봉, 비용 {args.costs})")
    print(res.summary())

    if len(res.trades) and args.show_trades:
        t = res.trades.copy()
        t["entry_date"] = pd.to_datetime(t["entry_date"]).dt.date
        t["exit_date"] = pd.to_datetime(t["exit_date"]).dt.date
        cols = ["entry_date", "exit_date", "entry_price", "exit_price", "pnl_pct", "bars", "exit_reason"]
        print("\n■ 체결 내역")
        print(t[cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    if args.out:
        res.equity.to_csv(args.out)
        print(f"\n자산곡선 저장 → {args.out}")
    return 0


def cmd_optimize(args) -> int:
    df = _load(args)
    costs, tax = _costs_for(args.ticker, args.costs)
    cfg = BacktestConfig(initial_capital=args.capital, costs=costs, interval=args.interval,
                         stop_loss=args.stop, take_profit=args.target)

    if args.walk_forward:
        print(f"\n■ {args.ticker} 워크포워드 검증 ({args.splits}구간)")
        wf = O.walk_forward(df, config=cfg, n_splits=args.splits, apply_tax=tax)
        if not wf["folds"]:
            print(wf["verdict"]); return 1
        t = wf["table"].copy()
        t["params"] = t["params"].apply(lambda d: " ".join(f"{k}={v}" for k, v in d.items()))
        print(t[["fold", "test_period", "params", "is_sharpe", "oos_sharpe",
                 "oos_return", "oos_trades"]].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
        print(f"\n  in-sample 평균 샤프 : {wf['is_sharpe_mean']:.2f}")
        print(f"  out-of-sample 평균  : {wf['oos_sharpe_mean']:.2f}")
        print(f"  성능 저하율         : {wf['decay']:.0%}")
        print(f"  OOS PSR             : {wf['oos_psr']:.1%}")
        print(f"\n  판정: {wf['verdict']}")
        return 0

    print(f"\n■ {args.ticker} 파라미터 스윕")
    res = O.sweep(df, config=cfg, metric=args.metric, apply_tax=tax)
    if res.empty:
        print("결과 없음"); return 1
    valid = res[res["valid"]]
    show = (valid if len(valid) else res).head(args.top)
    cols = [c for c in show.columns if c != "valid"]
    print(show[cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    ds = res.attrs.get("deflated_sharpe")
    print(f"\n  탐색 조합 {res.attrs.get('n_trials')}개 / 유효 {len(valid)}개")
    if ds is not None and ds == ds:
        print(f"  Deflated Sharpe: {ds:.1%}  ← 시행횟수를 반영한 신뢰도")
        print("  " + ("⚠ 50% 미만이면 1등 조합도 우연일 가능성이 크다."
                      if ds < 0.5 else "✅ 탐색 편향을 감안해도 유의미하다."))
    print("\n  ※ 이 표는 in-sample이다. 반드시 --walk-forward로 재검증할 것.")
    return 0


def cmd_scan(args) -> int:
    uni = args.tickers if args.tickers else args.universe
    print(f"\n■ 스캔 시작: {uni if isinstance(uni, str) else f'{len(uni)}종목'}")
    df = SC.scan(uni, start=args.start, interval=args.interval,
                 run_backtest=not args.fast, workers=args.workers, min_score=args.min_score)
    if df.empty:
        print("결과 없음"); return 1
    print(f"\n■ 상위 {min(args.top, len(df))}종목 (총 {len(df)}개 수집)")
    print(df.head(args.top).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\n저장 → {args.out}")
    return 0


def cmd_portfolio(args) -> int:
    frames = D.load_many(args.tickers, start=args.start, interval=args.interval)
    if len(frames) < 2:
        print(f"최소 2종목이 필요하다 (수집 성공 {len(frames)}개)"); return 1
    px = D.close_matrix(frames).dropna()
    print(f"\n■ 포트폴리오 {len(px.columns)}종목  {px.index[0].date()} ~ {px.index[-1].date()}")

    if args.compare:
        print("\n■ 방법별 비교")
        print(PF.compare(px, max_weight=args.max_weight).to_string(
            index=False, float_format=lambda v: f"{v:,.2f}"))
        return 0

    w = PF.optimize(px, args.method, max_weight=args.max_weight)
    print(f"\n■ 비중 ({args.method}, 종목 상한 {args.max_weight:.0%})")
    for k, v in w.items():
        bar = "█" * int(v * 60)
        print(f"  {k:<10} {v:>6.2%}  {bar}")
    print(f"  유효 종목수 {1/(w**2).sum():.2f}")

    eq = PF.backtest_weights(px, w, rebalance=args.rebalance)
    perf = M.perf_stats(eq, "D", benchmark=px.mean(axis=1))
    print(f"\n■ {args.rebalance} 리밸런싱 성과")
    print(M.summary_table(perf, M.trade_stats(pd.DataFrame())))
    return 0


def cmd_ml(args) -> int:
    from . import ml as ML

    df = _load(args)
    costs, tax = _costs_for(args.ticker, args.costs)
    bcfg = BacktestConfig(initial_capital=args.capital, costs=costs, interval=args.interval,
                          stop_loss=args.stop, take_profit=args.target)
    cfg = ML.MLConfig(horizon=args.horizon, n_splits=args.splits,
                      prob_cut=args.prob_cut, threshold=args.label_threshold)

    print(f"\n■ {args.ticker} — {args.horizon}봉 뒤 상승 확률 (purged walk-forward {args.splits}구간)")
    try:
        r = ML.compare(df, cfg, bcfg, apply_tax=tax)
    except ValueError as e:
        print(f"  {e}"); return 1
    ev = r["eval"]

    print(f"\n  OOS 표본 {ev['n_oos']}봉 · {ev['folds']}구간")
    print(f"  AUC      {ev['auc']:.3f}   ← 0.5면 동전던지기, 0.55 넘어야 의미가 생긴다")
    print(f"  정확도   {ev['accuracy']:.1f}%")
    print(f"  기준선   {max(ev['base_rate'], 100-ev['base_rate']):.1f}%   (항상 한쪽으로만 찍었을 때)")
    print(f"  엣지     {ev['edge']:+.1f}%p")

    if ev["auc"] < 0.53:
        print("\n  ❌ 예측력이 사실상 없다. 이 종목·이 기간에서는 쓰지 마라.")
    elif ev["auc"] < 0.57:
        print("\n  △ 미약하다. 거래비용을 빼면 남지 않을 수 있다.")
    else:
        print("\n  ✅ 기준선을 넘는다. 단, 아래 백테스트까지 확인할 것.")

    print("\n■ 확률 구간별 실제 상승률 (아래로 갈수록 높아져야 한다)")
    print(ev["calibration"].to_string())

    print("\n■ 같은 구간 백테스트 (수수료·슬리피지 반영)")
    print(f"  {'전략':<20} {'수익률':>9} {'샤프':>7} {'MDD':>8} {'거래':>7}")
    for name, res in [(f"ML (확률>{cfg.prob_cut})", r["ml"]), ("규칙 기반 시그널", r["rule"])]:
        p, t = res.performance, res.stats
        print(f"  {name:<20} {p.total_return:>8.2f}% {p.sharpe:>7.2f} {p.max_drawdown:>7.1f}% {t['n_trades']:>6}회")
    print(f"  {'단순보유':<20} {r['buy_hold']:>8.2f}% {'-':>7} {'-':>8} {'-':>7}")

    if args.importance:
        print("\n■ 피처 중요도 상위 15 (해석용 · in-sample이라 성능 근거는 아니다)")
        print(ML.feature_importance(df, cfg, 15).to_string())

    print("\n  ※ 한 종목의 한 기간 결과다. 여러 종목에서 재현되지 않으면 우연이다.")
    return 0


# ── 파서 ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quant", description="charts 퀀트 엔진 — 백테스트 / 최적화 / 스캔 / 포트폴리오",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""예시:
  python -m quant backtest AAPL --start 2022-01-01 --show-trades
  python -m quant backtest 005930 --interval W
  python -m quant optimize AAPL --walk-forward
  python -m quant scan --universe kr --top 15
  python -m quant portfolio AAPL MSFT NVDA GOOGL --method hrp --compare
""")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, ticker=True):
        if ticker:
            sp.add_argument("ticker", nargs="?", default="LOCAL",
                            help="티커 (AAPL, 005930 …). --csv 사용 시 생략 가능")
        sp.add_argument("--csv", default=None,
                        help="로컬 CSV에서 읽기 (네트워크 대신). 첫 컬럼이 날짜")
        sp.add_argument("--start", default=None, help="시작일 YYYY-MM-DD")
        sp.add_argument("--end", default=None, help="종료일")
        sp.add_argument("--interval", default="D", choices=["D", "W", "M"], help="봉 주기")
        sp.add_argument("--capital", type=float, default=10_000_000, help="초기자금")
        sp.add_argument("--costs", default="auto", choices=["auto", "none"], help="거래비용")
        sp.add_argument("--stop", type=float, default=0.05, help="손절 (0.05=-5%%)")
        sp.add_argument("--target", type=float, default=0.15, help="익절 (0.15=+15%%)")

    bt = sub.add_parser("backtest", help="단일 종목 백테스트")
    common(bt)
    bt.add_argument("--max-bars", type=int, default=60, help="최대 보유 봉")
    bt.add_argument("--show-trades", action="store_true", help="체결 내역 출력")
    bt.add_argument("--out", default=None, help="자산곡선 CSV 경로")
    bt.set_defaults(func=cmd_backtest)

    op = sub.add_parser("optimize", help="파라미터 스윕 / 워크포워드")
    common(op)
    op.add_argument("--walk-forward", action="store_true", help="구간분할 OOS 검증")
    op.add_argument("--splits", type=int, default=4, help="워크포워드 구간 수")
    op.add_argument("--metric", default="sharpe", help="정렬 기준")
    op.add_argument("--top", type=int, default=10, help="상위 N개")
    op.set_defaults(func=cmd_optimize)

    sc = sub.add_parser("scan", help="유니버스 스캔")
    sc.add_argument("tickers", nargs="*", help="직접 지정 (없으면 --universe)")
    sc.add_argument("--universe", default="us", choices=["us", "kr", "all"])
    sc.add_argument("--start", default=None)
    sc.add_argument("--interval", default="D", choices=["D", "W", "M"])
    sc.add_argument("--top", type=int, default=20)
    sc.add_argument("--workers", type=int, default=8)
    sc.add_argument("--min-score", type=int, default=0)
    sc.add_argument("--fast", action="store_true", help="백테스트 생략(점수만)")
    sc.add_argument("--out", default=None, help="CSV 저장 경로")
    sc.set_defaults(func=cmd_scan)

    mlp = sub.add_parser("ml", help="머신러닝 상승확률 예측 (그래디언트 부스팅)")
    common(mlp)
    mlp.add_argument("--horizon", type=int, default=5, help="며칠 뒤를 예측할지")
    mlp.add_argument("--splits", type=int, default=5, help="워크포워드 구간 수")
    mlp.add_argument("--prob-cut", type=float, default=0.55, help="매수 판정 확률")
    mlp.add_argument("--label-threshold", type=float, default=0.0, help="상승 라벨 기준 수익률(%%)")
    mlp.add_argument("--importance", action="store_true", help="피처 중요도 출력")
    mlp.set_defaults(func=cmd_ml)

    pf = sub.add_parser("portfolio", help="포트폴리오 최적화")
    pf.add_argument("tickers", nargs="+")
    pf.add_argument("--start", default=None)
    pf.add_argument("--interval", default="D", choices=["D", "W", "M"])
    pf.add_argument("--method", default="hrp", choices=PF.METHODS)
    pf.add_argument("--max-weight", type=float, default=0.35, help="종목당 상한")
    pf.add_argument("--rebalance", default="ME", help="리밸런싱 주기 (ME/QE/YE)")
    pf.add_argument("--compare", action="store_true", help="모든 방법 비교")
    pf.set_defaults(func=cmd_portfolio)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except D.DataError as e:
        print(f"\n[데이터 오류] {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n중단됨", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
