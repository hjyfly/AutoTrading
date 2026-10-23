"""매일 자동 실행. 작업 스케줄러가 평일 15:20 에 부른다.

1. 오늘이 거래일인지 확인한다(주말·공휴일이면 아무것도 하지 않는다).
2. 다른 종목(운영 종목 밖 레지스트리 종목 + 채택 분야 동료 종목 원본) 원본을 덧붙이고 정식 신호를 먼저 만든다(정시 실행에서만).
   운영 종목 쪽이 실패해도 빠지지 않게 따로 만든다.
3. 운영 종목 신호를 만든다: 운영 모델(최종, 뉴스 채점·기사별 추출은 참고용), 후보 모델·비교 묶음·하락 위험 비교 조건(비교용).
4. 실시간 성적표(운영 종목·다른 종목)를 다시 계산한다. 금요일에는 저장 데이터를 갱신(config.REFRESH_WEEKDAY)한 뒤
   운영 종목 조건 모니터링 리포트(config.MONITOR_WEEKDAY)를 다시 만든다. 뉴스 경보·계좌 줄도 붙인다(정시 실행에서만).
5. Gmail 로 결과를 보낸다. 실패하면 원인(과 2단계 다른 종목 신호)을 담은 메일을 보낸다.
6. 금요일 정시 실행이면 알림을 보낸 뒤 다른 종목 조건 모니터링을 종목마다 돈다.
날짜를 지정한 재실행은 운영 종목 신호와 성적표(그 날짜 전 확정분)만 다시 만든다.
실행: .venv\\Scripts\\python -m src.daily_run [--no-email] [--no-news] [YYYY-MM-DD]
"""
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd
import requests

from src import config, llm, notify, tools
from src.pipeline import experiment, live_signal, refresh_data, scoreboard

DISCLAIMER = "투자 권유가 아니라 모델 판단의 설명입니다. 매매는 직접 판단해 주세요."


def is_trading_day(today: pd.Timestamp) -> tuple[bool, str]:
    """주말이면 아니다. 평일이면 네이버 증권의 마지막 체결 시각이 오늘인지 본다. 확인에 실패하면 거래일로 본다."""
    if today.weekday() >= 5:
        return False, "주말"
    try:
        resp = requests.get(config.NAVER_BASIC_URL.format(code=config.MAIN_TICKER), headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        traded = pd.Timestamp(resp.json()["localTradedAt"]).tz_convert(config.KST).tz_localize(None).normalize()
        if traded < today:
            return False, f"오늘 체결이 없다(마지막 체결 {traded.date()}). 공휴일로 본다"
        return True, "거래일"
    except Exception as e:
        return True, f"거래일 확인 실패({type(e).__name__}), 평일이라 진행"


def compose(today: pd.Timestamp, out: dict, board_until: pd.Timestamp | None = None, others: list[str] | None = None) -> tuple[str, str]:
    """알림 제목과 본문을 만든다. others 는 run() 이 운영 종목 신호보다 먼저 만든 다른 종목 신호 요약이다(정시 실행에서만)."""
    sig = tools.signal_for(str(today.date()))
    judgement = tools.news_judgement(str(today.date()))
    final = sig.get("with_news", {}).get("signal") or sig.get("numeric", {}).get("signal", "없음")
    numeric = sig.get("numeric", {})
    subject = f"[AutoTrading] {today.date()} {final} (확신도 {numeric.get('confidence', 0):.0%})"
    action = config.DOWN_ACTION_LABELS[config.DOWN_ACTION]
    main = tools.ticker_name(config.MAIN_TICKER)
    lines = [f"{today.date()} {config.SIGNAL_CUTOFF_KST} 기준 종목별 {config.LABEL_HORIZON_DAYS}거래일 방향 신호 — 운영 종목 {main}",
             f"(모의 매매 규칙: 상승 = {main}, 보합 = 유지, 하락 = {action} · 매매 종목: {tools.role_names(config.TRADE_TICKERS)}, 주문은 사람이 직접)", "",
             f"■ 최종 신호: {final}"]
    if numeric:
        p = numeric["probabilities"]
        lines.append(f"  숫자 모델: {numeric['signal']} (상승 {p['상승']:.0%} / 보합 {p['보합']:.0%} / 하락 {p['하락']:.0%})")
    if "with_news" in sig:
        w = sig["with_news"]
        note = ('보정함' if w['adjusted'] else '보정 없음') if config.NEWS_ADJUST_IN_SIGNAL else '참고용(신호 보정 안 함)'
        lines.append(f"  뉴스 합의: {w['news_consensus']:+d} (확신도 {w['news_confidence']:.2f}) → {note}")
    lines.append(f"  뉴스 채점: {out.get('news_status', '-')}")
    if "final" in judgement:
        lines += ["", "■ 뉴스 판단 근거", f"  {judgement['final']['rationale']}",
                  "  핵심 기사: " + " / ".join(t for t in judgement["final"]["key_articles"] if t)]
    if "candidate" in sig:
        c = sig["candidate"]
        lines += ["", f"■ 비교용 후보 모델(A+B+D): {c['signal']} (확신도 {c['confidence']:.0%}) — 최종 신호 아님"]
    lines += ["", f"■ 기간별 참고 신호(11일차, 매매 기준은 {config.OPERATIONAL_HORIZON}거래일)"] + [f"  {x}" for x in horizon_lines(today)]
    scheduled = board_until is None  # 날짜를 지정한 재실행에서는 갱신·모니터링·다른 종목 신호·경보·계좌를 하지 않는다(그 날짜 뒤 정보가 섞이므로)
    board = scoreboard.build(board_until)
    lines += ["", "■ 실시간 성적표"] + [f"  {x}" for x in scoreboard.summary_lines(board)]
    if scheduled and today.weekday() == config.REFRESH_WEEKDAY:
        try:  # 주 1회 저장 데이터 갱신(새 확정 날짜만 덧붙임). 실패해도 신호 알림은 보낸다
            lines += ["", "■ 저장 데이터 주간 갱신"] + [f"  {x}" for x in refresh_data.summary_lines(refresh_data.run())]
        except Exception as e:
            lines += ["", f"■ 저장 데이터 갱신 실패: {type(e).__name__}: {e}"]
    if scheduled and (today.weekday() == config.MONITOR_WEEKDAY or not Path(config.MONITOR_REPORT_PATH).exists()):
        try:  # 주 1회 조건 모니터링. 실패해도 신호 알림은 보낸다
            summary = experiment.monitor()["summary"]
            lines += ["", "■ 조건 모니터링 (주간, 탐색 · 자세한 해석은 experiments_monitor.md)"] + [f"  {x}" for x in summary]
        except Exception as e:
            lines += ["", f"■ 조건 모니터링 실패: {type(e).__name__}: {e}"]
    if scheduled:
        lines += ["", f"■ 뉴스 참고 지표·하락 위험 경보 (9일차, 경보는 {config.DAY9_ALERT_STATUS})"] + [f"  {x}" for x in news_alert_lines(today)]
        lines += ["", "■ 실제 계좌(한국투자증권, 읽기 전용)"] + [f"  {x}" for x in account_lines()]
        lines += ["", f"■ 다른 종목 신호(성적표 판정 · 줄 끝은 쓴 알고리즘 · 매매 종목: {tools.role_names(config.TRADE_TICKERS)})"] + [f"  {x}" for x in (others or ["만들지 않음"])]
        lines += ["", "■ 다른 종목 실시간 성적표(SERVICE.md 5.6)"] + [f"  {x}" for x in scoreboard.ticker_lines(board)]
    warnings = [f"{k}: {v['note']}" for k, v in out.get("data_sources", {}).items() if "실패" in v["note"]]
    if warnings:
        lines += ["", "■ 데이터 경고"] + [f"  {w}" for w in warnings]
    lines += ["", f"LLM 누적 비용 ${llm.total_cost():.2f} / 한도 ${config.COST_LIMIT_POC_USD:.0f}", "", DISCLAIMER]
    return subject, "\n".join(lines)


def horizon_lines(today: pd.Timestamp) -> list[str]:
    """운영 종목의 기간별(1·3·10·20거래일) 참고 신호 한 줄씩. 신호가 없으면 이유를 적는다."""
    lines = []
    for n in config.HORIZONS:
        if n == config.OPERATIONAL_HORIZON:
            continue
        r = tools.signal_for(str(today.date()), None, n)
        x = r.get("numeric")
        label = f"{config.HORIZON_LABELS[n]}({n}거래일 ±{config.HORIZONS[n]:.1%})"
        if x:
            p = x["probabilities"]
            lines.append(f"{label}: {x['signal']} (상승 {p['상승']:.0%} / 보합 {p['보합']:.0%} / 하락 {p['하락']:.0%})")
        else:
            lines.append(f"{label}: 신호 없음")
    return lines


def setup_log(today: pd.Timestamp) -> logging.Logger:
    """data/logs/daily_YYYYMMDD.log 에 남긴다."""
    Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("daily_run")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(Path(config.LOG_DIR) / f"daily_{today:%Y%m%d}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def news_alert_lines(today: pd.Timestamp) -> list[str]:
    """오늘 기사 피처로 A1 하락 위험 경보를 판단하고, 경보면 기록하고 메일 문구를 돌려준다(DAY9_PLAN.md 3.2). 실패해도 알림은 보낸다."""
    from src.pipeline import news_signal
    try:
        articles = news_signal.load_article_features()
        if articles.empty or today not in set(articles["signal_date"]):
            return ["오늘 기사 피처가 없다(뉴스 채점 전이거나 실패)"]
        daily = news_signal.daily_features(articles)
        scores = pd.read_parquet(config.SCORES_PATH).set_index("signal_date")["news_downside_risk"]
        today_risk = tools.news_judgement(str(today.date())).get("final", {}).get("downside_risk")
        if today_risk is not None:
            scores.loc[today] = today_risk
        v1 = news_signal.v1_signal(today, tools.ticker_series(config.NEWS_TICKERS[0], "ohlcv", today)[0]["Close"].astype(float), articles)
        head = [f"뉴스 모델 V1(참고, 장부 V1 확인 통과): {v1['signal']} (상승−하락 확률 {v1['score']:+.2f})" if v1
                else "뉴스 모델 V1(참고): 학습 행이 모자라 없음"]
        alerts = news_signal.risk_alerts(articles, daily, scores)
        if today not in alerts.index:
            return head + ["오늘 경보 없음"]
        alert = alerts.loc[today].to_dict()
        news_signal.record_alert(today, alert)
        return head + news_signal.render_alert(today, alert, config.DAY9_ALERT_STATUS).split("\n")
    except Exception as e:  # 경보 계산 실패가 신호 알림을 막지 않게 한다
        return [f"경보 계산 실패: {type(e).__name__}: {str(e)[:80]}"]


def account_lines() -> list[str]:
    """계좌 잔고를 읽어 기록하고 메일용 요약을 돌려준다(DAY6 D-4). 키가 없거나 실패해도 알림은 보낸다. 주문은 하지 않는다."""
    from src.pipeline import kis_client
    try:
        kis_client.settings()
    except kis_client.KisError:
        return ["키가 없어 읽지 않았다(.env 의 KIS_APP_KEY·KIS_APP_SECRET·KIS_ACCOUNT_NO)"]
    try:
        r = kis_client.record_account()
        return [f"{r['env']} 계좌 · {tools.role_names(config.TRADE_TICKERS)} {r['samsung_qty']}주 · 보유 비중 {r['samsung_weight']:.0%}"]  # 매매 종목이 하나라는 전제(kis_client.record_account 가 그 종목만 읽는다). 열 이름은 옛 그대로
    except Exception as e:  # 계좌 조회 실패가 신호 알림을 막지 않게 한다. 메시지에 계좌번호가 섞일 수 있어 종류만 남긴다
        return [f"계좌 조회 실패: {type(e).__name__}"]


def other_ticker_lines(today: pd.Timestamp) -> list[str]:
    """운영 종목 밖 레지스트리 종목과 채택 분야 동료 종목(multi_ticker.live_tickers) 원본에 어제까지 새 날짜만 덧붙이고,
    레지스트리 종목의 오늘 신호를 만든다(SERVICE.md 5.6, 13절). 실패해도 알림은 보낸다. 주문과 연결하지 않는다."""
    from src.pipeline import multi_ticker
    try:
        multi_ticker.refresh(today)
    except Exception as e:  # 수집이 실패해도 저장된 원본으로 신호는 만든다
        return [f"수집 실패: {type(e).__name__}"] + _safe_other(today)
    return _safe_other(today)


def _safe_other(today: pd.Timestamp) -> list[str]:
    """다른 종목 신호 요약. 실패하면 한 줄로 남긴다."""
    from src.pipeline import multi_ticker
    try:
        return multi_ticker.summary_lines(multi_ticker.live_signals(str(today.date())))
    except Exception as e:
        return [f"신호 실패: {type(e).__name__}: {e}"]


def run(as_of: str | None = None, with_news: bool = True, email: bool = True) -> dict:
    """하루치 자동 실행. 결과(제목, 본문, 성공 여부)를 돌려준다."""
    today = tools.resolve_as_of(as_of)
    log = setup_log(today)
    trading, reason = is_trading_day(today) if as_of is None else (True, "날짜 지정 실행")
    log.info(f"{today.date()} 시작: {reason}")
    if not trading:
        return {"ok": True, "skipped": reason}
    # 다른 종목 정식 신호는 운영 종목 신호·뉴스 채점(수 분)보다 먼저, 따로 만든다: 운영 종목 쪽이 실패해도 빠지지 않고,
    # 장 마감(15:30) 전에 기록돼야 실시간 성적표에 들어가기 때문이다. 날짜를 지정한 재실행에서는 만들지 않는다(그 날짜 뒤 정보가 섞이므로)
    others = other_ticker_lines(today) if as_of is None else None
    try:
        out = live_signal.make_signal(str(today.date()), with_news=with_news)
        subject, body = compose(today, out, board_until=today if as_of else None, others=others)  # 날짜 지정 실행은 그 날짜 전 확정분만
        ok = True
    except Exception as e:
        subject = f"[AutoTrading] {today.date()} 자동 실행 실패: {type(e).__name__}"
        other_part = ("■ 다른 종목 신호(운영 종목 실패와 별개로 기록됨)\n" + "\n".join(f"  {x}" for x in others) + "\n\n") if others else ""
        body = f"{e}\n\n{traceback.format_exc()}\n\n{other_part}{DISCLAIMER}"
        ok = False
    log.info(subject)
    log.info(body)
    if email:
        try:
            notify.send_email(subject, body)
            log.info("Gmail 알림 보냄")
        except Exception as e:
            log.error(f"Gmail 알림 실패: {type(e).__name__}")  # 예외 메시지에는 메일 주소가 들어갈 수 있어 남기지 않는다
    if as_of is None and today.weekday() == config.MONITOR_WEEKDAY:
        for line in other_ticker_monitors():  # 알림을 보낸 뒤에 돈다(종목당 4~5분, 신호·알림을 늦추지 않게)
            log.info(line)
    return {"ok": ok, "subject": subject, "body": body}


def other_ticker_monitors() -> list[str]:
    """운영 종목 밖의 레지스트리 종목마다 조건 모니터링 리포트를 다시 만든다(LLM 없음). 한 종목이 실패해도 나머지는 계속한다."""
    out = []
    for code in [c for c in config.MULTI_TICKERS if c != config.MAIN_TICKER]:
        try:
            summary = experiment.monitor(experiment.Data(code=code))["summary"]
            out.append(f"조건 모니터링 {code}: " + " / ".join(summary))
        except Exception as e:
            out.append(f"조건 모니터링 {code} 실패: {type(e).__name__}: {str(e)[:120]}")
    return out


def main(args: list[str]):
    """명령줄 실행."""
    date = next((a for a in args if not a.startswith("--")), None)
    result = run(date, with_news="--no-news" not in args, email="--no-email" not in args)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
