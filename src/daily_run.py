"""매일 자동 실행. 작업 스케줄러가 평일 15:20 에 부른다.

1. 오늘이 거래일인지 확인한다(주말·공휴일이면 아무것도 하지 않는다).
2. 오늘 신호를 만든다: 운영 모델(최종, 뉴스 채점은 참고용), 후보 모델·비교 묶음·하락 위험 비교 조건(비교용).
3. 실시간 성적표를 다시 계산한다. 금요일에는 저장 데이터를 갱신(config.REFRESH_WEEKDAY)한 뒤 조건 모니터링 리포트(config.MONITOR_WEEKDAY)를 다시 만든다.
4. Gmail 로 결과를 보낸다. 실패하면 원인을 담은 메일을 보낸다.
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
        resp = requests.get(config.NAVER_BASIC_URL.format(code=config.SAMSUNG_CODE), headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        traded = pd.Timestamp(resp.json()["localTradedAt"]).tz_convert(config.KST).tz_localize(None).normalize()
        if traded < today:
            return False, f"오늘 체결이 없다(마지막 체결 {traded.date()}). 공휴일로 본다"
        return True, "거래일"
    except Exception as e:
        return True, f"거래일 확인 실패({type(e).__name__}), 평일이라 진행"


def compose(today: pd.Timestamp, out: dict, board_until: pd.Timestamp | None = None) -> tuple[str, str]:
    """알림 제목과 본문을 만든다."""
    sig = tools.signal_for(str(today.date()))
    judgement = tools.news_judgement(str(today.date()))
    final = sig.get("with_news", {}).get("signal") or sig.get("numeric", {}).get("signal", "없음")
    numeric = sig.get("numeric", {})
    subject = f"[AutoTrading] {today.date()} {final} (확신도 {numeric.get('confidence', 0):.0%})"
    action = config.DOWN_ACTION_LABELS[config.DOWN_ACTION]
    lines = [f"{today.date()} {config.SIGNAL_CUTOFF_KST} 기준 삼성전자 {config.LABEL_HORIZON_DAYS}거래일 방향 신호",
             f"(모의 매매 규칙: 상승 = 삼성전자, 보합 = 유지, 하락 = {action})", "",
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
    board = scoreboard.build(board_until)
    lines += ["", "■ 실시간 성적표"] + [f"  {x}" for x in scoreboard.summary_lines(board)]
    scheduled = board_until is None  # 날짜를 지정한 재실행에서는 갱신·모니터링을 하지 않는다(그 날짜 뒤 정보가 섞이므로)
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
    warnings = [f"{k}: {v['note']}" for k, v in out.get("data_sources", {}).items() if "실패" in v["note"]]
    if warnings:
        lines += ["", "■ 데이터 경고"] + [f"  {w}" for w in warnings]
    lines += ["", f"LLM 누적 비용 ${llm.total_cost():.2f} / 한도 ${config.COST_LIMIT_POC_USD:.0f}", "", DISCLAIMER]
    return subject, "\n".join(lines)


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


def run(as_of: str | None = None, with_news: bool = True, email: bool = True) -> dict:
    """하루치 자동 실행. 결과(제목, 본문, 성공 여부)를 돌려준다."""
    today = tools.resolve_as_of(as_of)
    log = setup_log(today)
    trading, reason = is_trading_day(today) if as_of is None else (True, "날짜 지정 실행")
    log.info(f"{today.date()} 시작: {reason}")
    if not trading:
        return {"ok": True, "skipped": reason}
    try:
        out = live_signal.make_signal(str(today.date()), with_news=with_news)
        subject, body = compose(today, out, board_until=today if as_of else None)  # 날짜 지정 실행은 그 날짜 전 확정분만
        ok = True
    except Exception as e:
        subject = f"[AutoTrading] {today.date()} 자동 실행 실패: {type(e).__name__}"
        body = f"{e}\n\n{traceback.format_exc()}\n\n{DISCLAIMER}"
        ok = False
    log.info(subject)
    log.info(body)
    if email:
        try:
            notify.send_email(subject, body)
            log.info("Gmail 알림 보냄")
        except Exception as e:
            log.error(f"Gmail 알림 실패: {type(e).__name__}")  # 예외 메시지에는 메일 주소가 들어갈 수 있어 남기지 않는다
    return {"ok": ok, "subject": subject, "body": body}


def main(args: list[str]):
    """명령줄 실행."""
    date = next((a for a in args if not a.startswith("--")), None)
    result = run(date, with_news="--no-news" not in args, email="--no-email" not in args)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
