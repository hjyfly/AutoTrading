"""한국투자증권 주문 (DAY6_PLAN.md, 2026-10-22 사용자 결정: 주문 가능한 API 까지 개발, 신호로 자동 주문은 하지 않음).

사람이 명령줄에서 직접 실행할 때만 주문한다. daily_run·live_signal·agent·ui 는 이 모듈을 부르지 않는다.
안전장치(모두 통과해야 주문이 나간다):
- .env 의 KIS_TRADING_ENABLED=true (기본 꺼짐), 명령줄 --yes
- 실전은 KIS_ENV=real 일 때만(기본 모의투자)
- 삼성전자만(config.KIS_ORDER_CODES), 지정가만, 수량·가격은 양수 정수
- 한 번 금액 ≤ config.KIS_MAX_ORDER_KRW, 하루 주문 ≤ config.KIS_MAX_ORDERS_PER_DAY (같은 환경 기준)
- 모든 시도(거부 포함)를 data/kis/orders.jsonl 에 남긴다(키·계좌번호 제외)

명령줄: python -m src.broker buy 1 70000 --yes        (매수 1주, 지정가 70,000원)
        python -m src.broker sell 1 72000 --dry-run    (보내지 않고 검사만)
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

from src import config
from src.pipeline.kis_client import KisClient, KisError, mask

SIDES = {"buy": "매수", "sell": "매도"}


class OrderRefused(RuntimeError):
    """안전장치에 걸려 주문을 보내지 않았다."""


def order_log_path(env: str) -> Path:
    """주문 기록 파일. 파일은 하나이고 하루 횟수는 orders_today 가 환경별로 걸러 센다."""
    return Path(config.KIS_DIR) / "orders.jsonl"


def _log(entry: dict):
    """주문 시도 한 건을 기록한다."""
    path = order_log_path(entry.get("env", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def orders_today(env: str, today: pd.Timestamp) -> int:
    """오늘(KST) 같은 환경에서 실제로 보낸 주문 수."""
    path = order_log_path(env)
    if not path.exists():
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sum(1 for r in rows if r.get("env") == env and r.get("sent") and str(r.get("at", ""))[:10] == str(today.date()))


def check_order(side: str, code: str, qty: int, price: int, env: str, enabled: bool, confirmed: bool, today: pd.Timestamp) -> None:
    """안전장치 검사. 하나라도 어기면 OrderRefused."""
    if side not in SIDES:
        raise OrderRefused(f"주문 종류는 buy 또는 sell 이다: {side}")
    if code not in config.KIS_ORDER_CODES:
        raise OrderRefused(f"주문할 수 있는 종목은 {config.KIS_ORDER_CODES} 뿐이다: {code}")
    if not (isinstance(qty, int) and qty > 0 and isinstance(price, int) and price > 0):
        raise OrderRefused("수량과 지정가는 양수 정수여야 한다(시장가 주문은 하지 않는다)")
    if qty * price > config.KIS_MAX_ORDER_KRW:
        raise OrderRefused(f"주문 금액 {qty * price:,}원이 한 번 상한 {config.KIS_MAX_ORDER_KRW:,}원을 넘는다")
    if not enabled:
        raise OrderRefused("주문 스위치가 꺼져 있다(.env KIS_TRADING_ENABLED=true 일 때만 주문)")
    if not confirmed:
        raise OrderRefused("명령줄 --yes 로 직접 확인해야 주문한다")
    if orders_today(env, today) >= config.KIS_MAX_ORDERS_PER_DAY:
        raise OrderRefused(f"오늘 {env} 주문이 이미 {config.KIS_MAX_ORDERS_PER_DAY}회라 더 보내지 않는다")


def place_order(side: str, qty: int, price: int, code: str = config.SAMSUNG_CODE, confirmed: bool = False,
                dry_run: bool = False, client: KisClient | None = None) -> dict:
    """지정가 현금 주문 한 건. 안전장치를 모두 통과해야 보낸다. dry_run 이면 검사만 하고 보내지 않는다."""
    client = client or KisClient()
    now = pd.Timestamp.now(tz=config.KST).tz_localize(None)
    enabled = os.getenv("KIS_TRADING_ENABLED", "false").strip().lower() == "true"
    entry = {"at": now, "env": client.env, "side": side, "code": code, "qty": qty, "price": price, "sent": False}
    try:
        check_order(side, code, qty, price, client.env, enabled or dry_run, confirmed or dry_run, now)
    except OrderRefused as e:
        _log({**entry, "refused": str(e)})
        raise
    if dry_run:
        _log({**entry, "dry_run": True})
        return {**entry, "dry_run": True, "note": "검사만 통과했고 보내지 않았다"}
    body = {"CANO": client.cfg["cano"], "ACNT_PRDT_CD": client.cfg["acnt_prdt_cd"], "PDNO": code, "ORD_DVSN": "00",
            "ORD_QTY": str(qty), "ORD_UNPR": str(price)}
    try:
        data = client.request("POST", "order", side, body=body)
    except KisError as e:
        _log({**entry, "error": mask(str(e), client.cfg)})
        raise
    out = data.get("output") or {}
    result = {**entry, "sent": True, "order_no": out.get("ODNO"), "order_time": out.get("ORD_TMD"), "msg": data.get("msg1")}
    _log(result)
    return result


def main(args: list[str]):
    """명령줄: python -m src.broker buy|sell 수량 지정가 [--yes] [--dry-run]"""
    plain = [a for a in args if not a.startswith("--")]
    if len(plain) != 3:
        print(main.__doc__)
        return
    side, qty, price = plain[0], int(plain[1]), int(plain[2])
    try:
        r = place_order(side, qty, price, confirmed="--yes" in args, dry_run="--dry-run" in args)
        print({k: r[k] for k in ("env", "side", "qty", "price", "sent") if k in r}, r.get("order_no") or r.get("note"))
    except (OrderRefused, KisError) as e:
        print(f"주문하지 않음: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
