"""한국투자증권 Open API 조회 클라이언트 (DAY6_PLAN.md, 2026-10-22 사용자 결정).

- 토큰을 받아 data/kis/token_<env>.json 에 저장하고 만료 전까지 다시 쓴다(발급 횟수 제한 때문).
- 호출 사이 최소 간격(config.KIS_MIN_INTERVAL_SEC)을 지킨다.
- 실전/모의는 .env 의 KIS_ENV(mock | real, 기본 mock)로 고른다. 키·계좌번호는 .env 에서만 읽고 로그에 남기지 않는다.
- 시세·수급 조회는 연결 확인과 예비 출처용이다. 운영 수집은 지금 출처(FinanceDataReader·네이버) 그대로다.
- 주문은 이 파일에 없다(src/broker.py, 사람이 직접 실행).

명령줄: python -m src.pipeline.kis_client --probe   (모든 조회가 되는지 확인. 결과에 비밀 값은 나오지 않는다)
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src import config

load_dotenv()
SECRET_KEYS = ("KIS_APP_KEY", "KIS_APP_SECRET", "KIS_ACCOUNT_NO")


class KisError(RuntimeError):
    """한국투자증권 API 호출 실패(응답 코드가 성공이 아님)."""


def settings() -> dict:
    """환경 설정을 읽는다. 키가 없으면 KisError."""
    env = os.getenv("KIS_ENV", config.KIS_DEFAULT_ENV).strip().lower()
    if env not in config.KIS_BASE_URLS:
        raise KisError(f"KIS_ENV 는 mock 또는 real 이어야 한다: {env}")
    missing = [k for k in SECRET_KEYS if not os.getenv(k)]
    if missing:
        raise KisError(f".env 에 {', '.join(missing)} 가 없다")
    account = os.environ["KIS_ACCOUNT_NO"].replace("-", "").strip()
    if len(account) != 10 or not account.isdigit():
        raise KisError("KIS_ACCOUNT_NO 는 계좌 8자리 + 상품 2자리(예: 12345678-01)여야 한다")
    return {"env": env, "app_key": os.environ["KIS_APP_KEY"], "app_secret": os.environ["KIS_APP_SECRET"],
            "cano": account[:8], "acnt_prdt_cd": account[8:]}


def mask(text: str, cfg: dict) -> str:
    """문자열 안의 키·시크릿·계좌번호를 가린다(오류 메시지·기록에 남기기 전에)."""
    for value in (cfg.get("app_key"), cfg.get("app_secret"), cfg.get("cano")):
        if value:
            text = text.replace(value, "***")
    return text


class KisClient:
    """조회용 클라이언트. session 을 바꿔 끼우면 테스트에서 가짜 응답을 쓸 수 있다."""

    def __init__(self, cfg: dict | None = None, session=None, now=time.time):
        self.cfg = cfg or settings()
        self.env = self.cfg["env"]
        self.base = config.KIS_BASE_URLS[self.env]
        self.session = session or requests.Session()
        self.now = now
        self._last_call = 0.0
        self.token_path = Path(config.KIS_DIR) / f"token_{self.env}.json"

    def key_id(self) -> str:
        """저장 토큰이 어떤 키로 받은 것인지 가리는 값(키 자체가 아니라 해시 앞부분)."""
        import hashlib
        return hashlib.sha256(self.cfg["app_key"].encode()).hexdigest()[:12]

    def _wait(self):
        """호출 사이 최소 간격을 지킨다."""
        gap = config.KIS_MIN_INTERVAL_SEC[self.env] - (self.now() - self._last_call)
        if gap > 0:
            time.sleep(gap)
        self._last_call = self.now()

    def token(self) -> str:
        """저장된 토큰이 만료 KIS_TOKEN_MARGIN_MIN 분 전까지 남아 있으면 다시 쓰고, 아니면 새로 받아 저장한다."""
        if self.token_path.exists():
            saved = json.loads(self.token_path.read_text(encoding="utf-8"))
            fresh = saved.get("expires_at", 0) - config.KIS_TOKEN_MARGIN_MIN * 60 > self.now()
            if fresh and saved.get("key_id") == self.key_id():  # 키를 바꾸면(실전 ↔ 모의 키) 새로 받는다
                return saved["access_token"]
        self._wait()
        resp = self.session.post(self.base + config.KIS_PATHS["token"], timeout=config.KIS_TIMEOUT_SEC,
                                 json={"grant_type": "client_credentials", "appkey": self.cfg["app_key"],
                                       "appsecret": self.cfg["app_secret"]})
        body = resp.json()
        if resp.status_code != 200 or "access_token" not in body:
            raise KisError(mask(f"토큰 발급 실패({resp.status_code}): {body}", self.cfg))
        expires_at = self.now() + int(body.get("expires_in", 86400))
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps({"access_token": body["access_token"], "expires_at": expires_at, "key_id": self.key_id()}),
                                   encoding="utf-8")
        return body["access_token"]

    def headers(self, tr_name: str) -> dict:
        """호출 머리글. tr_name 은 config.KIS_TR_IDS 의 이름(실전·모의에 맞는 거래 ID 를 고른다)."""
        real, mock = config.KIS_TR_IDS[tr_name]
        return {"content-type": "application/json; charset=utf-8", "authorization": f"Bearer {self.token()}",
                "appkey": self.cfg["app_key"], "appsecret": self.cfg["app_secret"],
                "tr_id": real if self.env == "real" else mock, "custtype": "P"}

    def request(self, method: str, path_name: str, tr_name: str, params: dict | None = None, body: dict | None = None) -> dict:
        """API 를 한 번 부르고 응답을 돌려준다. 응답 코드(rt_cd)가 성공('0')이 아니면 KisError."""
        headers = self.headers(tr_name)
        self._wait()
        resp = self.session.request(method, self.base + config.KIS_PATHS[path_name], headers=headers,
                                    params=params, json=body, timeout=config.KIS_TIMEOUT_SEC)
        data = resp.json()
        if resp.status_code != 200 or str(data.get("rt_cd")) != "0":
            raise KisError(mask(f"{path_name} 실패({resp.status_code}): {data.get('msg_cd')} {data.get('msg1')}", self.cfg))
        return data

    def daily_prices(self, code: str = config.SAMSUNG_CODE) -> pd.DataFrame:
        """최근 일봉(수정주가). 연결 확인·예비 출처용."""
        data = self.request("GET", "daily_price", "daily_price", params={
            "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code, "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "1"})
        rows = [{"date": pd.Timestamp(r["stck_bsop_date"]), "Open": float(r["stck_oprc"]), "High": float(r["stck_hgpr"]),
                 "Low": float(r["stck_lwpr"]), "Close": float(r["stck_clpr"]), "Volume": float(r["acml_vol"])}
                for r in data.get("output", []) if r.get("stck_bsop_date")]
        return pd.DataFrame(rows).set_index("date").sort_index() if rows else pd.DataFrame()

    def investor_flows(self, code: str = config.SAMSUNG_CODE) -> pd.DataFrame:
        """최근 투자자별 순매수 수량(외국인·기관·개인). 네이버 수급의 예비 출처용."""
        data = self.request("GET", "investor", "investor", params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code})
        rows = [{"date": pd.Timestamp(r["stck_bsop_date"]), "foreign_net": float(r.get("frgn_ntby_qty") or 0),
                 "institution_net": float(r.get("orgn_ntby_qty") or 0), "individual_net": float(r.get("prsn_ntby_qty") or 0)}
                for r in data.get("output", []) if r.get("stck_bsop_date")]
        return pd.DataFrame(rows).set_index("date").sort_index() if rows else pd.DataFrame()

    def balance(self) -> dict:
        """계좌 잔고(읽기 전용): 종목별 보유 수량·평가 금액과 예수금·총평가."""
        data = self.request("GET", "balance", "balance", params={
            "CANO": self.cfg["cano"], "ACNT_PRDT_CD": self.cfg["acnt_prdt_cd"], "AFHR_FLPR_YN": "N", "OFL_YN": "",
            "INQR_DVSN": "02", "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00", "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""})
        holdings = [{"code": r["pdno"], "name": r.get("prdt_name"), "qty": int(float(r.get("hldg_qty") or 0)),
                     "avg_price": float(r.get("pchs_avg_pric") or 0), "eval_amount": float(r.get("evlu_amt") or 0)}
                    for r in data.get("output1", []) if int(float(r.get("hldg_qty") or 0)) > 0]
        summary = (data.get("output2") or [{}])[0]
        return {"holdings": holdings, "cash": float(summary.get("dnca_tot_amt") or 0),
                "total_eval": float(summary.get("tot_evlu_amt") or 0)}


def record_account(client: KisClient | None = None, today: pd.Timestamp | None = None) -> dict:
    """계좌 잔고를 읽어 그날 삼성전자 보유 비중과 함께 config.ACCOUNT_LIVE_PATH 에 한 줄 남긴다(같은 날은 덮어씀)."""
    client = client or KisClient()
    today = (pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz=config.KST).tz_localize(None)).normalize()
    bal = client.balance()
    samsung = next((h for h in bal["holdings"] if h["code"] == config.SAMSUNG_CODE), None)
    total = bal["total_eval"] or (bal["cash"] + sum(h["eval_amount"] for h in bal["holdings"]))
    row = {"date": today, "env": client.env, "samsung_qty": samsung["qty"] if samsung else 0,
           "samsung_eval": samsung["eval_amount"] if samsung else 0.0, "cash": bal["cash"], "total_eval": total,
           "samsung_weight": (samsung["eval_amount"] / total) if samsung and total else 0.0,
           "recorded_at": pd.Timestamp.now(tz=config.KST).tz_localize(None)}
    path = Path(config.ACCOUNT_LIVE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([row]).set_index("date")
    if path.exists():
        old = pd.read_parquet(path)
        table = pd.concat([old[old.index != today], table]).sort_index()
    table.to_parquet(path)
    return row


def probe() -> dict:
    """연결 확인(시험): 토큰·일봉·수급·잔고를 한 번씩 불러 건수만 돌려준다. 저장하지 않고 비밀 값은 내보내지 않는다."""
    client = KisClient()
    out = {"env": client.env}
    for name, fn in [("daily_prices", client.daily_prices), ("investor_flows", client.investor_flows), ("balance", client.balance)]:
        try:
            result = fn()
            out[name] = {"ok": True, "rows": len(result) if isinstance(result, pd.DataFrame) else len(result["holdings"]),
                         **({"last_date": str(result.index.max().date())} if isinstance(result, pd.DataFrame) and len(result) else {})}
        except Exception as e:
            out[name] = {"ok": False, "error": mask(f"{type(e).__name__}: {e}", client.cfg)}
    return out


if __name__ == "__main__":
    try:
        settings()
    except KisError as e:
        sys.exit(f"한국투자증권 설정이 없다: {e}")
    if "--probe" in sys.argv[1:]:
        print(json.dumps(probe(), ensure_ascii=False, indent=1))
    elif "--account" in sys.argv[1:]:
        try:
            r = record_account()
        except Exception as e:  # 연결 오류 메시지의 URL 에 계좌번호가 들어갈 수 있어 종류만 보인다
            sys.exit(f"계좌 조회 실패: {type(e).__name__}")
        print({k: v for k, v in r.items() if k not in ("samsung_eval", "cash", "total_eval")}, "저장:", config.ACCOUNT_LIVE_PATH)
