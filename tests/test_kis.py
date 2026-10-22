import re
from pathlib import Path

import pandas as pd
import pytest

from src import broker, config
from src.pipeline import kis_client as kc

CFG = {"env": "mock", "app_key": "APPKEY123", "app_secret": "SECRET456", "cano": "12345678", "acnt_prdt_cd": "01"}


class FakeResp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data


class FakeSession:
    """가짜 API: 토큰 발급 횟수와 마지막 요청을 기록한다."""

    def __init__(self, responses=None):
        self.token_calls, self.calls, self.responses = 0, [], responses or {}

    def post(self, url, json=None, timeout=None):
        self.token_calls += 1
        return FakeResp({"access_token": "TOKEN", "expires_in": 86400})

    def request(self, method, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "params": params, "json": json})
        for key, data in self.responses.items():
            if url.endswith(config.KIS_PATHS[key]):
                return FakeResp(data)
        return FakeResp({"rt_cd": "1", "msg1": "없음"})


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "KIS_DIR", str(tmp_path / "kis"))
    monkeypatch.setattr(config, "ACCOUNT_LIVE_PATH", str(tmp_path / "kis" / "account_live.parquet"))
    monkeypatch.setattr(config, "KIS_MIN_INTERVAL_SEC", {"real": 0, "mock": 0})


def test_토큰은_저장해_두고_만료_전까지_다시_쓴다():
    session = FakeSession({"daily_price": {"rt_cd": "0", "output": []}})
    client = kc.KisClient(CFG, session=session)
    client.daily_prices(); client.daily_prices()
    assert session.token_calls == 1
    kc.KisClient(CFG, session=session).daily_prices()  # 새 클라이언트도 저장된 토큰을 쓴다
    assert session.token_calls == 1


def test_모의_환경은_모의_거래_ID_와_모의_서버를_쓴다():
    session = FakeSession({"balance": {"rt_cd": "0", "output1": [], "output2": [{"dnca_tot_amt": "1000", "tot_evlu_amt": "1000"}]}})
    kc.KisClient(CFG, session=session).balance()
    call = session.calls[-1]
    assert call["headers"]["tr_id"] == config.KIS_TR_IDS["balance"][1] and call["url"].startswith(config.KIS_BASE_URLS["mock"])


def test_잔고를_읽어_삼성전자_보유_비중을_기록한다():
    session = FakeSession({"balance": {"rt_cd": "0",
                                       "output1": [{"pdno": "005930", "prdt_name": "삼성전자", "hldg_qty": "10", "pchs_avg_pric": "70000",
                                                    "evlu_amt": "750000"}],
                                       "output2": [{"dnca_tot_amt": "250000", "tot_evlu_amt": "1000000"}]}})
    row = kc.record_account(kc.KisClient(CFG, session=session), today=pd.Timestamp("2026-10-23"))
    assert row["samsung_qty"] == 10 and row["samsung_weight"] == pytest.approx(0.75)
    saved = pd.read_parquet(config.ACCOUNT_LIVE_PATH)
    assert list(saved.index) == [pd.Timestamp("2026-10-23")] and "12345678" not in saved.to_csv()


def test_오류_메시지에서_키와_계좌번호를_가린다():
    session = FakeSession({"daily_price": {"rt_cd": "1", "msg_cd": "E1", "msg1": "APPKEY123 12345678 오류"}})
    with pytest.raises(kc.KisError) as e:
        kc.KisClient(CFG, session=session).daily_prices()
    assert "APPKEY123" not in str(e.value) and "12345678" not in str(e.value)


def test_계좌번호_형식이_틀리면_멈춘다(monkeypatch):
    for k, v in {"KIS_APP_KEY": "a", "KIS_APP_SECRET": "b", "KIS_ACCOUNT_NO": "123"}.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(kc.KisError):
        kc.settings()


ORDER_OK = {"order": {"rt_cd": "0", "msg1": "주문 전송 완료", "output": {"ODNO": "0001", "ORD_TMD": "152000"}}}


def test_주문은_스위치가_꺼져_있으면_보내지_않는다(monkeypatch):
    monkeypatch.delenv("KIS_TRADING_ENABLED", raising=False)
    session = FakeSession(ORDER_OK)
    with pytest.raises(broker.OrderRefused):
        broker.place_order("buy", 1, 70000, confirmed=True, client=kc.KisClient(CFG, session=session))
    assert session.calls == []


@pytest.mark.parametrize("kwargs, reason", [
    ({"side": "buy", "qty": 1, "price": 70000, "code": "000660"}, "종목"),
    ({"side": "buy", "qty": 0, "price": 70000}, "양수"),
    ({"side": "buy", "qty": 100, "price": 70000}, "상한"),
    ({"side": "short", "qty": 1, "price": 70000}, "종류"),
])
def test_주문_안전장치(monkeypatch, kwargs, reason):
    monkeypatch.setenv("KIS_TRADING_ENABLED", "true")
    session = FakeSession(ORDER_OK)
    with pytest.raises(broker.OrderRefused, match=reason):
        broker.place_order(confirmed=True, client=kc.KisClient(CFG, session=session), **kwargs)
    assert session.calls == []


def test_확인한_주문은_모의_서버로_한_번만_나가고_기록된다(monkeypatch):
    monkeypatch.setenv("KIS_TRADING_ENABLED", "true")
    session = FakeSession(ORDER_OK)
    client = kc.KisClient(CFG, session=session)
    with pytest.raises(broker.OrderRefused):  # --yes 없이는 보내지 않는다
        broker.place_order("buy", 1, 70000, client=client)
    r = broker.place_order("buy", 1, 70000, confirmed=True, client=client)
    assert r["sent"] and r["order_no"] == "0001"
    call = session.calls[-1]
    assert call["headers"]["tr_id"] == config.KIS_TR_IDS["buy"][1] and call["json"]["ORD_DVSN"] == "00"
    with pytest.raises(broker.OrderRefused, match="오늘"):  # 하루 1회
        broker.place_order("sell", 1, 71000, confirmed=True, client=client)
    log = (Path(config.KIS_DIR) / "orders.jsonl").read_text(encoding="utf-8")
    assert "12345678" not in log and "APPKEY123" not in log and log.count('"sent": true') == 1


def test_검사만_하면_보내지_않는다(monkeypatch):
    monkeypatch.delenv("KIS_TRADING_ENABLED", raising=False)
    session = FakeSession(ORDER_OK)
    r = broker.place_order("buy", 1, 70000, dry_run=True, client=kc.KisClient(CFG, session=session))
    assert r["dry_run"] and session.calls == []


def test_운영_경로는_주문_모듈을_부르지_않는다():
    for path in ["src/daily_run.py", "src/pipeline/live_signal.py", "src/agent.py", "src/ui.py", "src/tools.py", "src/server.py"]:
        text = Path(path).read_text(encoding="utf-8")
        assert not re.search(r"\bbroker\b", text), path


def test_키가_바뀌면_저장된_토큰을_쓰지_않고_새로_받는다():
    session = FakeSession({"daily_price": {"rt_cd": "0", "output": []}})
    kc.KisClient(CFG, session=session).daily_prices()
    kc.KisClient({**CFG, "app_key": "OTHERKEY"}, session=session).daily_prices()
    assert session.token_calls == 2
