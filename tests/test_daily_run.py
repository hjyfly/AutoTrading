import pandas as pd
import pytest

from src import daily_run, notify


class FakeResp:
    def __init__(self, traded):
        self._traded = traded

    def json(self):
        return {"localTradedAt": self._traded}


def test_주말은_거래일이_아니다():
    assert daily_run.is_trading_day(pd.Timestamp("2026-10-24"))[0] is False


def test_오늘_체결이_없으면_공휴일로_본다(monkeypatch):
    monkeypatch.setattr(daily_run.requests, "get", lambda *a, **k: FakeResp("2026-10-08T15:30:00+09:00"))
    ok, reason = daily_run.is_trading_day(pd.Timestamp("2026-10-09"))
    assert not ok and "공휴일" in reason


def test_오늘_체결이_있으면_거래일(monkeypatch):
    monkeypatch.setattr(daily_run.requests, "get", lambda *a, **k: FakeResp("2026-10-21T15:20:01+09:00"))
    assert daily_run.is_trading_day(pd.Timestamp("2026-10-21"))[0]


def test_확인에_실패하면_평일은_진행(monkeypatch):
    monkeypatch.setattr(daily_run.requests, "get", lambda *a, **k: (_ for _ in ()).throw(TimeoutError()))
    ok, reason = daily_run.is_trading_day(pd.Timestamp("2026-10-21"))
    assert ok and "실패" in reason


def test_신호_생성이_실패하면_실패_메일을_보낸다(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_run.config, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(daily_run.live_signal, "make_signal", lambda *a, **k: (_ for _ in ()).throw(ValueError("데이터 없음")))
    sent = []
    monkeypatch.setattr(daily_run.notify, "send_email", lambda s, b: sent.append((s, b)))
    result = daily_run.run("2026-10-21")
    assert not result["ok"] and "실패" in sent[0][0] and "데이터 없음" in sent[0][1]


def test_알림_설정이_없으면_보내지_않고_알린다(monkeypatch):
    for k in notify.REQUIRED:
        monkeypatch.delenv(k, raising=False)
    assert not notify.is_configured()
    with pytest.raises(RuntimeError, match="GMAIL_SENDER"):
        notify.send_email("제목", "본문")
