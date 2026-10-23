import json

import numpy as np
import pandas as pd
import pytest

from src import config, tools

DAYS = pd.bdate_range("2025-06-02", periods=200)


@pytest.fixture
def stored(monkeypatch, tmp_path):
    """저장된 원본 대신 가짜 데이터를 돌려주고, 실시간 조회는 막는다."""
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(DAYS)))), index=DAYS)
    data = {
        ("prices", "samsung_ohlcv"): pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                                                   "Volume": rng.integers(1000, 2000, len(DAYS)).astype(float)}, index=DAYS),
        ("prices", "samsung_investor_flows"): pd.DataFrame({"foreign_net": rng.normal(0, 100, len(DAYS)),
                                                            "institution_net": rng.normal(0, 100, len(DAYS)),
                                                            "individual_net": rng.normal(0, 100, len(DAYS)),
                                                            "foreign_hold_ratio": 50.0}, index=DAYS),
    }
    for name in config.YF_TICKERS:
        data[("macro", name)] = pd.DataFrame({"value": 100 + np.cumsum(rng.normal(0, 1, len(DAYS)))}, index=DAYS)
    monkeypatch.setattr(tools, "_load_raw", lambda kind, name: data.get((kind, name), pd.DataFrame(index=pd.DatetimeIndex([]))))

    def no_live(*args):
        raise RuntimeError("실시간 조회 금지(테스트)")
    monkeypatch.setattr(tools, "_live", no_live)
    monkeypatch.setattr(config, "PROCESSED_DIR", str(tmp_path))
    monkeypatch.setattr(config, "TRACES_DIR", str(tmp_path / "traces"))
    return data


def test_as_of_해석():
    assert tools.resolve_as_of("2026-03-11") == pd.Timestamp("2026-03-11")
    assert tools.resolve_as_of(pd.Timestamp("2026-03-10 16:00", tz="UTC")) == pd.Timestamp("2026-03-11")  # = 3/11 01:00 KST


def test_일봉은_as_of_전날까지만(stored):
    t = DAYS[100]
    out = tools.price_history(str(t.date()), days=5)
    assert [r["date"] for r in out["rows"]][-1] == str(DAYS[99].date())
    assert all(r["date"] < str(t.date()) for r in out["rows"])


def test_해외_지표는_같은_날짜를_쓰지_않는다(stored):
    t = DAYS[100]
    out = tools.macro_snapshot(str(t.date()), days=3)
    for s in out["series"].values():
        assert all(r["date"] < str(t.date()) for r in s["rows"])


def test_지표는_학습용_특징과_같은_계산(stored):
    from src.pipeline import features
    t = DAYS[150]
    out = tools.indicators(str(t.date()))
    raw = {"ohlcv": stored[("prices", "samsung_ohlcv")], "flows": stored[("prices", "samsung_investor_flows")],
           **{n: stored[("macro", n)] for n in config.YF_TICKERS}}
    expected = features.build_features(raw=raw).loc[t]
    assert out["signal_row_date"] == str(t.date())
    assert out["chart_and_flow"]["ma20_gap"] == pytest.approx(round(float(expected["ma20_gap"]), 4))


def test_지표는_미래_데이터가_바뀌어도_같다(stored):
    t = DAYS[150]
    before = tools.indicators(str(t.date()))
    for df in stored.values():
        df.loc[df.index >= t] = df.loc[df.index >= t] * 3
    assert tools.indicators(str(t.date())) == before


def test_실시간_조회가_실패하면_저장본으로_답하고_밝힌다(stored):
    out = tools.price_history("2030-01-01", days=3)
    assert "실패" in out["source"]
    assert out["rows"][-1]["date"] == str(DAYS[-1].date())


def test_공시는_접수_당일을_빼고(stored, tmp_path):
    pd.DataFrame({"report_nm": ["분기", "사업"], "rcept_dt": pd.to_datetime(["2025-11-14", "2026-03-10"]),
                  "overview": ["가", "나"]}).to_parquet(tmp_path / "dart_index.parquet")
    assert tools.company_report("2026-03-10")["report"]["name"] == "분기"
    assert tools.company_report("2026-03-11")["report"]["name"] == "사업"


def test_판단_기록이_없으면_없다고_답한다(stored):
    assert "하지 않았다" in tools.news_judgement("2026-03-11")["note"]


def test_판단_기록_요약(stored, tmp_path):
    (tmp_path / "traces").mkdir()
    trace = {"signal_date": "2026-03-11", "as_of": "2026-03-11 15:15 KST", "model": "m", "company": None,
             "articles": [{"no": 1, "title": "삼성전자 신고가"}],
             "final": {"score": 2, "confidence": 0.7, "rationale": "r", "disagreement": "없음", "key_articles": [1]},
             "round2": {"short_trader": {"score": 3, "confidence": 0.7, "rationale": "r", "change_reason": "유지", "key_articles": []}}}
    (tmp_path / "traces" / "2026-03-11.json").write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
    out = tools.news_judgement("2026-03-11")
    assert out["final"]["key_articles"] == ["삼성전자 신고가"]
    assert out["personas"]["short_trader"]["score"] == 3


def test_요청_시점보다_늦은_날짜는_상한으로_낮춘다(stored):
    with tools.request_as_of(str(DAYS[100].date())):
        out = tools.get_price.invoke({"as_of": str(DAYS[150].date()), "days": 3})
    assert out["as_of"] == str(DAYS[100].date())
    assert "이후" in out["as_of_note"]
    assert all(r["date"] < str(DAYS[100].date()) for r in out["rows"])


def test_날짜를_비우면_요청_시점을_쓴다(stored):
    with tools.request_as_of(str(DAYS[100].date())):
        assert tools.get_price.invoke({"days": 1})["as_of"] == str(DAYS[100].date())


def test_요청_시점보다_뒤의_신호_판단기록_성과는_거부한다(stored):
    with tools.request_as_of("2026-03-05"):
        assert "refused" in tools.get_signal.invoke({"date": "2026-03-12"})
        assert "refused" in tools.get_news_judgement.invoke({"date": "2026-03-12"})
        assert "refused" in tools.get_backtest_summary.invoke({})


def test_뉴스_키워드는_정규식이_아니다(stored, tmp_path, monkeypatch):
    days = pd.DatetimeIndex(pd.bdate_range("2026-03-02", "2026-03-13"))
    monkeypatch.setattr(tools.collect_news, "trading_days", lambda: days)
    pd.DataFrame({"signal_date": [pd.Timestamp("2026-03-11")], "title": ["(속보) 삼성"], "source": ["a"], "url": ["u"],
                  "published_utc": [pd.Timestamp("2026-03-11 01:00", tz="UTC")]}).to_parquet(tmp_path / "news.parquet")
    assert len(tools.news_for_signal("2026-03-11", keyword="(")["articles"]) == 1


def test_휴장일과_미수집일은_네트워크_없이_답한다(stored, tmp_path, monkeypatch):
    days = pd.DatetimeIndex(pd.bdate_range("2026-03-02", "2026-03-13"))
    monkeypatch.setattr(tools.collect_news, "trading_days", lambda: days)
    monkeypatch.setattr(tools.collect_news, "collect_rss", lambda *a: (_ for _ in ()).throw(RuntimeError("네트워크 금지")))
    pd.DataFrame({"signal_date": [], "title": [], "published_utc": []}).to_parquet(tmp_path / "news.parquet")
    assert "휴장" in tools.news_for_signal("2026-03-07")["source"]
    assert "수집" in tools.news_for_signal("2026-03-05")["source"]


def test_실시간_조회_끝날은_종가_확정일을_넘지_않는다(stored, monkeypatch):
    seen = []
    monkeypatch.setattr(tools, "_live", lambda kind, name, start, end: seen.append((start, end)) or pd.DataFrame(index=pd.DatetimeIndex([])))
    tools.series_until("prices", "samsung_ohlcv", pd.Timestamp("2099-01-01"))
    assert seen and pd.Timestamp(seen[0][1]) <= tools.last_closed_date("prices", "samsung_ohlcv")
    assert pd.Timestamp(seen[0][0]) == DAYS[-1] + pd.Timedelta(days=1)  # 저장본 다음 날부터만 받는다


def test_실시간_값은_저장본_뒤에만_이어_붙인다(stored, monkeypatch):
    extra_days = pd.bdate_range(DAYS[-1] + pd.Timedelta(days=1), periods=3)
    live = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=extra_days)
    monkeypatch.setattr(tools, "_live", lambda *a: pd.concat([stored[("macro", "nvda")].iloc[-5:] * 0, live]))
    out, note = tools.series_until("macro", "nvda", extra_days[-1] + pd.Timedelta(days=3))
    pd.testing.assert_frame_equal(out.loc[:DAYS[-1]], stored[("macro", "nvda")])  # 저장본은 그대로
    assert out.index[-3:].equals(extra_days) and "3일 추가" in note


@pytest.mark.parametrize("now_utc, kind, name, expected", [
    ("2026-10-21 16:00", "macro", "nvda", "2026-10-20"),     # 뉴욕 10/21 12:00 → 21일 장중이라 20일까지
    ("2026-10-21 21:00", "macro", "nvda", "2026-10-21"),     # 뉴욕 10/21 17:00 → 21일 마감
    ("2026-10-21 05:00", "prices", "samsung_ohlcv", "2026-10-20"),  # 서울 14:00 장중
    ("2026-10-21 07:00", "prices", "samsung_ohlcv", "2026-10-21"),  # 서울 16:00 마감 후
    ("2026-10-21 08:00", "macro", "hstech", "2026-10-21"),   # 홍콩 16:00 → 16:10 전이라... 아래에서 확인
])
def test_시장별_종가_확정일(now_utc, kind, name, expected):
    got = tools.last_closed_date(kind, name, pd.Timestamp(now_utc, tz="UTC"))
    if name == "hstech":
        assert got == pd.Timestamp("2026-10-20")  # 홍콩 16:00 은 16:10 마감 전
    else:
        assert got == pd.Timestamp(expected)


def test_주말_신호는_휴장일이라고_알린다(stored, monkeypatch, tmp_path):
    monkeypatch.setattr(tools.collect_news, "trading_days", lambda: pd.DatetimeIndex(pd.bdate_range("2026-03-02", "2026-03-13")))
    pd.DataFrame({"signal": ["보합"], "confidence": [0.5], "p_상승": [0.2], "p_보합": [0.5], "p_하락": [0.3],
                  "retrained_on": [pd.Timestamp("2026-03-02")], "train_last_future_date": [pd.Timestamp("2026-02-27")]},
                 index=[pd.Timestamp("2026-03-06")]).to_parquet(tmp_path / "signals_numeric.parquet")
    monkeypatch.setattr(config, "SIGNALS_NUMERIC_PATH", str(tmp_path / "signals_numeric.parquet"))
    assert "휴장" in tools.signal_for("2026-03-07")["note"]


def test_에이전트_도구_목록():
    names = {t.name for t in tools.TOOLS}
    assert {"get_price", "get_indicators", "search_news", "get_signal", "get_news_judgement"} <= names
    assert all(t.description for t in tools.TOOLS)


def test_수능일은_16시30분_마감():
    assert tools.last_closed_date("prices", "samsung_ohlcv", pd.Timestamp("2026-11-19 07:00", tz="UTC")) == pd.Timestamp("2026-11-18")  # 16:00 KST
    assert tools.last_closed_date("prices", "samsung_ohlcv", pd.Timestamp("2026-11-19 07:31", tz="UTC")) == pd.Timestamp("2026-11-19")


def test_늘어난_신호가_있어도_공식_판정_구간은_고정되고_요청_시점_뒤는_계산하지_않는다(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "signals_last_date", lambda: pd.Timestamp("2026-10-29"))
    monkeypatch.setattr(tools, "_backtest_summary", lambda until, last, *_: calls.append((until, last)) or {})
    with tools.request_as_of("2026-10-22"):
        tools.get_backtest_summary.invoke({})
    assert calls == [("2026-10-21", "2026-10-29")]  # 요청 시점 전날까지만
    with tools.request_as_of("2026-10-20"):
        assert "refused" in tools.get_backtest_summary.invoke({})  # 공식 구간 끝(10-20) 종가 확정 전


def test_뉴스가_참고용이면_최종_신호는_숫자_모델_신호다(monkeypatch):
    from pathlib import Path
    if not (Path(config.PROCESSED_DIR) / "signals_news.parquet").exists():
        pytest.skip("뉴스 보정 신호가 없다")
    monkeypatch.setattr(config, "NEWS_ADJUST_IN_SIGNAL", False)
    sig = tools.signal_for("2026-03-05")
    assert sig["with_news"]["signal"] == sig["numeric"]["signal"] and sig["with_news"]["adjusted"] is False


def test_데이터가_없는_날짜의_도구_호출은_에이전트를_멈추지_않고_이유를_돌려준다():
    result = tools.get_indicators.invoke({"as_of": "2019-05-03"})
    assert "error" in result and "2019-05-03" in result["error"]


@pytest.fixture
def ticker_stored(stored):
    """다른 종목(SK하이닉스) 원본을 가짜 데이터로 더한다."""
    base = stored[("prices", "samsung_ohlcv")]
    stored[("tickers", "000660_ohlcv")] = base * 2
    stored[("tickers", "000660_flows")] = stored[("prices", "samsung_investor_flows")] * 3
    return stored


def test_종목은_이름과_코드로_고르고_없는_종목은_거부한다():
    assert tools.ticker_code("") == config.SAMSUNG_CODE and tools.ticker_code("SK하이닉스") == "000660"
    assert tools.ticker_code("277810") == "277810"
    with pytest.raises(ValueError):
        tools.ticker_code("카카오")


def test_다른_종목_일봉_수급은_그_종목_원본을_as_of_전날까지(ticker_stored):
    t = DAYS[150]
    p = tools.price_history(str(t.date()), days=5, code="SK하이닉스")
    assert p["name"] == "SK하이닉스" and p["rows"][-1]["date"] == str(DAYS[149].date())
    assert p["rows"][-1]["Close"] == pytest.approx(round(float(ticker_stored[("prices", "samsung_ohlcv")]["Close"].iloc[149] * 2), 4))
    f = tools.investor_flows(str(t.date()), days=3, code="000660")
    assert f["rows"][-1]["date"] == str(DAYS[149].date())


def test_다른_종목_지표는_미래_데이터가_바뀌어도_같다(ticker_stored):
    t = DAYS[150]
    before = tools.indicators(str(t.date()), code="000660")
    ticker_stored[("tickers", "000660_ohlcv")].loc[DAYS[150]:, "Close"] *= 5  # t 이후 값을 바꾼다
    after = tools.indicators(str(t.date()), code="000660")
    assert before["chart_and_flow"] == after["chart_and_flow"] and before["name"] == "SK하이닉스"
    assert before["chart_and_flow"] != tools.indicators(str(t.date()))["chart_and_flow"]  # 삼성전자와는 다르다


def test_다른_종목_신호는_실시간_기록을_먼저_쓰고_없으면_워크포워드_신호(tmp_path, monkeypatch):
    t = pd.Timestamp("2026-10-23")
    cols = {"signal": "하락", "confidence": 0.5, "p_상승": 0.2, "p_보합": 0.3, "p_하락": 0.5,
            "retrained_on": pd.Timestamp("2026-10-19"), "train_last_future_date": pd.Timestamp("2026-10-16")}
    stored = pd.DataFrame([{**cols, "signal": "상승"}] * 2, index=[t - pd.Timedelta(days=1), t])
    monkeypatch.setattr(config, "SECTOR_DIR", str(tmp_path))
    stored.to_parquet(tmp_path / "signals_S0_000660.parquet")
    live = pd.DataFrame([{**cols, "code": "000660", "model": "S0 단독", "created_at": t + pd.Timedelta(hours=15, minutes=20)}], index=[t])
    monkeypatch.setattr(config, "MULTI_LIVE_PATH", str(tmp_path / "live.parquet"))
    live.to_parquet(tmp_path / "live.parquet")
    now = tools.signal_for(str(t.date()), "SK하이닉스")["numeric"]
    assert now["signal"] == "하락" and now["source"].startswith("실시간 기록(장 마감 전")
    before = tools.signal_for(str((t - pd.Timedelta(days=1)).date()), "000660")["numeric"]
    assert before["signal"] == "상승" and before["source"].startswith("워크포워드")


def test_다른_종목_신호도_요청_시점보다_뒤는_거부하고_없는_종목은_이유를_돌려준다():
    with tools.request_as_of("2026-10-01"):
        assert "refused" in tools.get_signal.invoke({"date": "2026-10-15", "code": "SK하이닉스"})
        assert "error" in tools.get_signal.invoke({"date": "2026-09-30", "code": "카카오"})


def test_기간별_신호는_실시간_기록을_먼저_쓰고_없으면_기간별_검증_신호(tmp_path, monkeypatch):
    t = pd.Timestamp("2026-10-23")
    cols = {"confidence": 0.5, "p_상승": 0.2, "p_보합": 0.3, "p_하락": 0.5,
            "retrained_on": pd.Timestamp("2026-10-19"), "train_last_future_date": pd.Timestamp("2026-09-18")}
    monkeypatch.setattr(config, "HORIZON_DIR", str(tmp_path / "h"))
    monkeypatch.setattr(config, "PROCESSED_DIR", str(tmp_path))
    (tmp_path / "h").mkdir()
    pd.DataFrame([{**cols, "signal": "상승"}] * 2, index=[t - pd.Timedelta(days=1), t]).to_parquet(tmp_path / "h" / "signals_numeric_h20.parquet")
    pd.DataFrame([{**cols, "signal": "하락", "model": "h20"}, {**cols, "signal": "보합", "model": "base"}], index=[t, t]).to_parquet(tmp_path / "signals_live.parquet")
    now = tools.signal_for(str(t.date()), None, 20)
    assert now["numeric"]["signal"] == "하락" and now["numeric"]["source"].startswith("실시간") and now["horizon_days"] == 20
    before = tools.signal_for(str((t - pd.Timedelta(days=1)).date()), None, 20)
    assert before["numeric"]["signal"] == "상승" and before["numeric"]["source"].startswith("기간별 과거 검증")
    assert "매매 기준은 5거래일" in before["note"]
    with tools.request_as_of(str(t.date())):
        assert "error" in tools.get_signal.invoke({"date": str(t.date()), "code": "SK하이닉스", "horizon": 20})
        assert "error" in tools.get_signal.invoke({"date": str(t.date()), "horizon": 7})


def test_기간_모델_이름():
    assert tools.horizon_model(None) == "base" and tools.horizon_model(5) == "base" and tools.horizon_model(20) == "h20"
    assert tools.model_horizon("h20") == 20 and tools.model_horizon("base") == 5 and tools.model_horizon("candidate") == 5
    assert tools.model_horizon("h7") == 5  # 등록되지 않은 기간은 운영 기간으로 본다
