import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, tools
from src.pipeline import features
from src.pipeline import multi_ticker as mt
from src.pipeline import train as tr
from src.pipeline.labels import make_labels

DAYS = pd.bdate_range("2024-01-01", periods=420)
CODE = "999990"


def _ohlcv(rng, n=len(DAYS), index=DAYS):
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=index)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                         "Volume": rng.integers(1000, 2000, n).astype(float)}, index=index)


def _flows(rng):
    return pd.DataFrame({"foreign_net": rng.normal(0, 100, len(DAYS)), "institution_net": rng.normal(0, 100, len(DAYS)),
                         "individual_net": rng.normal(0, 100, len(DAYS)), "foreign_hold_ratio": 50 + rng.normal(0, 0.1, len(DAYS))},
                        index=DAYS)


@pytest.fixture
def stored(monkeypatch, tmp_path):
    """가짜 공통 원본(tools)과 가짜 종목 원본(tmp). 실시간 조회는 금지, 작은 LightGBM."""
    rng = np.random.default_rng(7)
    data = {("prices", "samsung_ohlcv"): _ohlcv(rng), ("prices", "samsung_investor_flows"): _flows(rng)}
    for name in {**config.YF_TICKERS, **config.YF_TICKERS_EXT}:
        data[("macro", name)] = pd.DataFrame({"value": 100 + np.cumsum(rng.normal(0, 1, len(DAYS)))}, index=DAYS)
    for name in config.KR_EXTRA:
        data[("prices", name)] = pd.DataFrame({"Close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(DAYS))))}, index=DAYS)
    monkeypatch.setattr(tools, "_load_raw", lambda kind, name: data.get((kind, name), pd.DataFrame(index=pd.DatetimeIndex([]))))
    monkeypatch.setattr(tools, "_live", lambda *a: (_ for _ in ()).throw(RuntimeError("실시간 금지")))
    monkeypatch.setattr(config, "LGBM_PARAMS", {**config.LGBM_PARAMS, "n_estimators": 20, "min_child_samples": 5})
    monkeypatch.setattr(config, "MIN_TRAIN_ROWS", 30)
    monkeypatch.setattr(config, "TICKER_RAW_DIR", str(tmp_path))
    monkeypatch.setattr(config, "MULTI_TICKERS", {config.SAMSUNG_CODE: ("삼성전자", "반도체", None), CODE: ("가짜", "로봇", "etf")})
    ticker = {"ohlcv": _ohlcv(rng), "flows": _flows(rng),
              "sector": pd.DataFrame({"value": 100 + np.cumsum(rng.normal(0, 1, len(DAYS)))}, index=DAYS)}
    ticker["ohlcv"].to_parquet(tmp_path / f"{CODE}_ohlcv.parquet")
    ticker["flows"].to_parquet(tmp_path / f"{CODE}_flows.parquet")
    ticker["sector"].to_parquet(tmp_path / "sector_etf.parquet")
    shared = {"ohlcv": data[("prices", "samsung_ohlcv")], "flows": data[("prices", "samsung_investor_flows")],
              **{n: data[("macro", n)] for n in {**config.YF_TICKERS, **config.YF_TICKERS_EXT}},
              **{n: data[("prices", n)] for n in config.KR_EXTRA}}
    return {"data": data, "ticker": ticker, "shared": shared}


def test_덧붙이기는_저장된_행을_바꾸지_않는다(tmp_path):
    path = tmp_path / "x.parquet"
    old = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    assert mt.append_only(path, old) == 2
    fresh = pd.DataFrame({"Close": [9.0, 9.0, 3.0]}, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    assert mt.append_only(path, fresh) == 1
    saved = pd.read_parquet(path)
    assert saved["Close"].tolist() == [1.0, 2.0, 3.0]
    assert mt.append_only(path, fresh) == 0


def test_종목_원본은_그_종목_일봉으로_바꿔_끼운다(stored):
    raw = mt.ticker_raw(CODE, stored["shared"])
    assert raw["ohlcv"]["Close"].equals(stored["ticker"]["ohlcv"]["Close"])
    assert raw["sox"] is stored["shared"]["sox"]  # 거시·해외 지표는 공통 원본
    assert mt.ticker_raw(config.SAMSUNG_CODE, stored["shared"])["ohlcv"] is stored["shared"]["ohlcv"]


def test_종목_특징은_미래_값이_바뀌어도_같다(stored):
    """t 이후 종목·분야 ETF 값을 바꿔도 t 행 특징(국내는 t-1 종가까지, 해외는 현지 날짜 < t)은 같아야 한다."""
    t = DAYS[333]
    def row():
        raw = mt.ticker_raw(CODE, stored["shared"])
        table = features.build_features(raw=raw)
        return table.join(mt.sector_features(CODE, table.index)).loc[t]
    before = row()
    for kind in ("ohlcv", "flows", "sector"):
        df = stored["ticker"][kind]
        df.loc[df.index >= t] = df.loc[df.index >= t] * 3
        df.to_parquet(Path(config.TICKER_RAW_DIR) / (f"{CODE}_{kind}.parquet" if kind != "sector" else "sector_etf.parquet"))
    pd.testing.assert_series_equal(before, row())


def test_공동_학습은_정답_기간이_재학습일_전에_끝난_행만_쓴다(stored):
    tables = {}
    for code in config.MULTI_TICKERS:
        raw = mt.ticker_raw(code, stored["shared"])
        tables[code] = {"features": features.build_features(raw=raw), "labels": make_labels(raw["ohlcv"]["Close"])}
    cols = list(config.FEATURE_SET_BASE)
    out = mt.pooled_walk_forward(tables, cols, DAYS[300])
    for code, sig in out.items():
        assert (sig["train_last_future_date"] < sig["retrained_on"]).all()
        assert sig.index.min() >= DAYS[300]
    single = tr.walk_forward(tables[CODE]["features"], tables[CODE]["labels"], valid_start=DAYS[300], feature_set="base")
    assert (out[CODE]["train_rows"] > single.reindex(out[CODE].index)["train_rows"]).all()  # 두 종목 행을 합쳤다


def test_다른_종목_실시간_신호는_과거_검증과_같다(stored, monkeypatch):
    monkeypatch.setattr(config, "OPERATIONAL_EXTENDED_HISTORY", False)
    raw = mt.ticker_raw(CODE, stored["shared"])
    wf = tr.walk_forward(features.build_features(raw=raw), make_labels(raw["ohlcv"]["Close"]),
                         valid_start=DAYS[300], feature_set="base")
    for t in [DAYS[300], DAYS[333], DAYS[400]]:
        live = mt.live_signal(CODE, str(t.date()), valid_start=DAYS[300])
        assert live["signal"] == wf.loc[t, "signal"]
        assert live["confidence"] == pytest.approx(wf.loc[t, "confidence"])
        assert live["retrained_on"] == wf.loc[t, "retrained_on"]


def test_실시간_신호는_t_이후_종목_값을_쓰지_않는다(stored, monkeypatch):
    monkeypatch.setattr(config, "OPERATIONAL_EXTENDED_HISTORY", False)
    t = DAYS[333]
    before = mt.live_signal(CODE, str(t.date()))
    for kind in ("ohlcv", "flows"):
        df = stored["ticker"][kind]
        df.loc[df.index >= t] = df.loc[df.index >= t] * 3
        df.to_parquet(Path(config.TICKER_RAW_DIR) / f"{CODE}_{kind}.parquet")
    after = mt.live_signal(CODE, str(t.date()))
    assert (before["signal"], before["confidence"]) == (after["signal"], pytest.approx(after["confidence"]))


def test_다른_종목은_주문_모듈과_연결하지_않는다():
    text = Path("src/pipeline/multi_ticker.py").read_text(encoding="utf-8")
    assert not re.search(r"\bbroker\b|kis_client", text)


def test_종목_원본이_직전_거래일까지_없으면_실시간_신호를_만들지_않는다(stored, monkeypatch):
    monkeypatch.setattr(config, "OPERATIONAL_EXTENDED_HISTORY", False)
    t = DAYS[333]
    stale = stored["ticker"]["ohlcv"][stored["ticker"]["ohlcv"].index < DAYS[331]]
    stale.to_parquet(Path(config.TICKER_RAW_DIR) / f"{CODE}_ohlcv.parquet")
    with pytest.raises(ValueError, match="직전 거래일"):
        mt.live_signal(CODE, str(t.date()))


def test_구간_성과는_손계산과_같다():
    """가격 100·110·99·99·108.9, 신호 상승·보합·하락·상승·보합 (수수료 0.1%, 하락 = 현금)."""
    days = pd.bdate_range("2024-01-01", periods=5)
    close = pd.Series([100, 110, 99, 99, 108.9], index=days, dtype=float)
    signal = pd.Series(["상승", "보합", "하락", "상승", "보합"], index=days)
    labels = pd.DataFrame({"label": ["보합", "하락", "상승", "상승", "보합"],
                           "future_date": list(days[1:]) + [days[-1] + pd.Timedelta(days=7)]}, index=days)
    r = mt.period_report(signal, close, labels, days[0], days[-1])
    assert r["return"] == pytest.approx(0.085736, abs=1e-6)
    assert r["mdd"] == pytest.approx(-0.101800, abs=1e-5)
    assert r["A"] is False and r["B"] is False
    assert r["accuracy"] == pytest.approx(0.25) and r["most_common"] == pytest.approx(0.5)


def test_판정_경계():
    def row(code, model, calmar, mdd):
        return {"code": code, "model": model, "period": "확장(판정)", "calmar": calmar, "mdd": mdd, "A": False, "B": True, "C": None}
    codes = list(config.MULTI_TICKERS)
    rows = [row(c, "단독", 1.0, -0.3) for c in codes]
    rows += [row(codes[0], "공동", 1.0, -0.2),   # 칼마 같음 → 아님
             row(codes[1], "공동", 1.1, -0.3)]   # MDD 같음 → 나음
    v = mt.judge(pd.DataFrame(rows))
    assert v["Q2"] == {codes[0]: False, codes[1]: True} and v["Q2_adopt"] is False


def test_지표_지정_워크포워드는_운영_워크포워드와_같다(stored):
    raw = mt.ticker_raw(CODE, stored["shared"])
    feats, labels = features.build_features(raw=raw), make_labels(raw["ohlcv"]["Close"])
    a = mt._walk_forward_cols(feats, labels, list(config.FEATURE_SET_BASE), DAYS[300])
    b = tr.walk_forward(feats, labels, valid_start=DAYS[300], feature_set="base")
    assert (a["signal"] == b["signal"]).all()
    assert np.allclose(a["confidence"], b["confidence"])


def test_장_마감_전_기록은_다시_실행해도_덮어쓰지_않는다(tmp_path, monkeypatch):
    path = tmp_path / "live.parquet"
    monkeypatch.setattr(config, "MULTI_LIVE_PATH", str(path))
    t = pd.Timestamp("2026-10-23")
    old = pd.DataFrame({"code": ["000660", "012450"], "signal": ["상승", "하락"]}, index=[t, t])
    old["created_at"] = [t + pd.Timedelta(hours=15, minutes=20), t + pd.Timedelta(hours=21)]  # 하나는 실시간, 하나는 사후
    old.to_parquet(path)
    monkeypatch.setattr(mt, "_live_one", lambda code, as_of: {"date": t, "code": code, "name": code, "signal": "보합", "confidence": 0.5})
    monkeypatch.setattr(config, "MULTI_TICKERS", {"005930": ("삼성전자",), "000660": ("SK하이닉스",), "012450": ("한화",)})
    mt.live_signals("2026-10-23")
    saved = pd.read_parquet(path).set_index("code")["signal"]
    assert saved["000660"] == "상승"  # 15:20 실시간 기록 유지
    assert saved["012450"] == "보합"  # 사후 기록은 새 값으로 바뀜
