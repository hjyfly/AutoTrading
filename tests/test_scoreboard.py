import numpy as np
import pandas as pd
import pytest

from src import config
from src.pipeline import scoreboard as sb
from src.pipeline.labels import DOWN, FLAT, UP

DAYS = pd.bdate_range("2026-10-21", periods=40)


@pytest.fixture(autouse=True)
def fee(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FEE_RATE", 0.001)
    monkeypatch.setattr(config, "DRIFT_WINDOW", 10)
    monkeypatch.setattr(config, "MULTI_LIVE_PATH", str(tmp_path / "no_tickers.parquet"))  # 실제 다른 종목 기록을 읽지 않는다


def prices(n=40, seed=0):
    rng = np.random.default_rng(seed)
    samsung = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, n))), index=DAYS[:n])
    inverse = 100 / samsung * 100
    return samsung, inverse


def test_정답_기간이_지난_신호만_채점한다():
    samsung, inverse = prices(8)
    signals = pd.Series([UP] * 8, index=DAYS[:8])
    s = sb.score_model(signals, samsung, inverse)
    h = config.LABEL_HORIZON_DAYS
    assert s["n_signals"] == 8 and s["n_matured"] == 8 - h  # 8거래일 중 앞 8-h 일만 h 거래일 뒤 종가가 있다


def test_기록이_빠진_날은_포지션을_유지한다():
    samsung, inverse = prices(10)
    sparse = pd.Series([UP, DOWN], index=[DAYS[0], DAYS[5]])
    full = pd.Series([UP] * 5 + [DOWN] * 5, index=DAYS[:10])
    a = sb.score_model(sparse, samsung, inverse)
    b = sb.score_model(full, samsung, inverse)
    assert a["model"]["total_return"] == pytest.approx(b["model"]["total_return"])


def test_정확도가_최빈_정답_규칙보다_낮으면_경고():
    labels = pd.Series([UP] * 10, index=DAYS[:10])
    bad = pd.Series([DOWN] * 10, index=DAYS[:10])
    assert sb.drift_check(bad, labels)["status"] == "경고"
    assert sb.drift_check(labels.copy(), labels)["status"] == "정상"
    assert sb.drift_check(bad.iloc[:5], labels)["status"] == "표본 부족"


def test_성적표를_만들고_요약한다(tmp_path, monkeypatch):
    samsung, inverse = prices(30)
    live = pd.DataFrame({"model": ["base"] * 20 + ["candidate"] * 20,
                         "signal": [UP] * 20 + [FLAT] * 20, "confidence": 0.5},
                        index=list(DAYS[:20]) * 2)
    live["created_at"] = live.index + pd.Timedelta(hours=15, minutes=20)  # 장중에 실제로 만든 기록
    live.to_parquet(tmp_path / "live.parquet")
    monkeypatch.setattr(sb, "LIVE_PATH", tmp_path / "live.parquet")
    monkeypatch.setattr(sb, "load_prices", lambda until: (samsung, inverse))
    monkeypatch.setattr(config, "SCOREBOARD_PATH", str(tmp_path / "board.json"))
    board = sb.build(DAYS[-1])
    assert set(board["models"]) == {"base", "candidate"}
    assert board["models"]["base"]["n_matured"] == 20
    assert board["models"]["candidate"]["model"]["total_return"] == pytest.approx(0)  # 계속 보합이면 현금
    assert len(sb.summary_lines(board)) == 3  # 모델 2줄 + 가격 기준일


def test_기록이_없으면_없다고_답한다(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "LIVE_PATH", tmp_path / "none.parquet")
    monkeypatch.setattr(config, "SCOREBOARD_PATH", str(tmp_path / "board.json"))
    assert sb.summary_lines(sb.build(DAYS[-1])) == ["아직 실시간 기록이 없다"]


def test_장_마감_뒤에_만든_신호는_실시간_성적에서_뺀다():
    days = pd.DatetimeIndex(["2026-10-21", "2026-10-22"])
    rows = pd.DataFrame({"created_at": pd.to_datetime(["2026-10-21 16:35", "2026-10-22 15:21"])}, index=days)
    assert sb.realtime_mask(rows).tolist() == [False, True]
    assert not sb.realtime_mask(pd.DataFrame(index=days)).any()  # 생성 시각이 없으면 사후 기록으로 본다


def test_성적표는_현재_규칙을_기본으로_두_규칙을_함께_계산한다():
    samsung, inverse = prices(12)
    s = sb.score_model(pd.Series([UP, DOWN] * 6, index=DAYS[:12]), samsung, inverse)
    assert set(s["by_rule"]) == set(config.DOWN_ACTIONS)
    assert s["model"] == s["by_rule"][config.DOWN_ACTION]


def test_목표_비중_비교_모델은_비중대로_채점한다():
    days = pd.bdate_range("2026-01-05", periods=10)
    samsung = pd.Series([100.0, 102, 104, 100, 96, 95, 97, 99, 101, 103], index=days)
    rows = pd.DataFrame({"weight": [1.0, 0.0]}, index=[days[0], days[3]])  # 4일째부터 현금, 빠진 날은 앞 비중 유지
    s = sb.score_weight_model(rows, samsung)
    assert s["n_signals"] == 2 and s["down_action"] == "cash"
    assert s["model"]["total_return"] == pytest.approx(0.999 * 1.0 * (100 / 100) * 0.999 - 1)
    assert s["risk_hit"] == 1.0  # 4일째(100) 뒤 5거래일 안에 95(-5%)


def test_성공_기준은_표본이_모자라면_참고로_표시하고_승격은_장부_등록을_본다(tmp_path, monkeypatch):
    samsung, inverse = prices(40)
    live = pd.DataFrame({"model": ["base"] * 30 + ["candidate"] * 30 + ["groups:price"] * 30,
                         "signal": ([UP] * 15 + [DOWN] * 15) * 3, "confidence": 0.5}, index=list(DAYS[:30]) * 3)
    live["created_at"] = live.index + pd.Timedelta(hours=15, minutes=20)
    live.to_parquet(tmp_path / "live.parquet")
    monkeypatch.setattr(sb, "LIVE_PATH", tmp_path / "live.parquet")
    monkeypatch.setattr(sb, "load_prices", lambda until: (samsung, inverse))
    monkeypatch.setattr(config, "SCOREBOARD_PATH", str(tmp_path / "board.json"))
    monkeypatch.setattr(config, "LIVE_EXTRA_FEATURE_SETS", ["groups:price"])
    board = sb.build(DAYS[-1])
    c = board["models"]["base"]["criteria"]
    assert not c["ready"] and "참고" in c["status"] and set(c) >= {"A", "B", "C"}
    assert board["models"]["candidate"]["promotion"]["registered"]
    assert not board["models"]["groups:price"]["promotion"]["registered"]
    assert not board["models"]["candidate"]["promotion"]["eligible"]  # 확정 신호 40개 미만
    assert any("성공 기준[" in line for line in sb.summary_lines(board))
    monkeypatch.setattr(config, "LIVE_PROMOTION_MIN_SIGNALS", 10)
    assert sb.build(DAYS[-1])["models"]["base"]["criteria"]["ready"]


def test_성공_기준_손계산():
    days = pd.bdate_range("2026-01-28", periods=12)
    samsung = pd.Series([100, 110, 99, 104.5, 115, 116, 117, 118, 119, 120, 121, 122.0], index=days)
    signals = pd.Series([UP, FLAT, DOWN, UP, FLAT, DOWN, FLAT, UP], index=days[:8])
    s = sb.score_model(signals, samsung, samsung * 0 + 100)
    c = s["criteria"]
    assert s["model"]["total_return"] == pytest.approx(0.130530, abs=1e-5)
    assert s["model"]["mdd"] == pytest.approx(-0.101799, abs=1e-5)
    assert (c["A"]["mdd_ok"], c["A"]["return_ok"], c["A"]["passed"]) == (False, True, False)
    assert c["B"]["hit"] == 0 and c["B"]["base"] == 0 and c["B"]["passed"] is False
    assert c["C"]["hold_share"] == pytest.approx(7 / 9) and c["C"]["up_months"] == 1 and c["C"]["passed"]


def test_승격은_시작일이_다르면_보류한다():
    base = {"first_date": "2026-10-22", "model": {"calmar": 1.0, "mdd": -0.2}}
    later = {"first_date": "2026-10-23", "n_matured": 50, "model": {"calmar": 2.0, "mdd": -0.1}}
    p = sb.promotion(later, base, "candidate")
    assert not p["eligible"] and "보류" in p["note"]


def test_다른_종목은_종목_알고리즘마다_그_종목_가격으로_채점한다(tmp_path, monkeypatch):
    samsung, inverse = prices(30)
    other, _ = prices(30, seed=1)
    live = pd.DataFrame({"model": ["base"] * 20}, index=DAYS[:20]).assign(signal=UP, confidence=0.5)
    live["created_at"] = live.index + pd.Timedelta(hours=15, minutes=20)
    live.to_parquet(tmp_path / "live.parquet")
    tickers = pd.DataFrame({"code": ["277810"] * 20 + ["000660"] * 20,
                            "model": [None] * 5 + ["S4 분야 공동 학습"] * 15 + ["S0 단독"] * 20,
                            "signal": [UP] * 40, "confidence": 0.5}, index=list(DAYS[:20]) * 2)
    tickers["created_at"] = tickers.index + pd.Timedelta(hours=15, minutes=20)
    tickers.iloc[0, tickers.columns.get_loc("created_at")] += pd.Timedelta(hours=6)  # 장 마감 뒤 사후 기록 1건
    tickers.to_parquet(tmp_path / "tickers.parquet")
    monkeypatch.setattr(sb, "LIVE_PATH", tmp_path / "live.parquet")
    monkeypatch.setattr(config, "MULTI_LIVE_PATH", str(tmp_path / "tickers.parquet"))
    monkeypatch.setattr(sb, "load_prices", lambda until: (samsung, inverse))
    monkeypatch.setattr(sb, "ticker_close", lambda code, until: other)
    monkeypatch.setattr(config, "SCOREBOARD_PATH", str(tmp_path / "board.json"))
    board = sb.build(DAYS[-1])
    t = board["tickers"]
    assert set(t) == {"277810|S0 단독", "277810|S4 분야 공동 학습", "000660|S0 단독"}  # 빈 알고리즘은 S0 단독
    assert t["277810|S0 단독"]["n_signals"] == 4 and t["277810|S0 단독"]["n_backfilled"] == 1
    assert t["000660|S0 단독"]["n_matured"] == 20 and "criteria" in t["000660|S0 단독"]
    assert set(t["000660|S0 단독"]["by_rule"]) == {"cash"}  # 인버스 가격이 없으니 현금 규칙만
    hold = other[other.index >= DAYS[0]]
    assert t["000660|S0 단독"]["hold"]["total_return"] == pytest.approx(hold.iloc[-1] / hold.iloc[0] * (1 - config.FEE_RATE) - 1)  # 그 종목 들고 있기(매수 수수료 포함)
    assert len(sb.ticker_lines(board)) == 3 and "promotion" not in t["000660|S0 단독"]


def test_다른_종목_채점이_실패해도_삼성전자_성적표는_만든다(tmp_path, monkeypatch):
    samsung, inverse = prices(30)
    live = pd.DataFrame({"model": ["base"] * 20}, index=DAYS[:20]).assign(signal=UP, confidence=0.5)
    live["created_at"] = live.index + pd.Timedelta(hours=15, minutes=20)
    live.to_parquet(tmp_path / "live.parquet")
    pd.DataFrame({"signal": [UP]}, index=DAYS[:1]).to_parquet(tmp_path / "broken.parquet")  # code 열이 없는 깨진 기록
    monkeypatch.setattr(sb, "LIVE_PATH", tmp_path / "live.parquet")
    monkeypatch.setattr(config, "MULTI_LIVE_PATH", str(tmp_path / "broken.parquet"))
    monkeypatch.setattr(sb, "load_prices", lambda until: (samsung, inverse))
    monkeypatch.setattr(config, "SCOREBOARD_PATH", str(tmp_path / "board.json"))
    board = sb.build(DAYS[-1])
    assert board["models"]["base"]["n_matured"] == 20 and board["tickers"] == {}
    assert sb.ticker_lines(board)[0].startswith("다른 종목 성적표 계산 실패")


def test_다른_종목은_운영_규칙이_바뀌어도_현금으로_판정하고_한_종목_실패는_그_줄만(tmp_path, monkeypatch):
    other, _ = prices(30, seed=1)
    tickers = pd.DataFrame({"code": ["000660"] * 10 + ["012450"] * 10, "model": "S0 단독",
                            "signal": [UP, DOWN] * 10, "confidence": 0.5}, index=list(DAYS[:10]) * 2)
    tickers["created_at"] = tickers.index + pd.Timedelta(hours=15, minutes=20)
    tickers.to_parquet(tmp_path / "tickers.parquet")
    monkeypatch.setattr(config, "MULTI_LIVE_PATH", str(tmp_path / "tickers.parquet"))
    monkeypatch.setattr(config, "DOWN_ACTION", "inverse")
    def close(code, until):
        if code == "012450":
            raise ValueError("원본 없음")
        return other
    monkeypatch.setattr(sb, "ticker_close", close)
    t = sb.score_tickers(DAYS[-1])
    assert t["000660|S0 단독"]["down_action"] == "cash" and "criteria" in t["000660|S0 단독"]
    assert "criteria" not in t["012450|S0 단독"] and t["012450|S0 단독"]["n_matured"] is None
    assert "원본 없음" in t["012450|S0 단독"]["drift"]["status"]
    assert sb.score_tickers(DAYS[5])["000660|S0 단독"]["n_signals"] == 5  # 기준일 뒤 기록은 세지 않는다
