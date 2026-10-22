import numpy as np
import pandas as pd
import pytest

from src import config
from src.pipeline import scoreboard as sb
from src.pipeline.labels import DOWN, FLAT, UP

DAYS = pd.bdate_range("2026-10-21", periods=40)


@pytest.fixture(autouse=True)
def fee(monkeypatch):
    monkeypatch.setattr(config, "FEE_RATE", 0.001)
    monkeypatch.setattr(config, "DRIFT_WINDOW", 10)


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
