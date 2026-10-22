import numpy as np
import pandas as pd
import pytest

from src import config
from src.pipeline import experiment as ex


@pytest.fixture(autouse=True)
def small_model(monkeypatch):
    """테스트는 작은 LightGBM 으로 빠르게 돈다."""
    monkeypatch.setattr(config, "LGBM_PARAMS", {**config.LGBM_PARAMS, "n_estimators": 20, "min_child_samples": 5})
    monkeypatch.setattr(config, "MIN_TRAIN_ROWS", 30)
    monkeypatch.setattr(config, "VALID_START", "2025-05-01")
    monkeypatch.setattr(config, "DATA_END", "2025-10-31")


class FakeData(ex.Data):
    """파일 대신 무작위 가격·특징으로 만든 실험 데이터."""

    def __init__(self, scores=None):
        rng = np.random.default_rng(0)
        days = pd.bdate_range("2025-01-01", "2025-10-31")
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
        self.samsung = pd.DataFrame({"Open": close * 0.99, "Close": close}, index=days)
        self.inverse = pd.DataFrame({"Open": 1e4 / close, "Close": 1e4 / close}, index=days)
        self.features = pd.DataFrame({c: rng.normal(size=len(days)) for c in config.FEATURE_SETS["base"]}, index=days)
        self.scores = scores
        self._signals = {}


def test_기준_조건은_운영_설정과_같다():
    assert ex.expand(dict(config.EXPERIMENT_BASE), {}) == [config.EXPERIMENT_BASE]


def test_조합은_모든_경우를_만든다():
    cfgs = ex.expand(dict(config.EXPERIMENT_BASE), {"down_action": ["cash", "inverse"], "horizon_days": [3, 5]})
    assert len(cfgs) == 4 and {(c["down_action"], c["horizon_days"]) for c in cfgs} == {("cash", 3), ("cash", 5), ("inverse", 3), ("inverse", 5)}


def test_알_수_없는_조건은_거부한다():
    with pytest.raises(ValueError):
        ex.expand(dict(config.EXPERIMENT_BASE), {"signal_time": ["14:00"]})


def test_비교는_같은_구간에서_하고_학습은_조건별로_한_번만():
    data = FakeData()
    cfgs = ex.expand(dict(config.EXPERIMENT_BASE), {"down_action": ["cash", "inverse"], "execution": ["close", "next_open"]})
    table, curves = ex.compare(cfgs, data)
    assert len(table) == 4 and len(data._signals) == 1  # 학습 조건이 같으면 다시 학습하지 않는다
    assert table["start"].nunique() == 1 and table["end"].nunique() == 1 and table["days"].nunique() == 1  # 체결 시점이 섞여도 같은 날들
    assert "조건 1" in curves and any(k.startswith("들고 있기") for k in curves)


def test_다음날_시가_체결은_다음날_시가를_t_행에_둔다():
    ohlcv = pd.DataFrame({"Open": [10.0, 11.0, 12.0], "Close": [1.0, 2.0, 3.0]}, index=pd.bdate_range("2026-01-05", periods=3))
    px = ex.trade_prices(ohlcv, "next_open")
    assert px.iloc[:2].tolist() == [11.0, 12.0] and pd.isna(px.iloc[2])


def test_뉴스를_쓰는_조건이_있으면_뉴스_채점_구간으로_맞춘다():
    days = pd.bdate_range("2025-08-01", "2025-08-29")
    scores = pd.DataFrame({"signal_date": days, "news_consensus": 4, "news_consensus_conf": 0.9, "no_news": False})
    cfgs = ex.expand(dict(config.EXPERIMENT_BASE), {"news_mode": ["none", "adjust"]})
    table, _ = ex.compare(cfgs, FakeData(scores))
    assert set(table["start"]) == {"2025-08-01"} and set(table["end"]) == {"2025-08-29"}


def test_보고서에는_탐색과_다중_비교_경고가_붙는다():
    table, _ = ex.compare(ex.expand(dict(config.EXPERIMENT_BASE), {"down_action": ["cash", "inverse"]}), FakeData())
    md = ex.to_markdown(table)
    assert "탐색 결과" in md and "다중 비교 경고" in md and "하락 행동" in md


def test_다음날_시가_체결_손계산():
    days = pd.bdate_range("2025-06-02", periods=4)
    data = FakeData()
    data.samsung = pd.DataFrame({"Open": [10.0, 11, 12, 13], "Close": [1.0, 1, 1, 1]}, index=days)
    data.inverse = data.samsung.copy()
    sig = pd.DataFrame({"signal": ["상승", "보합", "하락", "보합"], "confidence": 0.5}, index=days)
    cfg = {**config.EXPERIMENT_BASE, "down_action": "cash", "execution": "next_open", "start": str(days[0].date()), "end": str(days[-1].date())}
    data._signals[tuple(cfg[k] for k in ex.TRAIN_KEYS)] = sig
    r = ex.run_one(cfg, data)
    assert r["days"] == 3 and r["trades"] == 2
    assert r["equity"].iloc[-1] == pytest.approx(0.999 * 13 / 11 * 0.999)
    assert r["hold_equity"].iloc[-1] == pytest.approx(0.999 * 13 / 11)


def test_작은_정답_기준도_기본값으로_바뀌지_않는다():
    close = pd.Series([100.0, 100.0, 101.0], index=pd.bdate_range("2026-01-05", periods=3))
    assert ex.make_labels(close, horizon=1, threshold=0.005)["label"].tolist() == ["보합", ex.UP]  # +1% 는 기준 0.5% 면 상승


def test_명령줄_빈_값은_기본_조건으로():
    assert ex.parse_value("horizon_days", "none") == config.EXPERIMENT_BASE["horizon_days"]
    assert ex.parse_value("news_risk_veto", "none") is None


def test_모니터링은_하나씩_바꾼_조건과_해석을_만든다(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MONITOR_AXES", {"down_action": ["inverse"], "execution": ["next_open"]})
    monkeypatch.setattr(config, "MONITOR_NEWS_AXES", {"news_min_score": [2], "news_field": ["news_macro_score"]})
    monkeypatch.setattr(config, "MONITOR_REPORT_PATH", str(tmp_path / "report.md"))
    monkeypatch.setattr(config, "MONITOR_HISTORY_PATH", str(tmp_path / "history.csv"))
    monkeypatch.setattr(config, "SCOREBOARD_PATH", str(tmp_path / "none.json"))
    days = pd.bdate_range("2025-08-01", "2025-08-29")
    scores = pd.DataFrame({"signal_date": days, "news_consensus": 4, "news_consensus_conf": 0.9, "no_news": False})
    result = ex.monitor(FakeData(scores))
    numeric = result["tables"]["숫자 조건 (검증 구간)"]
    assert numeric["name"].tolist()[0] == "기준" and len(numeric) == 3 and numeric["days"].nunique() == 1
    news = result["tables"]["뉴스 조건 (뉴스 채점 구간)"]
    assert news["error"].notna().sum() == 1  # 버전 1 점수표라 분류별 점수는 계산하지 못한다
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    for text in ["한눈에 보기", "탐색 결과", "다중 비교 경고", "조건별 해석", "달별 수익", "첫 실행", "계산하지 못했다"]:
        assert text in report
    ex.monitor(FakeData(scores))  # 두 번째 실행은 지난 실행과 비교한다
    assert "데이터가 같아 숫자가 같다" in (tmp_path / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("d_ret, d_mdd, expected", [
    (0.1, -0.1, "모두 좋아졌다"), (0.1, 0.1, "위험을 더 지고"), (-0.1, -0.1, "방어적"), (-0.1, 0.1, "모두 나빠졌다"), (0.001, 0.001, "거의 같다"),
])
def test_기준_대비_변화_분류(d_ret, d_mdd, expected):
    assert expected in ex.change_kind(d_ret, d_mdd)


def test_지표_그룹은_겹치지_않고_기존과_후보_묶음을_나눈다():
    from src.pipeline import train
    groups = config.FEATURE_GROUPS
    all_cols = [c for g in groups.values() for c in g]
    assert len(all_cols) == len(set(all_cols))
    assert set(all_cols) == set(config.FEATURE_SETS["candidate"])
    assert train.feature_set_columns(train.group_set_name(config.FEATURE_GROUPS_BASE)) == config.FEATURE_SETS["base"]  # 순서까지 같다


def test_그룹_묶음_이름은_고른_순서와_상관없고_모르는_그룹은_거부한다():
    from src.pipeline import train
    assert train.group_set_name(["flow", "price"]) == train.group_set_name(["price", "flow"]) == "groups:price+flow"
    with pytest.raises(ValueError):
        train.feature_set_columns("groups:price+nothing")
    with pytest.raises(ValueError):
        ex.run_one({**config.EXPERIMENT_BASE, "feature_set": "groups:nothing"}, FakeData())


@pytest.mark.parametrize("acc, ret, mdd, expected, rec", [
    (0.05, 0.0, -0.40, "정확도 좋아짐", True), (0.05, -0.1, -0.40, "수익 나빠짐", False),
    (0.0, 0.0, -0.30, "MDD 좋아짐", True), (0.0, 0.0, -0.40, "거의 같음", False), (-0.05, 0.2, -0.40, "정확도 나빠짐", False),
])
def test_지표_진단_추천_규칙(acc, ret, mdd, expected, rec):
    b = pd.Series({"accuracy": 0.3, "total_return": 1.0, "mdd": -0.40})
    verdict, recommend = ex.study_verdict(pd.Series({"accuracy": 0.3 + acc, "total_return": 1.0 + ret, "mdd": mdd}), b)
    assert expected in verdict and recommend is rec


def test_지표_진단은_그룹을_빼고_더해_보고_리포트를_남긴다(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FEATURE_STUDY_REPORT_PATH", str(tmp_path / "features.md"))
    result = ex.feature_study(FakeData())  # 가짜 데이터에는 후보 지표가 없어 더하기 줄은 계산하지 못하고 빠진다
    removed = result["table"][result["table"]["change"].str.startswith("빼기")]
    assert len(removed) == len(config.FEATURE_GROUPS_BASE)
    report = (tmp_path / "features.md").read_text(encoding="utf-8")
    for text in ["탐색 결과", "다중 비교 경고", "기준선", "'상승' 이라고만", "추천 규칙", "## 추천"]:
        assert text in report


def test_인버스와_분할_매매는_섞지_않는다():
    with pytest.raises(ValueError):
        ex.run_one({**config.EXPERIMENT_BASE, "down_action": "inverse", "position_steps": 2}, FakeData())
    cfgs = ex.expand(dict(config.EXPERIMENT_BASE), {"down_action": ["cash", "inverse"], "position_steps": [1, 2]})
    table, _ = ex.compare(cfgs, FakeData())
    assert len(table) == 3 and table.attrs["skipped"] == 1 and "계산하지 않았다" in ex.to_markdown(table)
