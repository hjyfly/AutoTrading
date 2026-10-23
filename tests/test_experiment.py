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


def test_뉴스_평가_지표_손계산():
    import numpy as np
    from src.pipeline import experiment as ex
    idx = pd.bdate_range("2025-01-01", periods=10)
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=idx, dtype=float)
    r = pd.Series([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7], index=idx, dtype=float) / 100
    m = ex.score_metrics(s, r, 0)
    assert m["ic"] == pytest.approx(1.0) and m["hit"] == pytest.approx(7 / 10)  # 점수는 모두 양수, 수익 양수 7일
    assert m["spread"] == pytest.approx((0.065) - (-0.015))  # 상위 20%(9·10) 평균 − 하위 20%(1·2) 평균
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=200), index=pd.bdate_range("2025-01-01", periods=200))
    y = 0.3 * x + pd.Series(rng.normal(size=200), index=x.index)
    z = (x - x.mean()) / x.std()
    X = np.column_stack([np.ones(200), z])
    b = np.linalg.lstsq(X, y.values, rcond=None)[0]
    u = y.values - X @ b
    inv = np.linalg.inv(X.T @ X)
    hc0 = inv @ (X.T * u ** 2) @ X @ inv  # 지연 0 이면 Newey-West = White(HC0)
    assert ex.newey_west_t(x, y, 0) == pytest.approx(b[1] / np.sqrt(hc0[1, 1]))


# ---- 11일차 기간별 신호 ----

def test_N거래일마다_방식은_N일째에만_신호를_따르고_사이는_유지한다():
    idx = pd.bdate_range("2025-01-01", periods=7)
    sig = pd.Series(["상승", "하락", "하락", "하락", "상승", "하락", "하락"], index=idx)
    assert ex.every_n(sig, 3).tolist() == ["상승", "보합", "보합", "하락", "보합", "보합", "하락"]
    assert ex.every_n(sig, 1).tolist() == sig.tolist()


def test_기간별_조건과_경로():
    assert ex.horizon_cfg(None) == {**config.EXPERIMENT_BASE, "threshold": config.HORIZONS[config.OPERATIONAL_HORIZON]}
    assert ex.horizon_cfg(20)["horizon_days"] == 20 and ex.horizon_cfg(20)["threshold"] == 0.06
    assert ex.horizon_path("a/b.md", None).name == "b.md" and ex.horizon_path("a/b.md", config.OPERATIONAL_HORIZON).name == "b.md"
    assert ex.horizon_path("a/b_000660.md", 20).name == "b_000660_h20.md"


def test_기간별_구간_성적은_구간_끝까지_확정된_정답만_쓴다():
    data = FakeData()
    cfg = ex.horizon_cfg(10)
    sig = data.signals(cfg)
    labels = data.labels(10, cfg["threshold"])
    close = data.samsung["Close"]
    r = ex.horizon_period(sig, close, labels, "2025-05-01", "2025-06-30", 10)
    part = sig.loc["2025-05-01":"2025-06-30"]
    expect = labels[labels["future_date"] <= part.index.max()].index.intersection(part.index)
    assert r["labeled"] == len(expect) and r["independent"] == len(expect) // 10
    assert r["days"] == len(part) and set(r["every"]) == {"model", "judge_a", "hold_share"}
    assert abs(sum(r["label_shares"].values()) - 1) < 1e-9
    assert r["calibration"]["count"].sum() == len(expect)


def test_H_판정은_칼마와_MDD_로_한_번만_한다(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DAY11_JUDGED_PATH", str(tmp_path / "judged.json"))
    row = lambda calmar, mdd: {"start": "2022-03-04", "end": "2025-04-22", "days": 769, "model": {"calmar": calmar, "mdd": mdd, "total_return": 0.1}}
    rows = {5: row(1.0, -0.2), 1: row(1.5, -0.2), 3: row(1.5, -0.3), 10: row(0.9, -0.1), 20: row(float("nan"), 0.0)}
    v = ex.judge_day11(rows, ("2022-03-04", "2025-04-22"))
    assert not v["recorded"]
    assert [v["verdict"][h]["passed"] for h in ("H1", "H2", "H3", "H4")] == [True, False, False, False]
    with pytest.raises(ValueError):
        ex.judge_day11(rows, ("2022-03-04", "2025-04-30"))  # 등록 구간과 다르면 멈춘다
    with pytest.raises(ValueError):
        ex.judge_day11({**rows, 5: row(float("nan"), 0.0)})  # 기준 칼마가 없으면 멈춘다
    assert not (tmp_path / "judged.json").exists()  # 판정 함수는 기록하지 않는다
    ex.record_day11(v)
    again = ex.judge_day11({**rows, 3: row(9.0, -0.01)})  # 숫자가 바뀌어도 기록된 판정을 돌려준다
    assert again["recorded"] and again["verdict"]["H2"]["passed"] is False


def test_기간별_모니터링은_뉴스_조건을_빼고_h_경로에_저장한다(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "HORIZON_MONITOR_AXES", {"execution": ["next_open"]})
    monkeypatch.setattr(config, "MONITOR_REPORT_PATH", str(tmp_path / "report.md"))
    monkeypatch.setattr(config, "MONITOR_HISTORY_PATH", str(tmp_path / "history.csv"))
    days = pd.bdate_range("2025-08-01", "2025-08-29")
    scores = pd.DataFrame({"signal_date": days, "news_consensus": 4, "news_consensus_conf": 0.9, "no_news": False})
    result = ex.monitor(FakeData(scores), horizon=10)
    assert list(result["tables"]) == ["숫자 조건 (검증 구간)"]
    table = result["tables"]["숫자 조건 (검증 구간)"]
    assert set(table["horizon_days"]) == {10} and set(table["threshold"]) == {config.HORIZONS[10]}
    assert (tmp_path / "report_h10.md").exists() and (tmp_path / "history_h10.csv").exists() and not (tmp_path / "report.md").exists()


def test_기간별_분석은_모든_기간을_같은_구간에서_계산하고_리포트를_남긴다(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "HORIZON_DIR", str(tmp_path / "h"))
    monkeypatch.setattr(config, "DAY11_JUDGED_PATH", str(tmp_path / "judged.json"))
    monkeypatch.setattr(config, "DAY11_REPORT_PATH", str(tmp_path / "day11.md"))
    monkeypatch.setattr(config, "EXT_VALID_END", "2025-07-31")
    monkeypatch.setattr(config, "HORIZONS", {1: 0.013, 5: 0.03, 10: 0.042})
    monkeypatch.setattr(config, "DAY11_LEDGER", {1: "H1", 10: "H3"})

    def fake(extended=False):
        d = FakeData()
        d.valid_start = config.VALID_START
        return d
    monkeypatch.setattr(ex, "Data", fake)
    out = ex.horizon_study()
    starts = {n: r["rows"][0]["start"] for n, r in out["result"].items()}
    assert len(set(starts.values())) == 1
    assert set(out["verdict"]["verdict"]) == {"H1", "H3"}
    report = (tmp_path / "day11.md").read_text(encoding="utf-8")
    for text in ["장부 H1 ~ H4 판정", "확장 구간 한눈에", "연도별 장세", "확신도 보정", "정답 분포", "N일마다"]:
        assert text in report
    assert (tmp_path / "h" / "signals_numeric_h10.parquet").exists() and (tmp_path / "judged.json").exists()
