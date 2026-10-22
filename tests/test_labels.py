import pandas as pd
import pytest

from src import config
from src.pipeline import labels as lb


@pytest.fixture(autouse=True)
def fixed_rule(monkeypatch):
    """테스트는 5거래일·3% 기준으로 고정한다."""
    monkeypatch.setattr(config, "LABEL_HORIZON_DAYS", 5)
    monkeypatch.setattr(config, "LABEL_THRESHOLD", 0.03)


def make_close(values):
    """거래일 순서대로 종가 시계열을 만든다."""
    return pd.Series(values, index=pd.bdate_range("2026-02-02", periods=len(values)), dtype=float)


def test_5거래일_뒤_종가로_판정한다():
    close = make_close([100, 1, 1, 1, 1, 103, 1])
    result = lb.make_labels(close)
    assert result.iloc[0]["label"] == lb.UP
    assert result.iloc[0]["future_date"] == close.index[5]


@pytest.mark.parametrize("future, expected", [
    (103.0, lb.UP),     # 정확히 +3% 는 상승
    (102.99, lb.FLAT),
    (97.01, lb.FLAT),
    (97.0, lb.DOWN),    # 정확히 -3% 는 하락
])
def test_경계값(future, expected):
    close = make_close([100, 100, 100, 100, 100, future])
    assert lb.make_labels(close).iloc[0]["label"] == expected


def test_마지막_5거래일은_정답이_없다():
    close = make_close(range(100, 110))
    result = lb.make_labels(close)
    assert len(result) == 5
    assert result.index.max() == close.index[4]


def test_분포는_구간별_비율을_돌려준다():
    close = make_close([100] * 5 + [110] * 5 + [100] * 5)
    labels = lb.make_labels(close)
    dist = lb.label_distribution(labels, {"전체": (str(close.index[0].date()), str(close.index[-1].date()))})
    row = dist.iloc[0]
    assert row["일수"] == 10
    assert row[lb.UP] + row[lb.FLAT] + row[lb.DOWN] == pytest.approx(100)


def test_정답_기간과_기준을_인자로_바꿀_수_있다():
    close = pd.Series([100, 101, 101.5, 104], index=pd.bdate_range("2026-01-05", periods=4))
    out = lb.make_labels(close, horizon=2, threshold=0.02)
    assert out["label"].tolist() == [lb.FLAT, lb.UP]  # 2거래일 뒤 +1.5% 는 보합, +3.0% 는 상승


def test_하락_위험_정답은_기간_안_최저_종가로_정한다():
    from src.pipeline.labels import make_risk_labels
    days = pd.bdate_range("2026-01-05", periods=8)
    close = pd.Series([100, 101, 96.9, 105, 106, 107, 108, 109.0], index=days)
    out = make_risk_labels(close, horizon=3, threshold=0.03)
    assert list(out.index) == list(days[:5])  # 마지막 3일은 정답이 없다
    assert out["risk"].tolist() == [1, 1, 0, 0, 0]  # 1·2일째는 3일 안에 96.9(-3.1%, -4.1%)
    assert out["future_date"].iloc[0] == days[3]
