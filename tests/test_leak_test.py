import pandas as pd
import pytest

from src.pipeline import leak_test as lt


def test_이항검정_p값():
    assert lt.binomial_p_at_least(0, 10, 0.6) == pytest.approx(1.0)
    assert lt.binomial_p_at_least(10, 10, 0.5) == pytest.approx(0.5**10)


def make_labels():
    days = pd.bdate_range("2026-02-02", periods=10)
    return pd.DataFrame({"label": ["상승"] * 6 + ["보합"] * 2 + ["하락"] * 2}, index=days)


def test_기준선_수준의_적중과_이후_사건_없음이면_통과(monkeypatch):
    monkeypatch.setattr(lt.config, "CRITERION_B_START", "2026-02-01")
    monkeypatch.setattr(lt.config, "CRITERION_B_END", "2026-02-28")
    labels = make_labels()
    result = pd.DataFrame({"answer": ["상승"] * 10, "truth": labels["label"].tolist(), "events_after_dated": [0] * 10})
    verdict = lt.judge(result, labels)
    assert verdict["hits"] == 6 and verdict["passed"]


def test_이후_날짜_사건을_말하면_실패(monkeypatch):
    monkeypatch.setattr(lt.config, "CRITERION_B_START", "2026-02-01")
    monkeypatch.setattr(lt.config, "CRITERION_B_END", "2026-02-28")
    labels = make_labels()
    result = pd.DataFrame({"answer": ["상승"] * 10, "truth": labels["label"].tolist(), "events_after_dated": [0] * 9 + [1]})
    assert not lt.judge(result, labels)["passed"]


def test_모두_맞히면_실패(monkeypatch):
    monkeypatch.setattr(lt.config, "CRITERION_B_START", "2026-02-01")
    monkeypatch.setattr(lt.config, "CRITERION_B_END", "2026-02-28")
    labels = make_labels()
    result = pd.DataFrame({"answer": labels["label"].tolist() * 2, "truth": labels["label"].tolist() * 2,
                           "events_after_dated": [0] * 20})
    assert not lt.judge(result, labels)["pass_accuracy"]


@pytest.mark.parametrize("text, expected", [
    ("2026-03-12", "after"), ("2026-03-11", "after"),  # 당일도 이후로 센다
    ("2026-03-10", "before"),
    ("2026-03", "unclear"), (" ", "unclear"), ("04/01/2026", "unclear"),
    (" 2026-04-01 ", "after"),  # 앞뒤 공백은 지운다
])
def test_사건_날짜_분류(text, expected):
    assert lt.classify_event_date(text, pd.Timestamp("2026-03-11")) == expected


def test_날짜가_불분명한_사건도_실패로_센다(monkeypatch):
    monkeypatch.setattr(lt.config, "CRITERION_B_START", "2026-02-01")
    monkeypatch.setattr(lt.config, "CRITERION_B_END", "2026-02-28")
    labels = make_labels()
    result = pd.DataFrame({"answer": ["상승"] * 10, "truth": labels["label"].tolist(),
                           "events_after_dated": [0] * 10, "events_undated": [0] * 9 + [1]})
    assert not lt.judge(result, labels)["passed"]


def test_익명화하면_회사명과_날짜가_빠진다():
    q = lt.question("2026-03-11", anonymize=True)
    assert "삼성" not in q and "2026" not in q
