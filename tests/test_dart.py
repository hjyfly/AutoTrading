import pandas as pd
import pytest

from src.pipeline.collect_dart import dart_as_of

INDEX = pd.DataFrame({
    "report_nm": ["분기보고서 (2025.09)", "사업보고서 (2025.12)"],
    "rcept_dt": pd.to_datetime(["2025-11-14", "2026-03-10"]),
})


@pytest.mark.parametrize("as_of", [
    "2026-03-10",
    "2026-03-10 15:15",
    "2026-03-10 23:59",
    pd.Timestamp("2026-03-10 15:15", tz="Asia/Seoul"),
])
def test_접수일_당일에는_그_공시를_쓰지_않는다(as_of):
    assert dart_as_of(INDEX, as_of)["report_nm"].tolist() == ["분기보고서 (2025.09)"]


def test_접수_다음날부터_쓴다():
    assert len(dart_as_of(INDEX, "2026-03-11")) == 2


def test_UTC_시각은_한국_날짜로_바꿔_비교한다():
    # 2026-03-10 16:00 UTC = 2026-03-11 01:00 KST → 3월 10일 공시를 쓸 수 있다
    assert len(dart_as_of(INDEX, pd.Timestamp("2026-03-10 16:00", tz="UTC"))) == 2
