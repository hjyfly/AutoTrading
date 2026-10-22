import json

import pandas as pd
import pytest

from src import config
from src.pipeline import refresh_data as rd


def frame(days, values, extra=None):
    df = pd.DataFrame({"Close": values, "source": "x", "collected_at": "t", "is_dummy": False}, index=pd.DatetimeIndex(days))
    return df


def test_새_날짜만_덧붙이고_저장값은_그대로_둔다():
    stored = frame(["2026-10-19", "2026-10-20"], [100.0, 101.0])
    fresh = frame(["2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"], [999.0, 102.0, 103.0, 104.0])
    combined, added, mismatch = rd.append_new(stored, fresh, pd.Timestamp("2026-10-22"))
    assert combined.loc["2026-10-20", "Close"] == 101.0  # 겹친 날은 저장값 유지
    assert added == 2 and mismatch == 1
    assert combined.index.max() == pd.Timestamp("2026-10-22")  # 확정되지 않은 날(23일)은 버린다
    assert combined.iloc[:2].equals(stored)


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setattr(config, "RAW_BACKUP_DIR", str(tmp_path / "backup"))
    monkeypatch.setattr(rd, "series_list", lambda: [("prices", "samsung_ohlcv"), ("macro", "sox")])
    monkeypatch.setattr(rd, "last_closed_date", lambda kind, name, now=None: pd.Timestamp("2026-10-22"))
    monkeypatch.setattr(rd, "refresh_cap", lambda now=None: pd.Timestamp("2026-10-22"))
    for kind, name in rd.series_list():
        (tmp_path / "raw" / kind).mkdir(parents=True)
        frame(["2026-10-19", "2026-10-20"], [100.0, 101.0]).to_parquet(tmp_path / "raw" / kind / f"{name}.parquet")
    return tmp_path


def test_모두_받으면_덧붙이고_원본을_백업한다(raw_dir, monkeypatch):
    monkeypatch.setattr(rd, "fetch_series", lambda kind, name, s, e: frame(["2026-10-20", "2026-10-21", "2026-10-22"], [101.0, 102.0, 103.0]))
    table = rd.refresh_raw(stamp="t")
    assert table["written"].all() and (table["added"] == 2).all()
    assert len(pd.read_parquet(raw_dir / "raw" / "prices" / "samsung_ohlcv.parquet")) == 4
    assert len(pd.read_parquet(raw_dir / "backup" / "raw_t" / "prices" / "samsung_ohlcv.parquet")) == 2


def test_하나라도_실패하면_아무것도_쓰지_않는다(raw_dir, monkeypatch):
    def fetch(kind, name, s, e):
        if name == "sox":
            raise TimeoutError("응답 없음")
        return frame(["2026-10-21"], [102.0])
    monkeypatch.setattr(rd, "fetch_series", fetch)
    table = rd.refresh_raw(stamp="t")
    assert not table["written"].any()
    assert len(pd.read_parquet(raw_dir / "raw" / "prices" / "samsung_ohlcv.parquet")) == 2
    with pytest.raises(RuntimeError, match="아무 파일도"):
        monkeypatch.setattr(rd, "refresh_raw", lambda now=None, stamp=None: table)
        rd.run()


def test_새_확정_행이_없으면_받지_않는다(raw_dir, monkeypatch):
    monkeypatch.setattr(rd, "last_closed_date", lambda kind, name, now=None: pd.Timestamp("2026-10-20"))
    monkeypatch.setattr(rd, "fetch_series", lambda *a: (_ for _ in ()).throw(AssertionError("받으면 안 된다")))
    assert (rd.refresh_raw(stamp="t")["note"] == "새 확정 행 없음").all()


def test_이미_있던_날의_신호_변화를_센다():
    days = pd.bdate_range("2026-10-01", periods=3)
    before = pd.DataFrame({"signal": ["상승", "보합", "하락"]}, index=days)
    after = pd.DataFrame({"signal": ["상승", "상승", "하락", "보합"]}, index=pd.bdate_range("2026-10-01", periods=4))
    assert rd.signal_changes(before, after) == 1


def test_갱신_상태_파일이_없으면_처음_끝날을_쓴다(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_STATUS_PATH", str(tmp_path / "missing.json"))
    assert config._refreshed_data_end() is None
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"data_end": "2026-10-23"}), encoding="utf-8")
    monkeypatch.setattr(config, "DATA_STATUS_PATH", str(status))
    assert config._refreshed_data_end() == "2026-10-23"


def test_갱신은_실행일_전날까지만_받는다():
    # 금요일 15:31 KST 에 돌아도 금요일 행(장 마감 직후 잠정 수급)은 받지 않는다
    assert rd.refresh_cap(pd.Timestamp("2026-10-23 15:31", tz="Asia/Seoul")) == pd.Timestamp("2026-10-22")


def test_국내_원본끼리_끝날이_다르면_쓰지_않는다(raw_dir, monkeypatch):
    monkeypatch.setattr(rd, "series_list", lambda: [("prices", "samsung_ohlcv"), ("prices", "inv")])
    frame(["2026-10-19", "2026-10-20"], [1.0, 1.0]).to_parquet(raw_dir / "raw" / "prices" / "inv.parquet")
    monkeypatch.setattr(rd, "fetch_series", lambda kind, name, s, e: frame(["2026-10-21"], [1.0]) if name == "samsung_ohlcv"
                        else frame([], []))  # 인버스는 예외 없이 빈 표
    table = rd.refresh_raw(stamp="t")
    assert not table["written"].any() and table["note"].str.contains("끝날 불일치").all()
    assert len(pd.read_parquet(raw_dir / "raw" / "prices" / "samsung_ohlcv.parquet")) == 2


def test_다시_만들다_실패하면_원본과_끝날을_되돌린다(raw_dir, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", str(raw_dir / "processed"))
    monkeypatch.setattr(config, "DATA_STATUS_PATH", str(raw_dir / "status.json"))
    monkeypatch.setattr(config, "DATA_END", "2026-10-20")
    monkeypatch.setattr(rd.collect_dart, "main", lambda: None)
    monkeypatch.setattr(rd, "fetch_series", lambda kind, name, s, e: frame(["2026-10-21"], [102.0]))
    monkeypatch.setattr(rd, "rebuild", lambda: (_ for _ in ()).throw(RuntimeError("이미 있던 날짜의 신호·확률이 바뀌었다")))
    with pytest.raises(RuntimeError):
        rd.run()
    assert config.DATA_END == "2026-10-20"
    assert len(pd.read_parquet(raw_dir / "raw" / "prices" / "samsung_ohlcv.parquet")) == 2
    assert not (raw_dir / "status.json").exists()


def test_신호와_확률_변화를_모두_센다():
    days = pd.bdate_range("2026-10-01", periods=2)
    before = pd.DataFrame({"signal": ["상승", "보합"], "p_상승": [0.5, 0.4]}, index=days)
    after = pd.DataFrame({"signal": ["상승", "보합"], "p_상승": [0.5, 0.41]}, index=days)
    assert rd.signal_changes(before, after) == 1


def test_덧붙이기만_하면_이미_있던_날의_워크포워드_신호는_그대로다(monkeypatch):
    import numpy as np
    from src.pipeline import train
    from src.pipeline.labels import make_labels
    monkeypatch.setattr(config, "LGBM_PARAMS", {**config.LGBM_PARAMS, "n_estimators": 20, "min_child_samples": 5})
    monkeypatch.setattr(config, "MIN_TRAIN_ROWS", 30)
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2025-01-01", periods=260)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days)))), index=days)
    feats = pd.DataFrame({"f1": rng.normal(size=len(days)), "f2": rng.normal(size=len(days))}, index=days)
    cut = days[-20]
    short = train.walk_forward(feats.loc[:cut], make_labels(close.loc[:cut]), valid_start="2025-06-02")
    full = train.walk_forward(feats, make_labels(close), valid_start="2025-06-02")
    assert rd.signal_changes(short, full) == 0
