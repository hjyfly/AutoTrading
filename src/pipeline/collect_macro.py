import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

from src import config

load_dotenv()


def fetch_fred(series_id: str, start: str, end: str) -> pd.DataFrame:
    """FRED 시계열을 받는다. 날짜는 미국 현지 날짜 그대로 둔다.

    지금은 config.FRED_SERIES 가 비어 있어 쓰지 않는다(공개 지연, SERVICE.md 섹션 6). 이미 받은 FRED 원본을 다시 만들 때만 쓴다.
    """
    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    series = fred.get_series(series_id, start, end).dropna()
    df = series.to_frame("value")
    df.index.name = "date"
    return _tag(df, f"FRED:{series_id}")


def fetch_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    """yfinance 일봉 종가와 거래량을 받는다. 날짜는 해당 거래소 현지 날짜 그대로 둔다."""
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance 의 end 는 포함하지 않는다
    raw = yf.download(ticker, start=start, end=end_exclusive, progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close", "Volume"]].rename(columns={"Close": "value", "Volume": "volume"}).dropna(subset=["value"])
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    return _tag(df, f"yfinance:{ticker}")


def _tag(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """출처와 수집 시각 열을 붙인다."""
    df = df.copy()
    df["source"] = source
    df["collected_at"] = datetime.now(timezone.utc).isoformat()
    df["is_dummy"] = False
    return df


def save_raw(df: pd.DataFrame, name: str) -> Path:
    """data/raw/macro/<name>.parquet 에 저장한다. 이미 있으면 덮어쓰지 않고 예외를 던진다."""
    path = Path(config.RAW_DIR) / "macro" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"원본은 수정하지 않는다: {path}")
    df.to_parquet(path)
    return path


def main():
    """config 의 FRED·yfinance 지표를 받아 저장한다. 이미 받은 원본은 건너뛴다."""
    start, end = config.DATA_START, config.DATA_END
    jobs = [(name, fetch_fred, sid) for name, sid in config.FRED_SERIES.items()]
    jobs += [(name, fetch_yf, ticker) for name, ticker in {**config.YF_TICKERS, **config.YF_TICKERS_EXT}.items()]
    for name, fetch, source_id in jobs:
        if (Path(config.RAW_DIR) / "macro" / f"{name}.parquet").exists():
            print(name, "이미 있음")
            continue
        df = fetch(source_id, start, end)
        print(name, save_raw(df, name), len(df), df.index.min().date(), df.index.max().date())


def main_ext():
    """4일차 과거 확장: EXT_DATA_START ~ EXT_DATA_END 의 yfinance 지표를 '<이름>_ext' 로 따로 저장한다(기존 원본 불변)."""
    start, end = config.EXT_DATA_START, config.EXT_DATA_END
    for name, ticker in {**config.YF_TICKERS, **config.YF_TICKERS_EXT}.items():
        ext = name + config.EXT_SUFFIX
        if (Path(config.RAW_DIR) / "macro" / f"{ext}.parquet").exists():
            print(ext, "이미 있음")
            continue
        df = fetch_yf(ticker, start, end)
        if df.empty:
            print(ext, "자료 없음(이 기간에 상장 전)")
            continue
        print(ext, save_raw(df, ext), len(df), df.index.min().date(), df.index.max().date())


if __name__ == "__main__":
    main_ext() if "--ext" in sys.argv[1:] else main()
