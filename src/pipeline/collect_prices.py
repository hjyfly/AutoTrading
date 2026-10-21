import time
from datetime import datetime, timezone
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import requests

from src import config

NAVER_TREND_URL = "https://m.stock.naver.com/api/stock/{code}/trend"
NAVER_HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_ohlcv(code: str, start: str, end: str) -> pd.DataFrame:
    """종목 일봉(시가·고가·저가·종가·거래량)을 FinanceDataReader 로 받는다. 실패하면 pykrx 로 다시 받는다."""
    try:
        df = fdr.DataReader(code, start, end)[["Open", "High", "Low", "Close", "Volume"]]
        source = "FinanceDataReader"
    except Exception:
        from pykrx import stock

        df = stock.get_market_ohlcv(start.replace("-", ""), end.replace("-", ""), code)
        df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"})
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        source = "pykrx"
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    return _tag(df, source)


def _to_int(text: str) -> int:
    """네이버 응답의 '+1,801,659' 같은 문자열을 정수로 바꾼다."""
    return int(text.replace(",", "").replace("+", ""))


def fetch_foreign_net(code: str, start: str, end: str) -> pd.DataFrame:
    """외국인·기관·개인 순매수 수량(주)을 네이버 증권 모바일 API 에서 날짜를 거슬러 가며 받는다."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    cursor = (end_ts + pd.Timedelta(days=1)).strftime("%Y%m%d")  # bizdate 는 그 날짜 이전 자료를 돌려준다
    rows = []
    while True:
        resp = requests.get(NAVER_TREND_URL.format(code=code), params={"pageSize": 60, "bizdate": cursor},
                            headers=NAVER_HEADERS, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        oldest = min(item["bizdate"] for item in batch)
        if pd.Timestamp(oldest) <= start_ts or oldest == cursor:
            break
        cursor = oldest
        time.sleep(config.REQUEST_PAUSE_SEC)
    df = pd.DataFrame({
        "date": pd.to_datetime([r["bizdate"] for r in rows]),
        "foreign_net": [_to_int(r["foreignerPureBuyQuant"]) for r in rows],
        "institution_net": [_to_int(r["organPureBuyQuant"]) for r in rows],
        "individual_net": [_to_int(r["individualPureBuyQuant"]) for r in rows],
        "foreign_hold_ratio": [float(r["foreignerHoldRatio"].rstrip("%")) for r in rows],
    }).drop_duplicates("date").set_index("date").sort_index()
    return _tag(df.loc[start_ts:end_ts], "naver_mobile_api")


def select_inverse_etf(samsung: pd.DataFrame, candidates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """후보 인버스 ETF 마다 삼성전자 일간 수익률과의 상관계수, 삼성전자 하락일에 ETF 가 오른 비율을 계산한다."""
    base = samsung["Close"].pct_change()
    result = []
    for code, etf in candidates.items():
        ret = etf["Close"].pct_change()
        joined = pd.concat([base, ret], axis=1, keys=["samsung", "etf"]).dropna()
        down_days = joined[joined["samsung"] < 0]
        result.append({
            "code": code,
            "name": config.INVERSE_ETF_CANDIDATES[code],
            "corr": joined["samsung"].corr(joined["etf"]),
            "up_when_samsung_down": (down_days["etf"] > 0).mean(),
            "days": len(joined),
        })
    return pd.DataFrame(result).sort_values("corr").reset_index(drop=True)  # 상관이 가장 음수인 후보가 맨 위


def _tag(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """출처와 수집 시각 열을 붙인다."""
    df = df.copy()
    df["source"] = source
    df["collected_at"] = datetime.now(timezone.utc).isoformat()
    df["is_dummy"] = False
    return df


def save_raw(df: pd.DataFrame, name: str) -> Path:
    """data/raw/prices/<name>.parquet 에 저장한다. 이미 있으면 덮어쓰지 않고 예외를 던진다."""
    path = Path(config.RAW_DIR) / "prices" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"원본은 수정하지 않는다: {path}")
    df.to_parquet(path)
    return path


def fetch_close(code: str, start: str, end: str) -> pd.DataFrame:
    """지수·종목의 일별 종가만 받는다(확장 지표용 코스피·코스피200·SK하이닉스)."""
    df = fdr.DataReader(code, start, end)[["Close"]].astype(float)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    return _tag(df, f"FinanceDataReader:{code}")


def _exists(name: str) -> bool:
    """원본이 이미 있는지."""
    return (Path(config.RAW_DIR) / "prices" / f"{name}.parquet").exists()


def main():
    """삼성전자 일봉, 외국인 순매수, 인버스 ETF 후보, 확장 지표용 국내 시계열을 받아 저장한다. 이미 받은 원본은 건너뛴다."""
    start, end = config.DATA_START, config.DATA_END
    if not _exists("samsung_ohlcv"):
        samsung = fetch_ohlcv(config.SAMSUNG_CODE, start, end)
        print("samsung_ohlcv", save_raw(samsung, "samsung_ohlcv"), len(samsung))
        flows = fetch_foreign_net(config.SAMSUNG_CODE, start, end)
        print("samsung_investor_flows", save_raw(flows, "samsung_investor_flows"), len(flows))
        candidates = {}
        for code in config.INVERSE_ETF_CANDIDATES:
            candidates[code] = fetch_ohlcv(code, start, end)
            print(f"inverse_{code}", save_raw(candidates[code], f"inverse_{code}"), len(candidates[code]))
        print(select_inverse_etf(samsung, candidates).to_string())
    for name, code in config.KR_EXTRA.items():
        if _exists(name):
            print(name, "이미 있음")
            continue
        df = fetch_close(code, start, end)
        print(name, save_raw(df, name), len(df), df.index.min().date(), df.index.max().date())


if __name__ == "__main__":
    main()
