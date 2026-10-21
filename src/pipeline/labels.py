from pathlib import Path

import pandas as pd

from src import config

UP, FLAT, DOWN = "상승", "보합", "하락"


def make_labels(close: pd.Series) -> pd.DataFrame:
    """t일 종가 대비 t+N거래일 종가 수익률로 상승·보합·하락 정답을 만든다. 정답이 없는 마지막 N거래일은 뺀다.

    shift(-N) 은 정답을 만들 때만 쓴다. 이 결과는 학습 입력(특징)에 넣지 않는다.
    """
    close = close.sort_index()
    future_close = close.shift(-config.LABEL_HORIZON_DAYS)
    ret = future_close / close - 1
    label = pd.Series(FLAT, index=close.index)
    label[ret >= config.LABEL_THRESHOLD] = UP
    label[ret <= -config.LABEL_THRESHOLD] = DOWN
    df = pd.DataFrame({
        "close": close,
        "future_date": pd.Series(close.index, index=close.index).shift(-config.LABEL_HORIZON_DAYS),
        "future_close": future_close,
        "future_return": ret,
        "label": label,
    })
    return df.dropna(subset=["future_close"])


def label_distribution(labels: pd.DataFrame, periods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """구간별 상승·보합·하락 비율(%)과 일수를 돌려준다. 이 비율이 성공 기준의 기준선이다."""
    rows = []
    for name, (start, end) in periods.items():
        part = labels.loc[start:end, "label"]
        share = part.value_counts(normalize=True).reindex([UP, FLAT, DOWN], fill_value=0) * 100
        rows.append({"구간": name, "일수": len(part), **share.round(1).to_dict()})
    return pd.DataFrame(rows)


def main():
    """삼성전자 원본 일봉으로 정답을 만들어 저장하고 구간별 분포를 출력한다."""
    ohlcv = pd.read_parquet(Path(config.RAW_DIR) / "prices" / "samsung_ohlcv.parquet")
    labels = make_labels(ohlcv["Close"])
    labels["is_dummy"] = bool(ohlcv["is_dummy"].any())
    out = Path(config.PROCESSED_DIR) / "labels.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out)
    periods = {
        "전체": (config.DATA_START, config.DATA_END),
        "학습(첫 18개월)": (config.DATA_START, config.TRAIN_END),
        "검증": (config.VALID_START, config.DATA_END),
        "뉴스 구간(2~4월)": (config.NEWS_START, config.NEWS_END),
    }
    print(out, len(labels))
    print(label_distribution(labels, periods).to_string(index=False))


if __name__ == "__main__":
    main()
