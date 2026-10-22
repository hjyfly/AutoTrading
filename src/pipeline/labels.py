from pathlib import Path

import pandas as pd

from src import config

UP, FLAT, DOWN = "상승", "보합", "하락"


def make_labels(close: pd.Series, horizon: int | None = None, threshold: float | None = None) -> pd.DataFrame:
    """t일 종가 대비 t+N거래일 종가 수익률로 상승·보합·하락 정답을 만든다. 정답이 없는 마지막 N거래일은 뺀다.

    horizon(N)·threshold 를 주지 않으면 config.LABEL_HORIZON_DAYS·LABEL_THRESHOLD 를 쓴다(조건 비교 실험에서만 바꾼다).

    shift(-N) 은 정답을 만들 때만 쓴다. 이 결과는 학습 입력(특징)에 넣지 않는다.
    """
    horizon = config.LABEL_HORIZON_DAYS if horizon is None else horizon
    threshold = config.LABEL_THRESHOLD if threshold is None else threshold
    close = close.sort_index()
    future_close = close.shift(-horizon)
    ret = future_close / close - 1
    label = pd.Series(FLAT, index=close.index)
    label[ret >= threshold] = UP
    label[ret <= -threshold] = DOWN
    df = pd.DataFrame({
        "close": close,
        "future_date": pd.Series(close.index, index=close.index).shift(-horizon),
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
    official = make_labels(ohlcv["Close"], horizon=config.OFFICIAL_LABEL_HORIZON_DAYS)  # 1~3일차 공식 판정 재현용(5거래일)
    official["is_dummy"] = labels["is_dummy"].iloc[0] if len(labels) else False
    official.to_parquet(config.LABELS_OFFICIAL_PATH)
    periods = {
        "전체": (config.DATA_START, config.DATA_END),
        "학습(첫 18개월)": (config.DATA_START, config.TRAIN_END),
        "검증": (config.VALID_START, config.DATA_END),
        "기준 B 구간(2~4월)": (config.CRITERION_B_START, config.CRITERION_B_END),
        "뉴스 구간(확장)": (config.NEWS_START, config.NEWS_END),
    }
    print(out, len(labels))
    print(label_distribution(labels, periods).to_string(index=False))


def make_risk_labels(close: pd.Series, horizon: int | None = None, threshold: float | None = None) -> pd.DataFrame:
    """5일차 하락 위험 정답: t+1 ~ t+N 거래일 중 최저 종가가 t 종가 대비 -기준 이하로 한 번이라도 내려가면 1, 아니면 0.
    future_date 는 t+N 거래일(그날 종가까지 봐야 정답이 정해진다). 정답이 없는 마지막 N거래일은 뺀다."""
    horizon = config.RISK_HORIZON_DAYS if horizon is None else horizon
    threshold = config.RISK_THRESHOLD if threshold is None else threshold
    close = close.sort_index().astype(float)
    future_min = pd.concat([close.shift(-k) for k in range(1, horizon + 1)], axis=1).min(axis=1, skipna=False)
    out = pd.DataFrame({"close": close, "future_min": future_min,
                        "future_date": pd.Series(close.index, index=close.index).shift(-horizon)})
    out = out.dropna()
    out["risk"] = (out["future_min"] / out["close"] - 1 <= -threshold).astype(int)
    return out[["risk", "future_date"]]


def main_ext():
    """확장 원본까지 이어 붙인 삼성전자 종가로 정답을 만들어 LABELS_EXT_PATH 에 저장한다(운영 정답 기준 그대로)."""
    ext = pd.read_parquet(Path(config.RAW_DIR) / "prices" / f"samsung_ohlcv{config.EXT_SUFFIX}.parquet")["Close"]
    current = pd.read_parquet(Path(config.RAW_DIR) / "prices" / "samsung_ohlcv.parquet")["Close"]
    labels = make_labels(pd.concat([ext, current]).sort_index().astype(float))
    labels["is_dummy"] = False
    labels.to_parquet(config.LABELS_EXT_PATH)
    print(config.LABELS_EXT_PATH, len(labels), labels.index.min().date(), "~", labels.index.max().date())


if __name__ == "__main__":
    import sys
    main_ext() if "--ext" in sys.argv[1:] else main()
