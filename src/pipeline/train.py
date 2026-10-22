from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier

from src import config
from src.pipeline.features import assert_no_lookahead
from src.pipeline.labels import DOWN, FLAT, UP

CLASSES = [UP, FLAT, DOWN]


def feature_columns(features: pd.DataFrame, feature_set: str | None = None) -> list[str]:
    """학습에 쓰는 특징 열. feature_set 을 주면 config.FEATURE_SETS 의 열 목록(허용 목록)을, 없으면 표시 열을 뺀 전체를 쓴다.

    features.parquet 에는 원천 날짜 열이 없어, 특징의 시점 검사는 features.build_features 실행 때 이루어진다.
    여기서는 정답 관련 열 이름만 다시 막는다(assert_no_lookahead).
    """
    if feature_set is None:
        return [c for c in features.columns if c != "is_dummy"]
    cols = config.FEATURE_SETS[feature_set]
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise ValueError(f"지표 묶음 '{feature_set}' 의 열이 특징 표에 없다: {missing}")
    return list(cols)


def prediction_chunks(dates: pd.DatetimeIndex, valid_start, freq: str | None = None) -> list[tuple[pd.Timestamp, pd.DatetimeIndex]]:
    """검증 구간을 재학습 주기(freq, 기본 config.RETRAIN_FREQ: W 매주·M 매월·Q 분기) 단위로 나눈다.
    각 덩어리의 첫날이 재학습 시점이다. 첫 덩어리는 검증 시작일부터다."""
    freq = freq or config.RETRAIN_FREQ
    if freq not in config.RETRAIN_FREQS:
        raise ValueError(f"알 수 없는 재학습 주기: {freq}")
    valid = dates[dates >= pd.Timestamp(valid_start)]
    chunks = []
    for _, part in pd.Series(valid, index=valid).groupby(valid.to_period(freq)):
        chunks.append((part.index[0], part.index))
    return chunks


def walk_forward(features: pd.DataFrame, labels: pd.DataFrame, valid_start=None, feature_set: str | None = None,
                 freq: str | None = None) -> pd.DataFrame:
    """재학습 주기(기본 매월)의 첫 예측일에 다시 학습하고 그 기간 거래일마다 신호와 클래스 확률을 낸다.

    학습 행은 정답 기간이 끝나는 날(future_date)이 그 달 첫 예측일보다 앞선 것만 쓴다(정답이 검증 구간과 겹치지 않게).
    feature_set 은 config.FEATURE_SETS 의 이름('base' = 1~3일차 모델, 'candidate' = A+B+D 추가 후보).
    """
    cols = feature_columns(features, feature_set)
    X_all = features[cols]
    assert_no_lookahead(X_all)
    usable = X_all.dropna()
    train_pool = usable.join(labels[["label", "future_date"]], how="inner")
    rows = []
    for start, days in prediction_chunks(usable.index, valid_start or config.VALID_START, freq):
        train = train_pool[train_pool["future_date"] < start]
        if len(train) < config.MIN_TRAIN_ROWS:
            raise ValueError(f"{start.date()} 재학습: 학습 행 {len(train)}개 < 최소 {config.MIN_TRAIN_ROWS}개")
        if train["future_date"].max() >= start:
            raise AssertionError("학습 정답 기간이 예측 시작일과 겹친다")
        model = LGBMClassifier(**config.LGBM_PARAMS)
        model.fit(train[cols], train["label"].map(CLASSES.index))
        proba = pd.DataFrame(model.predict_proba(usable.loc[days, cols]), index=days,
                             columns=[CLASSES[i] for i in model.classes_]).reindex(columns=CLASSES, fill_value=0.0)
        chunk = pd.DataFrame({
            "signal": proba.idxmax(axis=1),
            "confidence": proba.max(axis=1),
            **{f"p_{c}": proba[c] for c in CLASSES},
            "retrained_on": start,
            "train_rows": len(train),
            "train_last_future_date": train["future_date"].max(),
        }, index=days)
        rows.append(chunk)
    out = pd.concat(rows)
    out.index.name = "date"
    if (out["train_last_future_date"] >= out["retrained_on"]).any():
        raise AssertionError("학습 정답 기간이 예측 시작일과 겹친다")
    return out


def main():
    """숫자 모델 신호를 만들어 data/processed/signals_numeric.parquet 에 저장한다."""
    features = pd.read_parquet(Path(config.PROCESSED_DIR) / "features.parquet")
    labels = pd.read_parquet(Path(config.PROCESSED_DIR) / "labels.parquet")
    for feature_set, path in [("base", config.SIGNALS_NUMERIC_PATH), ("candidate", config.SIGNALS_CANDIDATE_PATH)]:
        signals = walk_forward(features, labels, feature_set=feature_set)
        signals.to_parquet(path)
        label = feature_set + (" (①안 탐색: 검증 구간을 본 뒤 한 번만)" if feature_set == "candidate" else "")
        print(label, path, len(signals), signals.index.min().date(), "~", signals.index.max().date(),
              f"재학습 {signals['retrained_on'].nunique()}회, 첫 학습 {signals['train_rows'].iloc[0]}행, 특징 {len(config.FEATURE_SETS[feature_set])}개")
        print("  신호 분포", signals["signal"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
