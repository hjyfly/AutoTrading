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
    cols = feature_set_columns(feature_set)
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise ValueError(f"지표 묶음 '{feature_set}' 의 열이 특징 표에 없다: {missing}")
    return list(cols)


def feature_set_columns(feature_set: str) -> list[str]:
    """지표 묶음 이름을 열 목록으로 바꾼다. config.FEATURE_SETS 의 이름이거나 'groups:price+chart' 처럼 지표 그룹을 + 로 이은 이름이다.
    그룹 묶음의 열 순서는 그룹을 고른 순서와 상관없이 후보 묶음(기존 45개 → A → B → D)의 순서를 따른다."""
    if feature_set in config.FEATURE_SETS:
        return list(config.FEATURE_SETS[feature_set])
    if not str(feature_set).startswith(config.FEATURE_GROUP_PREFIX):
        raise ValueError(f"알 수 없는 지표 묶음: {feature_set}")
    groups = [g for g in feature_set[len(config.FEATURE_GROUP_PREFIX):].split("+") if g]
    unknown = [g for g in groups if g not in config.FEATURE_GROUPS]
    if not groups or unknown:
        raise ValueError(f"알 수 없는 지표 그룹: {unknown or '(없음)'}")
    chosen = {c for g in groups for c in config.FEATURE_GROUPS[g]}
    return [c for c in config.FEATURE_SETS["candidate"] if c in chosen]


def group_set_name(groups) -> str:
    """지표 그룹 목록을 묶음 이름으로 바꾼다. 같은 그룹이면 고른 순서와 상관없이 같은 이름이 된다."""
    picked = set(groups)
    return config.FEATURE_GROUP_PREFIX + "+".join(g for g in config.FEATURE_GROUPS if g in picked)


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
                 freq: str | None = None, window_years: int | None = None) -> pd.DataFrame:
    """재학습 주기(기본 매월)의 첫 예측일에 다시 학습하고 그 기간 거래일마다 신호와 클래스 확률을 낸다.

    학습 행은 정답 기간이 끝나는 날(future_date)이 그 달 첫 예측일보다 앞선 것만 쓴다(정답이 검증 구간과 겹치지 않게).
    feature_set 은 config.FEATURE_SETS 의 이름('base' = 1~3일차 모델, 'candidate' = A+B+D 추가 후보) 또는 그룹 묶음 이름('groups:...').
    window_years 를 주면 재학습 시점에서 그 햇수 안의 행만 학습한다(기본 config.TRAIN_WINDOW_YEARS, None = 누적 전체).
    """
    window_years = config.TRAIN_WINDOW_YEARS if window_years is None else window_years
    cols = feature_columns(features, feature_set)
    X_all = features[cols]
    assert_no_lookahead(X_all)
    usable = X_all.dropna()
    train_pool = usable.join(labels[["label", "future_date"]], how="inner")
    rows = []
    for start, days in prediction_chunks(usable.index, valid_start or config.VALID_START, freq):
        train = train_pool[train_pool["future_date"] < start]
        if window_years:
            train = train[train.index >= start - pd.DateOffset(years=window_years)]
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
    """숫자 모델 신호를 만든다. 운영(base)·후보(candidate)와, 1~3일차 공식 판정을 다시 만드는 45개 모델(base_v1)."""
    features = pd.read_parquet(Path(config.PROCESSED_DIR) / "features.parquet")
    labels = pd.read_parquet(Path(config.PROCESSED_DIR) / "labels.parquet")
    official_labels = pd.read_parquet(config.LABELS_OFFICIAL_PATH)
    if config.OPERATIONAL_EXTENDED_HISTORY:  # 운영·후보는 확장 원본까지 이어 붙인 표로 학습한다(공식 45개는 기존 표 그대로)
        op_features, op_labels = pd.read_parquet(config.FEATURES_EXT_PATH), pd.read_parquet(config.LABELS_EXT_PATH)
    else:
        op_features, op_labels = features, labels
    for feature_set, path in [("base", config.SIGNALS_NUMERIC_PATH), ("candidate", config.SIGNALS_CANDIDATE_PATH),
                              ("base_v1", config.SIGNALS_BASE_V1_PATH)]:
        if feature_set == "base_v1":  # 1~3일차 공식 판정 조건(5거래일 정답, 매월 재학습, 기존 표) 그대로
            signals = walk_forward(features, official_labels, feature_set=feature_set, freq=config.OFFICIAL_RETRAIN_FREQ)
        else:
            signals = walk_forward(op_features, op_labels, feature_set=feature_set)
        signals.to_parquet(path)
        label = feature_set + (" (①안 탐색: 운영과 같은 재학습 주기로 비교)" if feature_set == "candidate" else "")
        print(label, path, len(signals), signals.index.min().date(), "~", signals.index.max().date(),
              f"재학습 {signals['retrained_on'].nunique()}회, 첫 학습 {signals['train_rows'].iloc[0]}행, 특징 {len(config.FEATURE_SETS[feature_set])}개")
        print("  신호 분포", signals["signal"].value_counts(normalize=True).round(3).to_dict())


def _logit_model():
    """합의용 로지스틱 회귀(표준화 + 규제 C, 설정은 조정하지 않는다)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=config.LOGIT_C, max_iter=config.LOGIT_MAX_ITER))


def _risk_lgbm():
    """하락 위험 2분류 LightGBM(운영과 같은 설정)."""
    return LGBMClassifier(**config.LGBM_PARAMS)


def walk_forward_risk(features: pd.DataFrame, risk_labels: pd.DataFrame, valid_start, feature_set: str = "base",
                      freq: str | None = None) -> pd.DataFrame:
    """5일차 하락 위험 워크포워드. 재학습 시점마다 정답 기간이 그 전에 끝난 행만 학습한다(walk_forward 와 같은 간격 규칙).

    p_risk_raw: 학습 행 전체로 맞춘 LightGBM 위험 확률
    p_risk_cal: 학습 행의 앞쪽으로 LightGBM 을 맞추고, 정답이 그 뒤에 시작하는 마지막 CALIBRATION_SHARE 로 isotonic 보정
                (보정용 행이 CALIBRATION_MIN_ROWS 보다 적으면 p_risk_raw 그대로)
    p_logit: 같은 지표·같은 학습 행의 로지스틱 회귀 위험 확률
    """
    from sklearn.isotonic import IsotonicRegression
    cols = feature_columns(features, feature_set)
    X_all = features[cols]
    assert_no_lookahead(X_all)
    usable = X_all.dropna()
    pool = usable.join(risk_labels[["risk", "future_date"]], how="inner")
    rows = []
    for start, days in prediction_chunks(usable.index, valid_start, freq):
        train = pool[pool["future_date"] < start]
        if len(train) < config.MIN_TRAIN_ROWS:
            raise ValueError(f"{start.date()} 재학습: 학습 행 {len(train)}개 < 최소 {config.MIN_TRAIN_ROWS}개")
        X = usable.loc[days, cols]
        full = _risk_lgbm().fit(train[cols], train["risk"])
        p_raw = full.predict_proba(X)[:, 1]
        cal_start = train.index[int(len(train) * (1 - config.CALIBRATION_SHARE))]
        fit_part = train[train["future_date"] < cal_start]  # 보정용 행의 정답과 겹치지 않게
        cal_part = train[train.index >= cal_start]
        if len(cal_part) >= config.CALIBRATION_MIN_ROWS and len(fit_part) >= config.MIN_TRAIN_ROWS and cal_part["risk"].nunique() == 2:
            part = _risk_lgbm().fit(fit_part[cols], fit_part["risk"])
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(part.predict_proba(cal_part[cols])[:, 1], cal_part["risk"])
            p_cal, calibrated = iso.predict(part.predict_proba(X)[:, 1]), True
        else:
            p_cal, calibrated = p_raw, False
        p_logit = _logit_model().fit(train[cols], train["risk"]).predict_proba(X)[:, 1]
        rows.append(pd.DataFrame({"p_risk_raw": p_raw, "p_risk_cal": p_cal, "p_logit": p_logit, "calibrated": calibrated,
                                  "retrained_on": start, "train_rows": len(train),
                                  "train_last_future_date": train["future_date"].max()}, index=days))
    out = pd.concat(rows)
    out.index.name = "date"
    if (out["train_last_future_date"] >= out["retrained_on"]).any():
        raise AssertionError("학습 정답 기간이 예측 시작일과 겹친다")
    return out


def main_risk():
    """5일차: 확장 특징 표로 하락 위험 워크포워드를 돌려 SIGNALS_RISK_PATH 에 저장한다(운영 신호는 바꾸지 않는다)."""
    from src.pipeline.labels import make_risk_labels
    features = pd.read_parquet(config.FEATURES_EXT_PATH)
    raw = Path(config.RAW_DIR) / "prices"
    close = pd.concat([pd.read_parquet(raw / f"samsung_ohlcv{config.EXT_SUFFIX}.parquet")["Close"],
                       pd.read_parquet(raw / "samsung_ohlcv.parquet")["Close"]]).sort_index()
    out = walk_forward_risk(features, make_risk_labels(close), ext_valid_start(features))
    out["trend_vote"] = (features.loc[out.index, config.TREND_MA] < 0).astype(int)  # t-1 종가까지로 만든 지표
    out.to_parquet(config.SIGNALS_RISK_PATH)
    print(config.SIGNALS_RISK_PATH, len(out), out.index.min().date(), "~", out.index.max().date(),
          f"재학습 {out['retrained_on'].nunique()}회, 보정한 덩어리 {int(out.groupby('retrained_on')['calibrated'].first().sum())}")


def ext_valid_start(features: pd.DataFrame, feature_set: str = "base") -> pd.Timestamp:
    """확장 검증 시작일(고정 규칙): 운영 지표가 모두 있는 첫날부터 EXT_FIRST_TRAIN_MONTHS 개월 뒤의 첫 거래일."""
    usable = features[feature_set_columns(feature_set)].dropna().index
    target = usable.min() + pd.DateOffset(months=config.EXT_FIRST_TRAIN_MONTHS)
    return usable[usable >= target].min()


def main_ext():
    """4일차 확장 검증: 확장 특징 표·정답으로 운영 조건(지표 묶음·재학습 주기·정답 기준 그대로) 워크포워드 신호를 만든다.
    운영 신호(signals_numeric)와 공식 판정 신호는 바꾸지 않는다."""
    features = pd.read_parquet(config.FEATURES_EXT_PATH)
    labels = pd.read_parquet(config.LABELS_EXT_PATH)
    start = ext_valid_start(features)
    signals = walk_forward(features, labels, valid_start=start, feature_set="base")
    signals.to_parquet(config.SIGNALS_EXT_PATH)
    print(config.SIGNALS_EXT_PATH, len(signals), signals.index.min().date(), "~", signals.index.max().date(),
          f"재학습 {signals['retrained_on'].nunique()}회, 첫 학습 {signals['train_rows'].iloc[0]}행")


if __name__ == "__main__":
    import sys
    if "--risk" in sys.argv[1:]:
        main_risk()
    else:
        main_ext() if "--ext" in sys.argv[1:] else main()
