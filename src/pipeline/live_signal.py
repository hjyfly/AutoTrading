"""오늘(또는 지정한 날) 신호를 실시간 데이터로 만든다. 과거 검증(train.walk_forward)과 같은 규칙을 쓴다.

- 숫자 모델: 그 달 첫 거래일을 재학습 시점으로 보고, 정답 기간이 그날 전에 끝난 행으로 LightGBM 을 학습해 t 행을 예측한다.
  그래서 저장된 날짜에 대해 만들면 signals_numeric.parquet 과 같은 신호가 나온다.
- 뉴스 보정: t 15:15 이후에만 만든다. 그날 구간 [t-1 15:15, t 15:15) 기사를 받아 선별·채점하고 고정 규칙으로 보정한다.
  LLM 을 7번 부르므로(약 $0.15) 사용자가 요청할 때만 돌린다.
- 결과는 data/processed/signals_live.parquet 에 날짜별로 남긴다(저장된 검증 신호 파일은 건드리지 않는다).
"""
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier

from src import config, tools
from src.pipeline import collect_dart, collect_news, features, news_signal, scoring, select_news
from src.pipeline.labels import make_labels
from src.pipeline.train import CLASSES, feature_columns

LIVE_PATH = Path(config.PROCESSED_DIR) / "signals_live.parquet"


def block_start(t: pd.Timestamp, days: pd.DatetimeIndex, valid_start=None, freq: str | None = None) -> pd.Timestamp:
    """t 의 재학습 시점. 과거 검증(train.prediction_chunks)과 같게, 재학습 주기(기본 config.RETRAIN_FREQ) 안의 첫 거래일이되
    검증 시작일보다 앞서지 않는다."""
    freq = freq or config.RETRAIN_FREQ
    valid_start = pd.Timestamp(valid_start or config.VALID_START)
    same_block = days[(days.to_period(freq) == t.to_period(freq)) & (days <= t) & (days >= min(valid_start, t))]
    return same_block[0]


def numeric_signal(as_of: str | None = None, valid_start=None, feature_set: str = "base") -> dict:
    """t 일 15:15 에 알 수 있던 데이터로 숫자 모델 신호와 클래스 확률을 만든다. feature_set 은 'base'·'candidate' 또는 그룹 묶음 이름."""
    t = tools.resolve_as_of(as_of)
    if t.weekday() >= 5:
        raise ValueError(f"{t.date()} 는 주말이라 신호를 만들지 않는다")
    raw, sources = tools.raw_until(t)
    table = features.build_features(as_of=t, raw=raw)
    if table.index[-1] != t:
        raise ValueError(f"{t.date()} 행을 만들지 못했다")
    cols = feature_columns(table, feature_set)
    features.assert_no_lookahead(table[cols])
    labels = make_labels(raw["ohlcv"]["Close"].dropna())  # t 이전 종가만으로 만든 정답(정답 기간이 t 전에 끝난 것만 생긴다)
    start = block_start(t, table.index[table[cols].notna().all(axis=1)], valid_start)  # 과거 검증처럼 빈 값 없는 날로 재학습일을 정한다
    train = table[cols].dropna().join(labels[["label", "future_date"]], how="inner")
    train = train[train["future_date"] < start]
    if len(train) < config.MIN_TRAIN_ROWS:
        raise ValueError(f"학습 행 {len(train)}개 < 최소 {config.MIN_TRAIN_ROWS}개")
    model = LGBMClassifier(**config.LGBM_PARAMS)
    model.fit(train[cols], train["label"].map(CLASSES.index))
    row = table.loc[[t], cols]
    if row.isna().any(axis=None):
        raise ValueError(f"{t.date()} 특징에 빈 값이 있다: {list(row.columns[row.isna().iloc[0]])}")
    proba = pd.Series(model.predict_proba(row)[0], index=[CLASSES[i] for i in model.classes_]).reindex(CLASSES, fill_value=0.0)
    return {"date": t, "model": feature_set, "signal": proba.idxmax(), "confidence": float(proba.max()),
            **{f"p_{c}": float(proba[c]) for c in CLASSES},
            "retrained_on": start, "train_rows": len(train), "train_last_future_date": train["future_date"].max(),
            "data_sources": sources}


def news_ready(t: pd.Timestamp, now: pd.Timestamp | None = None) -> bool:
    """t 일 뉴스 구간이 닫혔는가(한국 시각 t 15:15 이후인가)."""
    now = now if now is not None else pd.Timestamp.now(tz=config.KST)
    hour, minute = map(int, config.SIGNAL_CUTOFF_KST.split(":"))
    return now.tz_convert(config.KST).tz_localize(None) >= t + pd.Timedelta(hours=hour, minutes=minute)


def news_score(t: pd.Timestamp) -> dict:
    """t 일 뉴스를 실시간으로 받아 선별·채점한다. 원본 기사는 파일로 남기지 않고, 판단 기록만 남긴다."""
    days = collect_news.trading_days()
    if t not in days:
        days = days.append(pd.DatetimeIndex([t])).sort_values()
    start, end = collect_news.signal_window(t, days)
    gdelt = collect_news.collect_gdelt(start, end)
    rss = collect_news.collect_rss(t, days)
    news = pd.concat([gdelt, rss], ignore_index=True).assign(signal_date=t)
    news = collect_news.dedupe(collect_news.within_window(news, days))
    articles = select_news.select_articles(news, t, days)
    reports = collect_dart.dart_as_of(pd.read_parquet(Path(config.PROCESSED_DIR) / "dart_index.parquet"), t)
    report = reports.sort_values("rcept_dt").iloc[-1] if len(reports) else None
    return scoring.score_day(t, articles=articles, report=report)


def make_signal(as_of: str | None = None, with_news: bool = False) -> dict:
    """기존 모델(base) 신호를 만들고, 비교용 후보 모델(candidate) 신호도 나란히 남긴다(사전 등록한 ②안).

    with_news 면 15:15 이후에만 기존 모델 신호에 뉴스 보정을 더한다. 후보 모델은 숫자 신호만 기록한다.
    """
    candidate = numeric_signal(as_of, feature_set="candidate")
    candidate.update(numeric_signal_label=candidate["signal"], news_status="후보 모델은 뉴스 보정 안 함",
                     news_consensus=None, news_consensus_conf=None)
    save(candidate)
    for feature_set in config.LIVE_EXTRA_FEATURE_SETS:  # 지표 그룹 진단에서 고른 묶음(비교용, 뉴스 보정 안 함)
        try:  # 비교용 묶음이 데이터 부족 등으로 실패해도 운영 신호(base)는 만든다. 시점 위반(AssertionError)은 그대로 멈춘다
            extra = numeric_signal(as_of, feature_set=feature_set)
        except ValueError as e:
            print(f"비교용 묶음 {feature_set} 신호 실패: {type(e).__name__}: {e}", flush=True)
            continue
        extra.update(numeric_signal_label=extra["signal"], news_status="비교용 묶음은 뉴스 보정 안 함",
                     news_consensus=None, news_consensus_conf=None)
        save(extra)
    out = numeric_signal(as_of)
    t = out["date"]
    out.update(numeric_signal_label=out["signal"], news_status="요청 안 함", news_consensus=None, news_consensus_conf=None)
    previous = _previous_news(t)
    if not with_news and previous is not None:  # 같은 날 다시 실행해도 이미 만든 뉴스 보정은 지우지 않는다
        out.update(news_consensus=previous[0], news_consensus_conf=previous[1],
                   signal=news_signal.adjust(out["signal"], previous[0], previous[1]),
                   news_status="이전에 만든 뉴스 채점을 다시 씀")
    if with_news:
        if not news_ready(t):
            out["news_status"] = f"{t.date()} {config.SIGNAL_CUTOFF_KST} 이후에 뉴스 보정을 만들 수 있다"
        else:
            trace = news_score(t)
            final = trace["final"]
            out.update(news_consensus=final["score"], news_consensus_conf=final["confidence"],
                       signal=news_signal.adjust(out["signal"], final["score"], final["confidence"]),
                       news_status=f"채점 완료 (기사 {len(trace['articles'])}건, ${trace['cost_usd']:.3f})")
    save(out)
    return out


def _previous_news(t: pd.Timestamp) -> tuple[float, float] | None:
    """그날 기존 모델 기록에 남아 있는 뉴스 합의 점수와 확신도."""
    if not LIVE_PATH.exists():
        return None
    live = pd.read_parquet(LIVE_PATH)
    if "model" not in live.columns:
        live["model"] = "base"
    rows = live[(live.index == t) & (live["model"] == "base")]
    if rows.empty or pd.isna(rows["news_consensus"].iloc[-1]):
        return None
    return float(rows["news_consensus"].iloc[-1]), float(rows["news_consensus_conf"].iloc[-1])


def save(result: dict):
    """신호 결과 한 줄을 signals_live.parquet 에 (날짜, 모델) 기준으로 덮어 쓴다."""
    row = pd.DataFrame([{k: v for k, v in result.items() if k != "data_sources"}]).set_index("date")
    row["model"] = result.get("model", "base")
    row["created_at"] = pd.Timestamp.now(tz=config.KST).tz_localize(None)  # 실제로 그 시각에 만든 신호인지 가리기 위해
    if LIVE_PATH.exists():
        old = pd.read_parquet(LIVE_PATH)
        if "model" not in old.columns:
            old["model"] = "base"
        same = (old.index == row.index[0]) & (old["model"] == row["model"].iloc[0])
        row = pd.concat([old[~same], row]).sort_index()
    LIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    row.to_parquet(LIVE_PATH)


def main(args: list[str]):
    """명령줄: python -m src.pipeline.live_signal [YYYY-MM-DD] [--news]"""
    date = next((a for a in args if not a.startswith("--")), None)
    out = make_signal(date, with_news="--news" in args)
    print({k: v for k, v in out.items() if k != "data_sources"})


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
