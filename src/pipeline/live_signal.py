"""오늘(또는 지정한 날) 신호를 실시간 데이터로 만든다. 과거 검증(train.walk_forward)과 같은 규칙을 쓴다.

- 숫자 모델: 재학습 주기(지금 매주)의 첫 거래일을 재학습 시점으로 보고, 정답 기간이 그날 전에 끝난 행으로 LightGBM 을 학습해 t 행을 예측한다.
  그래서 저장된 날짜에 대해 만들면 signals_numeric.parquet 과 같은 신호가 나온다.
- 뉴스 채점: t 15:15 이후에만 만든다. 그날 구간 [t-1 15:15, t 15:15) 기사를 받아 선별·채점한다. 운영 신호 보정은
  config.NEWS_ADJUST_IN_SIGNAL 일 때만 한다(2026-10-22 부터 꺼짐, 뉴스는 참고용).
  채점 7번 + 기사별 추출 1번(형식 오류면 최대 3번), 하루 약 $0.27 이라 15:15 이후 요청할 때만 돌린다(평일 15:20 자동 실행 포함).
- 결과는 data/processed/signals_live.parquet 에 날짜별로 남긴다(저장된 검증 신호 파일은 건드리지 않는다).
- 11일차 기간별 신호: 운영 기간(5거래일) 말고 config.HORIZONS 의 다른 기간도 같은 지표·재학습 규칙으로 학습해 model=h<N> 으로 기록한다
  (참고 신호, 뉴스 보정 없음, 매매 기준은 운영 기간 그대로). 과거 검증은 experiment.horizon_study 의 기간별 워크포워드와 같다.
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


def prepare(as_of: str | None = None) -> tuple:
    """t 일 15:15 에 알 수 있던 원본과 특징 표: (t, 원본, 출처, 특징 표). 여러 기간 신호가 같은 표를 다시 쓰도록 따로 둔다."""
    t = tools.resolve_as_of(as_of)
    if t.weekday() >= 5:
        raise ValueError(f"{t.date()} 는 주말이라 신호를 만들지 않는다")
    raw, sources = tools.raw_until(t)
    if config.OPERATIONAL_EXTENDED_HISTORY:  # 과거 검증(train.main)과 같은 학습 기간: 확장 원본까지 이어 붙인다
        raw = prepend_extended(raw)
    table = features.build_features(as_of=t, raw=raw)
    if table.index[-1] != t:
        raise ValueError(f"{t.date()} 행을 만들지 못했다")
    return t, raw, sources, table


def numeric_signal(as_of: str | None = None, valid_start=None, feature_set: str = "base", horizon: int | None = None,
                   prepared: tuple | None = None) -> dict:
    """t 일 15:15 에 알 수 있던 데이터로 숫자 모델 신호와 클래스 확률을 만든다. feature_set 은 'base'·'candidate' 또는 그룹 묶음 이름.
    horizon 은 정답 기간(거래일, 기본 운영 기간). 다른 기간이면 그 기간 정답(config.HORIZONS 기준)으로 학습하고 모델 이름은 h<N> 이다.
    prepared 는 같은 날 prepare() 결과(주면 원본·특징 표를 다시 만들지 않는다)."""
    n = horizon or config.OPERATIONAL_HORIZON
    model_name = feature_set if n == config.OPERATIONAL_HORIZON else tools.horizon_model(n)
    t, raw, sources, table = prepared or prepare(as_of)
    cols = feature_columns(table, feature_set)
    features.assert_no_lookahead(table[cols])
    labels = make_labels(raw["ohlcv"]["Close"].dropna(), n, config.HORIZONS[n])  # t 이전 종가만으로 만든 정답(정답 기간이 t 전에 끝난 것만 생긴다)
    start = block_start(t, table.index[table[cols].notna().all(axis=1)], valid_start)  # 과거 검증처럼 빈 값 없는 날로 재학습일을 정한다
    train = table[cols].dropna().join(labels[["label", "future_date"]], how="inner")
    train = train[train["future_date"] < start]
    if config.TRAIN_WINDOW_YEARS:  # 과거 검증(walk_forward)과 같은 학습 창
        train = train[train.index >= start - pd.DateOffset(years=config.TRAIN_WINDOW_YEARS)]
    if len(train) < config.MIN_TRAIN_ROWS:
        raise ValueError(f"학습 행 {len(train)}개 < 최소 {config.MIN_TRAIN_ROWS}개")
    model = LGBMClassifier(**config.LGBM_PARAMS)
    model.fit(train[cols], train["label"].map(CLASSES.index))
    row = table.loc[[t], cols]
    if row.isna().any(axis=None):
        raise ValueError(f"{t.date()} 특징에 빈 값이 있다: {list(row.columns[row.isna().iloc[0]])}")
    proba = pd.Series(model.predict_proba(row)[0], index=[CLASSES[i] for i in model.classes_]).reindex(CLASSES, fill_value=0.0)
    return {"date": t, "model": model_name, "horizon": n, "signal": proba.idxmax(), "confidence": float(proba.max()),
            **{f"p_{c}": float(proba[c]) for c in CLASSES},
            "retrained_on": start, "train_rows": len(train), "train_last_future_date": train["future_date"].max(),
            "data_sources": sources}


RAW_KEYS = {"ohlcv": ("prices", "samsung_ohlcv"), "flows": ("prices", "samsung_investor_flows"),
            **{n: ("macro", n) for n in {**config.YF_TICKERS, **config.YF_TICKERS_EXT}},
            **{n: ("prices", n) for n in config.KR_EXTRA}}


def prepend_extended(raw: dict) -> dict:
    """실시간 원본 앞에 4일차 확장 원본('<이름>_ext')을 붙인다. 확장 원본이 없거나 날짜가 겹치면 붙이지 않는다."""
    out = dict(raw)
    for key, (kind, name) in RAW_KEYS.items():
        if key not in raw:
            continue
        ext = tools._load_raw(kind, name + config.EXT_SUFFIX)
        if ext.empty:
            continue
        if ext.index.max() >= raw[key].index.min():  # 과거 검증(features.load_raw_extended)과 같게 겹치면 멈춘다
            raise ValueError(f"{name}: 확장 원본과 실시간 원본의 날짜가 겹친다")
        out[key] = pd.concat([ext[raw[key].columns.intersection(ext.columns)], raw[key]])
    return out


def risk_signal(as_of: str | None = None) -> dict:
    """5일차 하락 위험 조건(config.LIVE_RISK_CONDITION, 지금 P2)의 오늘 목표 비중. 확장 원본까지 이어 붙여 판정 때와 같은
    규칙(매주 재학습, 보정)으로 학습하고, 기준은 판정 기록(day5_judged.json)에 남은 개발 구간 값을 그대로 쓴다. 비교용이다."""
    import json
    from src.pipeline import backtest
    from src.pipeline.labels import make_risk_labels
    from src.pipeline.train import walk_forward_risk
    judged = json.loads(Path(config.JUDGED_PATH).read_text(encoding="utf-8"))[config.LIVE_RISK_CONDITION]
    t = tools.resolve_as_of(as_of)
    if t.weekday() >= 5:
        raise ValueError(f"{t.date()} 는 주말이라 신호를 만들지 않는다")
    raw, sources = tools.raw_until(t)
    raw = prepend_extended(raw)
    table = features.build_features(as_of=t, raw=raw)
    if table.index[-1] != t:
        raise ValueError(f"{t.date()} 행을 만들지 못했다")
    cols = feature_columns(table, "base")
    usable = table.index[table[cols].notna().all(axis=1)]
    start = block_start(t, usable, valid_start=config.DEV_START)  # 확장 검증 시작일 기준(운영 VALID_START 로 자르지 않는다)
    risk = walk_forward_risk(table, make_risk_labels(raw["ohlcv"]["Close"].dropna()), valid_start=start)
    if t not in risk.index:
        raise ValueError(f"{t.date()} 위험 확률을 만들지 못했다")
    r = risk.loc[t]
    col = "p_risk_cal" if config.LIVE_RISK_CONDITION == "P2" else "p_risk_raw"
    weight = float(backtest.risk_weights(pd.Series([r[col]]), *judged["thresholds"][col]).iloc[0])
    return {"date": t, "model": config.LIVE_RISK_MODEL, "signal": dict(zip(config.RISK_WEIGHTS, ("상승", "보합", "하락")))[weight],
            "weight": weight, "p_risk": float(r[col]), "confidence": float(r[col]), "retrained_on": start,
            "train_rows": int(r["train_rows"]), "train_last_future_date": r["train_last_future_date"],
            "numeric_signal_label": None, "news_status": "하락 위험 비교 모델은 뉴스 보정 안 함",
            "news_consensus": None, "news_consensus_conf": None, "data_sources": sources}


def news_ready(t: pd.Timestamp, now: pd.Timestamp | None = None) -> bool:
    """t 일 뉴스 구간이 닫혔는가(한국 시각 t 15:15 이후인가)."""
    now = now if now is not None else pd.Timestamp.now(tz=config.KST)
    hour, minute = map(int, config.SIGNAL_CUTOFF_KST.split(":"))
    return now.tz_convert(config.KST).tz_localize(None) >= t + pd.Timedelta(hours=hour, minutes=minute)


def news_score(t: pd.Timestamp) -> dict:
    """t 일 뉴스를 실시간으로 받아 선별·채점한다. 받은 기사는 수집 원본·수집본에 그날 행으로 남기고(collect_news.save_day,
    2026-10-23 사용자 결정: 화면·도구가 다시 받지 않게), 판단 기록·기사별 추출을 남긴다."""
    days = collect_news.trading_days()
    if t not in days:
        days = days.append(pd.DatetimeIndex([t])).sort_values()
    start, end = collect_news.signal_window(t, days)
    gdelt = collect_news.collect_gdelt(start, end)
    rss = collect_news.collect_rss(t, days)
    try:  # 저장이 실패해도 채점은 한다
        collect_news.save_day(t, gdelt, rss, days)
    except Exception as e:
        print(f"뉴스 저장 실패: {type(e).__name__}: {e}", flush=True)
    news = pd.concat([gdelt, rss], ignore_index=True).assign(signal_date=t)
    news = collect_news.dedupe(collect_news.within_window(news, days))
    articles = select_news.select_articles(news, t, days)
    reports = collect_dart.dart_as_of(pd.read_parquet(Path(config.PROCESSED_DIR) / "dart_index.parquet"), t)
    report = reports.sort_values("rcept_dt").iloc[-1] if len(reports) else None
    trace = scoring.score_day(t, articles=articles, report=report)
    try:  # 9일차 기사별 추출(참고·경보용). 실패해도 채점 결과는 그대로 쓴다
        scoring.extract_day(t, articles=articles, raw_news=news)
    except Exception as e:
        print(f"기사별 추출 실패: {type(e).__name__}: {e}", flush=True)
    return trace


def make_signal(as_of: str | None = None, with_news: bool = False) -> dict:
    """기존 모델(base) 신호를 만들고, 비교용 후보 모델(candidate) 신호도 나란히 남긴다(사전 등록한 ②안).

    with_news 면 15:15 이후에만 뉴스를 채점해 함께 남긴다(신호 보정은 config.NEWS_ADJUST_IN_SIGNAL 일 때만). 비교 모델은 숫자 신호만 기록한다.
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
    prepared = None
    for n in config.HORIZONS:  # 11일차 기간별 참고 신호(어떤 오류로 실패해도 운영 신호는 만든다). 시점 위반(AssertionError)은 그대로 멈춘다
        if n == config.OPERATIONAL_HORIZON:
            continue
        try:
            prepared = prepared or prepare(as_of)  # 원본·특징 표는 한 번만 만들고 기간마다 정답·학습만 다르게 한다
            other = numeric_signal(as_of, horizon=n, prepared=prepared)
        except AssertionError:
            raise
        except Exception as e:  # 참고 신호라 네트워크·학습 오류도 기록하고 넘어간다
            print(f"{n}거래일 신호 실패: {type(e).__name__}: {e}", flush=True)
            continue
        other.update(numeric_signal_label=other["signal"], news_status="기간별 신호는 뉴스 보정 안 함",
                     news_consensus=None, news_consensus_conf=None)
        save(other)
    try:  # 5일차 하락 위험 비교 조건(실패해도 운영 신호는 만든다). 시점 위반(AssertionError)은 그대로 멈춘다
        save(risk_signal(as_of))
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"하락 위험 비교 신호 실패: {type(e).__name__}: {e}", flush=True)
    out = numeric_signal(as_of)
    t = out["date"]
    out.update(numeric_signal_label=out["signal"], news_status="요청 안 함", news_consensus=None, news_consensus_conf=None)
    previous = _previous_news(t)
    if not with_news and previous is not None:  # 같은 날 다시 실행해도 이미 만든 뉴스 보정은 지우지 않는다
        out.update(news_consensus=previous[0], news_consensus_conf=previous[1],
                   signal=_adjust(out["signal"], previous[0], previous[1]),
                   news_status="이전에 만든 뉴스 채점을 다시 씀")
    made_at = save(out)  # 운영 신호를 먼저 남긴다(뉴스 채점이 실패해도 신호 기록은 빠지지 않게)
    if with_news:
        if not news_ready(t):
            out["news_status"] = f"{t.date()} {config.SIGNAL_CUTOFF_KST} 이후에 뉴스를 채점할 수 있다"
        else:
            try:
                trace = news_score(t)
            except Exception as e:  # LLM 요청 제한·하루 한도 등. 뉴스는 참고용이라 실패를 기록하고 신호는 그대로 둔다
                out["news_status"] = f"뉴스 채점 실패: {type(e).__name__}: {str(e)[:120]}"
            else:
                final = trace["final"]
                out.update(news_consensus=final["score"], news_consensus_conf=final["confidence"],
                           signal=_adjust(out["signal"], final["score"], final["confidence"]),
                           news_status=f"채점 완료 (기사 {len(trace['articles'])}건, ${trace['cost_usd']:.3f})")
        save(out, created_at=made_at)
    return out


def _adjust(signal: str, consensus: float, confidence: float) -> str:
    """운영 신호에 뉴스 보정을 쓸 때만 보정한다(config.NEWS_ADJUST_IN_SIGNAL). 끄면 숫자 모델 신호 그대로다."""
    return news_signal.adjust(signal, consensus, confidence) if config.NEWS_ADJUST_IN_SIGNAL else signal


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


def model_config(feature_set: str) -> str:
    """신호를 만든 모델 설정 요약(나중에 어떤 조건의 신호였는지 추적하려고 기록에 함께 남긴다). 기간별 모델(h<N>)은 운영 지표 묶음이다."""
    h = tools.model_horizon(feature_set)
    if h != config.OPERATIONAL_HORIZON:
        feature_set = "base"
    n = len(feature_columns(pd.DataFrame(columns=config.FEATURE_SETS["candidate"]), feature_set)) if feature_set != config.LIVE_RISK_MODEL else len(config.FEATURE_SET_BASE)
    return (f"지표 {feature_set}({n}개) · 정답 {h}거래일 ±{config.HORIZONS[h]:.1%} · 재학습 "
            f"{config.RETRAIN_FREQS[config.RETRAIN_FREQ]} · 학습 {(config.OPERATIONAL_TRAIN_FROM + '~(확장)') if config.OPERATIONAL_EXTENDED_HISTORY else (config.DATA_START + '~')}{'' if not config.TRAIN_WINDOW_YEARS else f'(최근 {config.TRAIN_WINDOW_YEARS}년)'} · "
            f"뉴스 {'보정' if config.NEWS_ADJUST_IN_SIGNAL else '참고용'} · 하락 {config.DOWN_ACTION_LABELS[config.DOWN_ACTION]}")


def save(result: dict, created_at: pd.Timestamp | None = None) -> pd.Timestamp:
    """신호 결과 한 줄을 signals_live.parquet 에 (날짜, 모델) 기준으로 덮어 쓰고 기록 시각을 돌려준다.
    created_at 을 주면 그 시각을 그대로 남긴다(같은 실행에서 뉴스 채점 결과를 덧붙일 때 신호를 만든 시각을 지키려고)."""
    row = pd.DataFrame([{k: v for k, v in result.items() if k != "data_sources"}]).set_index("date")
    row["model"] = result.get("model", "base")
    row["model_config"] = model_config(row["model"].iloc[0])
    stamp = created_at or pd.Timestamp.now(tz=config.KST).tz_localize(None)
    row["created_at"] = stamp  # 실제로 그 시각에 만든 신호인지 가리기 위해
    if LIVE_PATH.exists():
        old = pd.read_parquet(LIVE_PATH)
        if "model" not in old.columns:
            old["model"] = "base"
        same = (old.index == row.index[0]) & (old["model"] == row["model"].iloc[0])
        row = pd.concat([old[~same], row]).sort_index()
    LIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    row.to_parquet(LIVE_PATH)
    return stamp


def main(args: list[str]):
    """명령줄: python -m src.pipeline.live_signal [YYYY-MM-DD] [--news]"""
    date = next((a for a in args if not a.startswith("--")), None)
    out = make_signal(date, with_news="--news" in args)
    print({k: v for k, v in out.items() if k != "data_sources"})


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
