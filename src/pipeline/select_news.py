import re
from pathlib import Path

import pandas as pd

from src import config
from src.pipeline.collect_news import signal_window, trading_days


def categorize(title: str) -> str | None:
    """제목에 들어 있는 단어로 분류를 정한다. 여러 분류에 걸리면 config 순서상 앞의 분류를 쓴다."""
    for name, pattern in config.SELECT_PATTERNS.items():
        if re.search(pattern, title, flags=re.IGNORECASE):
            return name
    return None


def _dedup_key(title: str) -> str:
    """비슷한 제목을 묶기 위해 공백·기호를 지운 앞부분을 키로 쓴다."""
    return re.sub(r"[\W_]+", "", title.lower())[: config.SELECT_DEDUP_PREFIX]


def select_articles(news: pd.DataFrame, signal_date: pd.Timestamp, days: pd.DatetimeIndex) -> pd.DataFrame:
    """신호일 t 의 기사 중 채점할 기사를 최대 SELECT_TOTAL 건 고른다.

    분류별 할당대로 뽑고 모자라면 앞 분류부터 채운다. 분류 안에서는 15:15 에 가까운 기사부터,
    매체당 최대 SELECT_MAX_PER_SOURCE 건, 비슷한 제목은 하나만 쓴다. 구간 밖 기사가 있으면 예외를 던진다.
    """
    t = pd.Timestamp(signal_date)
    day = news[news["signal_date"] == t].copy()
    if day.empty:
        return day
    start, end = signal_window(t, days)
    published = pd.to_datetime(day["published_utc"], utc=True)
    if (published.isna() | (published < start) | (published >= end)).any():
        raise ValueError(f"{t.date()} 신호 구간 밖이거나 발행 시각이 없는 기사가 있다")

    day["title"] = day["title"].fillna("").str.strip().str.slice(0, config.TITLE_MAX_CHARS)
    day = day[day["title"] != ""]
    day["category"] = day["title"].map(categorize)
    day = day[day["category"].notna()]
    day["_key"] = day["title"].map(_dedup_key)
    day = day.sort_values("published_utc", ascending=False).drop_duplicates("_key")

    picked, per_source = [], {}

    def take(pool: pd.DataFrame, limit: int):
        for idx, row in pool.iterrows():
            if limit <= 0 or len(picked) >= config.SELECT_TOTAL:
                return
            if idx in picked or per_source.get(row["source"], 0) >= config.SELECT_MAX_PER_SOURCE:
                continue
            picked.append(idx)
            per_source[row["source"]] = per_source.get(row["source"], 0) + 1
            limit -= 1

    for name, quota in config.SELECT_QUOTAS.items():
        take(day[day["category"] == name], quota)
    for name in config.SELECT_QUOTAS:  # 할당을 못 채운 만큼 앞 분류부터 채운다
        take(day[day["category"] == name], config.SELECT_TOTAL - len(picked))

    out = day.loc[picked].drop(columns="_key")
    return out.sort_values("published_utc").reset_index(drop=True)


def main():
    """news.parquet 의 신호일마다 기사를 골라 data/processed/selected_news.parquet 로 저장하고 분류별 건수를 출력한다."""
    news = pd.read_parquet(Path(config.PROCESSED_DIR) / "news.parquet")
    days = trading_days()
    frames = [select_articles(news, t, days) for t in sorted(news["signal_date"].unique())]
    selected = pd.concat(frames, ignore_index=True)
    out = Path(config.PROCESSED_DIR) / "selected_news.parquet"
    selected.to_parquet(out)
    print(out, len(selected))
    print(selected.groupby(["signal_date", "category"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
