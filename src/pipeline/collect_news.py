import csv
import html
import io
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

from src import config

KST = config.KST
GKG_URL = "https://data.gdeltproject.org/gdeltv2/{stamp}.{kind}.csv.zip"
GKG_KINDS = ["gkg", "translation.gkg"]
RSS_URL = "https://news.google.com/rss/search"
RSS_HEADERS = {"User-Agent": "Mozilla/5.0"}
TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>")
csv.field_size_limit(sys.maxsize if sys.maxsize < 2**63 else 2**31 - 1)


def trading_days() -> pd.DatetimeIndex:
    """삼성전자 원본 일봉의 날짜를 한국 거래일 목록으로 쓴다."""
    ohlcv = pd.read_parquet(Path(config.RAW_DIR) / "prices" / "samsung_ohlcv.parquet")
    return pd.DatetimeIndex(ohlcv.index).sort_values()


def sample_signal_dates(days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """뉴스 구간에서 매주 지정 요일의 거래일을 고른다. 그날이 휴장이면 다음 거래일로 대신한다."""
    in_range = days[(days >= config.NEWS_START) & (days <= config.NEWS_END)]
    calendar = pd.date_range(config.NEWS_START, config.NEWS_END, freq="D")
    picked = []
    for target in calendar[calendar.weekday == config.SAMPLE_WEEKDAY]:
        later = in_range[in_range >= target]
        if len(later) and later[0] not in picked and later[0].isocalendar().week == target.isocalendar().week:
            picked.append(later[0])
    return picked


def signal_window(signal_date: pd.Timestamp, days: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    """신호일 t 의 뉴스 구간 [t-1 거래일 15:15 KST, t일 15:15 KST) 의 시작·끝을 UTC 시각으로 돌려준다."""
    prev_day = days[days < signal_date][-1]
    hour, minute = map(int, config.SIGNAL_CUTOFF_KST.split(":"))
    cutoff = pd.Timedelta(hours=hour, minutes=minute)
    start = (prev_day + cutoff).tz_localize(KST).tz_convert("UTC")
    end = (signal_date + cutoff).tz_localize(KST).tz_convert("UTC")
    return start, end


def assign_signal_date(published: pd.Timestamp, days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """발행 시각이 속하는 신호일을 돌려준다. KST 15:15 전이면 그날(거래일일 때), 15:15 정각 이후면 다음 거래일."""
    hour, minute = map(int, config.SIGNAL_CUTOFF_KST.split(":"))
    local = published.tz_convert(KST).tz_localize(None)
    day = local.normalize()
    if local - day >= pd.Timedelta(hours=hour, minutes=minute):
        day = day + pd.Timedelta(days=1)
    later = days[days >= day]
    return later[0] if len(later) else None


def rss_effective_time(pub_date: pd.Timestamp) -> pd.Timestamp:
    """Google 뉴스 RSS 의 pubDate 는 미국 서부 자정으로 잘려 날짜만 믿을 수 있다.

    그 날짜(미국 서부)의 끝을 발행 시각으로 보수적으로 잡아, 15:15 이후 기사가 신호에 섞이지 않게 한다.
    """
    pacific_day = pub_date.tz_convert("America/Los_Angeles").tz_localize(None).normalize()
    return (pacific_day + pd.Timedelta(days=1)).tz_localize("America/Los_Angeles").tz_convert("UTC")


def _matches(text: str, keywords: list[str]) -> list[str]:
    """본문에 들어 있는 키워드 목록을 돌려준다(대소문자 무시)."""
    lowered = text.lower()
    return [k for k in keywords if k.lower() in lowered]


def _fetch_gkg(stamp: str, kind: str) -> list[dict]:
    """GKG 원본 파일 하나를 메모리로 받아 키워드에 맞는 기사만 남긴다. 원본은 저장하지 않는다."""
    url = GKG_URL.format(stamp=stamp, kind=kind)
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            break
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    else:
        return [{"_error": url}]
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
    keywords = config.NEWS_KEYWORDS_KO + config.NEWS_KEYWORDS_EN
    rows = []
    for rec in csv.reader(io.StringIO(raw), delimiter="\t"):
        if len(rec) < 27:
            continue
        title_match = TITLE_RE.search(rec[26])
        title = html.unescape(title_match.group(1)) if title_match else ""
        hits = _matches(" ".join([title, rec[4], rec[13]]), keywords)  # 제목, URL, 기관명
        if not hits:
            continue
        rows.append({
            "published_utc": pd.Timestamp(datetime.strptime(rec[1], "%Y%m%d%H%M%S"), tz="UTC"),
            "source": rec[3],
            "url": rec[4],
            "title": title,
            "keywords": ",".join(hits),
            "tone": float(rec[15].split(",")[0]) if rec[15] else None,
            "stream": kind,
            "provider": "gdelt_gkg",
        })
    return rows


def collect_gdelt(window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    """구간에 해당하는 15분 단위 GKG 파일(일반·번역)을 받아 걸러진 기사를 모은다.

    15:15 정각 배치는 15:15 에 아직 받을 수 없으므로 넣지 않는다(구간 끝 미포함).
    """
    stamps = pd.date_range(window_start.floor("15min"), window_end, freq="15min", inclusive="left")
    jobs = [(s.strftime("%Y%m%d%H%M%S"), kind) for s in stamps for kind in GKG_KINDS]
    with ThreadPoolExecutor(max_workers=config.NEWS_DOWNLOAD_WORKERS) as pool:
        results = list(pool.map(lambda job: _fetch_gkg(*job), jobs))
    rows = [r for batch in results for r in batch]
    errors = [r["_error"] for r in rows if "_error" in r]
    df = pd.DataFrame([r for r in rows if "_error" not in r])
    if df.empty:
        return df.assign(download_errors=len(errors))
    df = df[(df["published_utc"] >= window_start) & (df["published_utc"] < window_end)]
    return df.assign(download_errors=len(errors))


def collect_rss(signal_date: pd.Timestamp, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Google 뉴스 RSS 에서 한국어 기사를 날짜 지정으로 검색하고, 보수적인 발행 시각으로 신호일을 배정한다."""
    prev_day = days[days < signal_date][-1]
    after = (prev_day - pd.Timedelta(days=config.RSS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    before = signal_date.strftime("%Y-%m-%d")
    rows = []
    for keyword in config.NEWS_KEYWORDS_KO:
        resp = requests.get(RSS_URL, params={"q": f"{keyword} after:{after} before:{before}", "hl": "ko", "gl": "KR", "ceid": "KR:ko"},
                            headers=RSS_HEADERS, timeout=60)
        resp.raise_for_status()
        for item in ET.fromstring(resp.content).findall(".//item"):
            pub_date = pd.Timestamp(parsedate_to_datetime(item.findtext("pubDate"))).tz_convert("UTC")
            rows.append({
                "rss_pub_date_utc": pub_date,
                "published_utc": rss_effective_time(pub_date),
                "source": item.findtext("source") or "",
                "url": item.findtext("link"),
                "title": item.findtext("title") or "",
                "keywords": keyword,
                "tone": None,
                "stream": "google_news_rss",
                "provider": "google_news_rss",
            })
        time.sleep(config.REQUEST_PAUSE_SEC)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["signal_date"] = [assign_signal_date(p, days) for p in df["published_utc"]]
    return df[df["signal_date"] == signal_date]


def within_window(df: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """각 기사가 자기 신호일 구간 [t-1 15:15, t 15:15) KST 안에 있는지 다시 확인해 벗어난 기사를 뺀다.

    15:15 정각 배치는 그날 신호에는 늦고, 다음 신호일에는 쓸 수 있다.
    """
    keep = []
    for t, part in df.groupby("signal_date"):
        start, end = signal_window(pd.Timestamp(t), days)
        published = pd.to_datetime(part["published_utc"], utc=True)
        keep.append(part[(published >= start) & (published < end)])
    return pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0]


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """같은 URL, 같은 제목(공백·대소문자 무시)의 중복 기사를 지운다. GDELT 를 먼저 남긴다."""
    df = df.sort_values(["provider", "published_utc"])
    df = df.drop_duplicates("url")
    norm = df["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return df[~(norm.duplicated() & (norm != ""))]


def _save_raw(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """수집 시각과 더미 표시를 붙여 원본으로 저장한다. 이미 있으면 덮어쓰지 않고 예외를 던진다."""
    if path.exists():
        raise FileExistsError(f"원본은 수정하지 않는다: {path}")
    df = df.assign(collected_at=datetime.now(timezone.utc).isoformat(), is_dummy=False)
    df.to_parquet(path)
    return df


def save_day(t: pd.Timestamp, gdelt: pd.DataFrame, rss: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """평일 15:20 실시간 채점이 받은 t 일 기사를 남긴다(main 과 같은 형식). 화면·도구가 같은 기사를 다시 받지 않게 한다.

    원본: data/raw/news/<날짜>.parquet(GDELT)·rss_<날짜>.parquet. 이미 있으면 덮어쓰지 않는다.
    수집본: data/processed/news.parquet 에 신호일이 t 인 행만 덧붙인다. 그날 행이 이미 있거나 다른 날 행은 바꾸지 않는다.
    15:15 신호 시각 뒤에 받지만 신호일 구간(전 거래일 15:15 ~ t 15:15) 안의 기사만 남긴다(within_window).
    """
    raw_dir = Path(config.RAW_DIR) / "news"
    raw_dir.mkdir(parents=True, exist_ok=True)
    gdelt_path, rss_path = raw_dir / f"{t.date()}.parquet", raw_dir / f"rss_{t.date()}.parquet"
    gdelt = pd.read_parquet(gdelt_path) if gdelt_path.exists() else _save_raw(gdelt.assign(signal_date=t), gdelt_path)  # 원본이 있으면 원본으로(수집본과 같게)
    if rss_path.exists():
        rss = pd.read_parquet(rss_path)
    elif len(rss):
        rss = _save_raw(rss, rss_path)
    path = Path(config.PROCESSED_DIR) / "news.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if not old.empty and (old["signal_date"] == t).any():
        return old[old["signal_date"] == t]
    day = dedupe(within_window(pd.concat([gdelt, rss], ignore_index=True), days))
    day = day[day["signal_date"] == t].copy()  # RSS 는 늦게 배정된 다음 신호일 행이 섞일 수 있어 그날 행만
    day["published_kst"] = pd.to_datetime(day["published_utc"], utc=True).dt.tz_convert(KST)
    pd.concat([old, day], ignore_index=True).to_parquet(path)
    return day


def main(signal_dates: list[str] | None = None):
    """샘플 신호일마다 GDELT·RSS 기사를 모아 data/raw/news 와 data/processed/news.parquet 에 저장한다.

    signal_dates 가 ["all"] 이면 뉴스 구간의 모든 거래일, 없으면 샘플 신호일(매주 수요일)을 수집한다.
    원본: GDELT 는 <날짜>.parquet, RSS 는 rss_<날짜>.parquet. 이미 있는 원본은 다시 받지 않는다.
    처음 수집한 <날짜>.parquet 에는 발행 시각 처리가 잘못된 RSS 행이 섞여 있어, 그 파일에서는 GDELT 행만 쓴다.
    """
    days = trading_days()
    if signal_dates == ["all"]:  # 뉴스 구간의 모든 거래일
        targets = list(days[(days >= config.NEWS_START) & (days <= config.NEWS_END)])
    elif signal_dates:
        targets = [pd.Timestamp(d) for d in signal_dates]
    else:
        targets = sample_signal_dates(days)
    raw_dir = Path(config.RAW_DIR) / "news"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in targets:
        gdelt_path, rss_path = raw_dir / f"{t.date()}.parquet", raw_dir / f"rss_{t.date()}.parquet"
        if gdelt_path.exists():
            gdelt = pd.read_parquet(gdelt_path)
            gdelt = gdelt[gdelt["provider"] == "gdelt_gkg"]
        else:
            start, end = signal_window(t, days)
            gdelt = _save_raw(collect_gdelt(start, end).assign(signal_date=t), gdelt_path)
        if rss_path.exists():
            rss = pd.read_parquet(rss_path)
        else:
            try:
                rss = _save_raw(collect_rss(t, days), rss_path)
            except Exception as e:
                print(t.date(), "RSS 실패:", type(e).__name__)
                rss = pd.DataFrame()
        frames += [gdelt, rss]
        print(t.date(), f"GDELT {len(gdelt)}건, RSS {len(rss)}건", flush=True)
    news = dedupe(within_window(pd.concat(frames, ignore_index=True), days))
    news["published_kst"] = pd.to_datetime(news["published_utc"], utc=True).dt.tz_convert(KST)
    path = Path(config.PROCESSED_DIR) / "news.parquet"
    if path.exists():  # 이번 대상이 아닌 신호일 행(15:20 실시간 채점일 save_day 행 등)은 그대로 둔다
        old = pd.read_parquet(path)
        news = pd.concat([old[~old["signal_date"].isin(news["signal_date"].unique())], news], ignore_index=True)
    news.to_parquet(path)
    print(path, len(news))
    print(news.groupby(["signal_date", "provider"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or None)
