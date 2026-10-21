import html
import io
import os
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src import config

load_dotenv()

DART_LIST_URL = "https://opendart.fss.or.kr/api/list.json"
DART_DOCUMENT_URL = "https://opendart.fss.or.kr/api/document.xml"
OVERVIEW_TITLE = re.compile(r"<TITLE[^>]*>\s*1\.\s*사업의\s*개요\s*</TITLE>")
NEXT_TITLE = re.compile(r"<TITLE[^>]*>\s*2\.\s*주요\s*제품")


def list_periodic_reports(start: str, end: str) -> pd.DataFrame:
    """삼성전자 정기공시(사업·반기·분기보고서) 목록을 접수일과 함께 받는다."""
    resp = requests.get(DART_LIST_URL, params={
        "crtfc_key": os.environ["DART_API_KEY"],
        "corp_code": config.DART_CORP_CODE,
        "bgn_de": start.replace("-", ""),
        "end_de": end.replace("-", ""),
        "pblntf_ty": "A",
        "page_count": 100,
    }, timeout=60)
    body = resp.json()
    if body.get("status") != "000":
        raise RuntimeError(f"DART 목록 조회 실패: {body.get('status')} {body.get('message')}")
    df = pd.DataFrame(body["list"])[["rcept_no", "report_nm", "rcept_dt"]]
    df["rcept_dt"] = pd.to_datetime(df["rcept_dt"])
    return df.sort_values("rcept_dt").reset_index(drop=True)


def extract_overview(xml_text: str) -> str:
    """공시 원문에서 '1. 사업의 개요' 부분을 태그를 지운 평문으로 뽑는다. 찾지 못하면 빈 문자열을 돌려준다."""
    start = OVERVIEW_TITLE.search(xml_text)
    if not start:
        return ""
    end = NEXT_TITLE.search(xml_text, start.end())
    section = xml_text[start.end(): end.start() if end else start.end() + 200_000]
    text = html.unescape(re.sub(r"<[^>]+>", " ", section))
    return re.sub(r"\s+", " ", text).strip()


def download_report(rcept_no: str) -> tuple[Path, str]:
    """공시 원문 zip 을 data/raw/dart 에 저장하고, 사업의 개요 평문을 돌려준다. 이미 받은 파일은 다시 받지 않는다."""
    path = Path(config.RAW_DIR) / "dart" / f"{rcept_no}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        resp = requests.get(DART_DOCUMENT_URL, params={"crtfc_key": os.environ["DART_API_KEY"], "rcept_no": rcept_no}, timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
        time.sleep(config.REQUEST_PAUSE_SEC)
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as z:
        main_doc = max(z.namelist(), key=lambda n: z.getinfo(n).file_size)  # 본문이 가장 큰 파일이다
        xml_text = z.read(main_doc).decode("utf-8", errors="replace")
    return path, extract_overview(xml_text)


def dart_as_of(index: pd.DataFrame, as_of) -> pd.DataFrame:
    """as_of 날짜보다 먼저 접수된 공시만 돌려준다. 같은 날 접수된 공시는 15:15 이전인지 알 수 없어 제외한다.

    as_of 에 시각이나 시간대가 붙어 있어도 날짜만 남겨 비교한다.
    """
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is not None:
        as_of_ts = as_of_ts.tz_convert(config.KST).tz_localize(None)
    return index[index["rcept_dt"] < as_of_ts.normalize()]


def main():
    """정기공시 목록과 원문을 받고, 사업의 개요를 뽑아 data/processed/dart_index.parquet 로 저장한다."""
    reports = list_periodic_reports(config.DATA_START, config.DATA_END)
    paths, overviews = [], []
    for rcept_no in reports["rcept_no"]:
        path, overview = download_report(rcept_no)
        paths.append(str(path))
        overviews.append(overview)
    reports["raw_path"] = paths
    reports["overview"] = overviews
    reports["is_dummy"] = False
    out = Path(config.PROCESSED_DIR) / "dart_index.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    reports.to_parquet(out)
    for _, row in reports.iterrows():
        print(row["rcept_dt"].date(), row["report_nm"], f"개요 {len(row['overview'])}자")


if __name__ == "__main__":
    main()
