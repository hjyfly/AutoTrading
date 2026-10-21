"""RAG 파이프라인. 판단 기록·DART 사업 개요·보고서를 Chroma 에 색인하고, as_of 시점에 공개돼 있던 문서만 검색한다.

문서마다 available_from(그 문서를 처음 알 수 있던 날)을 붙인다.
- 판단 기록: 그날(신호일) 15:15 에 만들어졌으므로 신호일부터.
- DART: 접수 다음 날부터(접수 당일은 15:15 이전인지 알 수 없다).
- 보고서: 작성일부터. 검증 구간 전체 결과를 담고 있어, 과거 날짜 질문에는 나오지 않는다.
"""
import json
import re
import shutil
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config, llm
from src.tools import capped, resolve_as_of

REPORT_DATE = re.compile(r"작성일:\s*(\d{4}-\d{2}-\d{2})")


def _day_int(ts) -> int:
    """날짜를 Chroma 에서 비교할 수 있는 정수(YYYYMMDD)로 바꾼다."""
    return int(pd.Timestamp(ts).strftime("%Y%m%d"))


def judgement_documents() -> list[Document]:
    """판단 기록 JSON 마다 중재자 정리 1건과 채점자 의견 3건을 문서로 만든다."""
    docs = []
    for path in sorted(Path(config.TRACES_DIR).glob("*.json")):
        t = json.loads(path.read_text(encoding="utf-8"))
        titles = {a["no"]: a["title"] for a in t["articles"]}
        meta = {"type": "judgement", "date": t["signal_date"], "available_from": _day_int(t["signal_date"]),
                "source": f"뉴스 채점 판단 기록 {t['signal_date']}"}
        f = t["final"]
        key = "; ".join(titles.get(n, "") for n in f["key_articles"])
        docs.append(Document(f"[{t['signal_date']} 뉴스 채점 최종] 점수 {f['score']}, 확신도 {f['confidence']}. "
                             f"근거: {f['rationale']} 남은 이견: {f['disagreement']} 핵심 기사: {key}",
                             metadata={**meta, "role": "moderator"}))
        for p, o in t["round2"].items():
            docs.append(Document(f"[{t['signal_date']} 채점자 {p}] 점수 {o['score']}, 확신도 {o['confidence']}. "
                                 f"근거: {o['rationale']} 판단 변경: {o.get('change_reason', '')}",
                                 metadata={**meta, "role": p}))
    return docs


def dart_documents() -> list[Document]:
    """정기보고서 사업 개요를 문서로 만든다. 접수 다음 날부터 검색된다."""
    index = pd.read_parquet(Path(config.PROCESSED_DIR) / "dart_index.parquet")
    return [Document(f"[{r.report_nm}, 접수일 {r.rcept_dt.date()}] {r.overview}",
                     metadata={"type": "dart", "date": str(r.rcept_dt.date()),
                               "available_from": _day_int(r.rcept_dt + pd.Timedelta(days=1)), "source": f"DART {r.report_nm}"})
            for r in index.itertuples()]


def split_markdown(text: str, max_chars: int) -> list[str]:
    """'## ' 제목 단위로 나누고, 긴 절은 max_chars 글자씩 자른다."""
    chunks = []
    for section in re.split(r"\n(?=## )", text):
        chunks += [section[i:i + max_chars] for i in range(0, len(section), max_chars)]
    return [c.strip() for c in chunks if c.strip()]


def report_documents() -> list[Document]:
    """보고서를 절 단위 문서로 만든다. 작성일부터 검색된다(작성일이 없으면 파일 수정일)."""
    docs = []
    for name in config.RAG_REPORTS:
        path = Path(name)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        found = REPORT_DATE.search(text)
        written = pd.Timestamp(found.group(1)) if found else \
            pd.Timestamp(path.stat().st_mtime, unit="s").tz_localize("UTC").tz_convert(config.KST).tz_localize(None).normalize()
        for chunk in split_markdown(text, config.RAG_CHUNK_CHARS):
            docs.append(Document(chunk, metadata={"type": "report", "date": str(written.date()),
                                                  "available_from": _day_int(written), "source": name}))
    return docs


def build_index() -> int:
    """색인을 처음부터 다시 만든다. 판단 기록이 늘면 다시 돌린다."""
    shutil.rmtree(config.INDEX_DIR, ignore_errors=True)
    docs = judgement_documents() + dart_documents() + report_documents()
    Chroma.from_documents(docs, llm.get_embeddings(), collection_name=config.INDEX_COLLECTION,
                          persist_directory=config.INDEX_DIR)
    return len(docs)


def _store(embeddings=None) -> Chroma:
    """저장된 색인을 연다."""
    return Chroma(collection_name=config.INDEX_COLLECTION, embedding_function=embeddings or llm.get_embeddings(),
                  persist_directory=config.INDEX_DIR)


def as_of_filter(as_of, types: list[str] | None = None) -> dict:
    """available_from ≤ as_of 인 문서만 고르는 Chroma 조건. types 를 주면 문서 종류도 거른다."""
    cond = {"available_from": {"$lte": _day_int(resolve_as_of(as_of))}}
    if types:
        return {"$and": [cond, {"type": {"$in": types}}]}
    return cond


def search(query: str, as_of: str | None = None, k: int = 5, types: list[str] | None = None, store: Chroma | None = None) -> list[dict]:
    """as_of 시점에 공개돼 있던 문서 중 질문과 가까운 k 개를 돌려준다. 결과마다 출처와 날짜를 붙인다.

    요청 상한(tools.request_as_of) 안이면 as_of 를 상한 이하로 맞춘다.
    """
    store = store or _store()
    t, _ = capped(as_of)
    docs = store.similarity_search(query, k=k, filter=as_of_filter(t, types))
    cut = _day_int(t)
    if any(d.metadata["available_from"] > cut for d in docs):
        raise ValueError("as_of 이후에 공개된 문서가 검색 결과에 섞였다")
    return [{"text": d.page_content, **d.metadata} for d in docs]


def main():
    """색인을 다시 만들고 문서 종류별 개수를 출력한다."""
    n = build_index()
    print(config.INDEX_DIR, f"문서 {n}개", f"누적 비용 ${llm.total_cost():.4f}")


if __name__ == "__main__":
    main()
