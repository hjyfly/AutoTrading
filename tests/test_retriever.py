import hashlib

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src import retriever as rt


class FakeEmbeddings(Embeddings):
    """글자 단위 해시로 만든 가짜 임베딩. 네트워크·비용 없이 검색 동작만 확인한다."""

    def _vec(self, text):
        v = [0.0] * 32
        for ch in text:
            v[int(hashlib.md5(ch.encode()).hexdigest(), 16) % 32] += 1
        return v

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def store(tmp_path):
    docs = [
        Document("3월 4일 판단: 반도체 우려로 부정", metadata={"type": "judgement", "date": "2026-03-04", "available_from": 20260304, "source": "a"}),
        Document("3월 11일 판단: 갤럭시 출시로 긍정", metadata={"type": "judgement", "date": "2026-03-11", "available_from": 20260311, "source": "b"}),
        Document("사업보고서 개요 반도체", metadata={"type": "dart", "date": "2026-03-10", "available_from": 20260311, "source": "c"}),
        Document("3일차 보고서: 기준 A 실패", metadata={"type": "report", "date": "2026-10-21", "available_from": 20261021, "source": "d"}),
    ]
    return Chroma.from_documents(docs, FakeEmbeddings(), collection_name="test_docs", persist_directory=str(tmp_path))


def test_as_of_이후에_공개된_문서는_나오지_않는다(store):
    found = rt.search("판단 반도체 보고서", as_of="2026-03-10", k=10, store=store)
    assert {d["source"] for d in found} == {"a"}


def test_공시는_접수_다음날부터_나온다(store):
    assert "c" not in {d["source"] for d in rt.search("사업보고서", as_of="2026-03-10", k=10, store=store)}
    assert "c" in {d["source"] for d in rt.search("사업보고서", as_of="2026-03-11", k=10, store=store)}


def test_보고서는_작성일_이후_질문에만(store):
    assert "d" not in {d["source"] for d in rt.search("기준 A", as_of="2026-04-30", k=10, store=store)}
    assert "d" in {d["source"] for d in rt.search("기준 A", as_of="2026-10-21", k=10, store=store)}


def test_문서_종류로_거른다(store):
    found = rt.search("반도체", as_of="2026-10-21", k=10, types=["dart"], store=store)
    assert [d["type"] for d in found] == ["dart"]


def test_마크다운_절_단위로_나눈다():
    chunks = rt.split_markdown("# 제목\n머리\n## 가\n" + "x" * 30 + "\n## 나\n짧음", max_chars=20)
    assert chunks[0].startswith("# 제목")
    assert any(c.startswith("## 나") for c in chunks)
    assert all(len(c) <= 20 for c in chunks)
