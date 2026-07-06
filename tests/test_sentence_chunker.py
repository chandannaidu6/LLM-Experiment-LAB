import pytest
from chunkers.sentence_chunking import SentenceChunker

TEXT = ("First sentence here. Second sentence here.\n\n"
        "A second paragraph with quite a few more words than the small limit allows for.")


@pytest.fixture
def sc():
    return SentenceChunker("x.pdf",sentence_per_chunk=3)

def test_groups_respect_sentence_count(sc, monkeypatch):
    units = [
        {"text": f"Sentence {i}.", "doc_name": "x.pdf", "page_number": 1, "global_idx": i}
        for i in range(1, 8)   # 7 sentences, size 3 -> groups of 3,3,1
    ]
    monkeypatch.setattr(sc, "extract", lambda: units)
    chunks = sc.build_chunks()
    assert [c["unit_count"] for c in chunks] == [3, 3, 1]

def test_no_sentences_lost(sc, monkeypatch):
    units = [
        {"text": f"S{i}.", "doc_name": "x.pdf", "page_number": 1, "global_idx": i}
        for i in range(1, 8)
    ]
    monkeypatch.setattr(sc, "extract", lambda: units)
    chunks = sc.build_chunks()
    total = sum(c["unit_count"] for c in chunks)
    assert total == 7
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_single_group_is_full(sc, monkeypatch):
    units = [{"text": "Only one.", "doc_name": "x.pdf", "page_number": 1, "global_idx": 1}]
    monkeypatch.setattr(sc, "extract", lambda: units)
    chunks = sc.build_chunks()
    assert len(chunks) == 1
    assert chunks[0]["position"] == "full"

