import pytest
from chunkers.recursive import RecursiveChunker

TEXT = ("First sentence here. Second sentence here.\n\n"
        "A second paragraph with quite a few more words than the small limit allows for.")


@pytest.fixture
def rc():
    return RecursiveChunker("x.pdf",max_size=8)

@pytest.fixture
def seps(rc):
    return [rc.split_paragraphs,rc.split_lines,rc.split_sentences,rc.split_spaces]

def test_no_chunk_exceeds_limit(rc,seps):
    for chunk in rc.recursive_split(TEXT,seps):
        assert len(chunk.split())<= rc.max_size

def test_no_words_lost(rc,seps):
    chunks = rc.recursive_split(TEXT,seps)
    assert " ".join(chunks).split() == TEXT.split()

def test_short_text(rc,seps):
    assert rc.recursive_split("just a few words", seps) == ["just a few words"]

def test_build_chunk_shape(rc,monkeypatch):
    units = [{"text": "hello there world", "doc_name": "x.pdf", "page_number": 1, "global_idx": 1}]
    monkeypatch.setattr(rc,"extract",lambda:units)
    chunks = rc.build_chunks()
    assert len(chunks) == 1
    c = chunks[0]
    assert c["text"] == "hello there world"
    assert c["chunker_name"] == "recursive"
    assert c["position"] == "full"
    assert c["word_count"] == 3

