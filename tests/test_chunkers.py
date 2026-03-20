import os
from pathlib import Path

import pytest
from chunkers.slide_chunking import SlideChunker

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "Data"

@pytest.fixture(scope="module")
def sample_pdf_path(tmp_path):
    pdf = DATA_DIR / "sample.pdf"
    if not pdf.exists():
        pytest.skip("sample.pdf does not exist")
    return pdf

def test_extract_text(sample_pdf_path):
    chunker = SlideChunker(str(sample_pdf_path))
    pages = chunker.extract_text()

    assert isinstance(pages,list)
    assert len(pages) > 0

    first = pages[0]
    assert "text" in first
    assert "number" in first
    assert "title" in first
    assert isinstance(first["text"],str)


def test_split_sentences():
    text = "This is a long sentence for testing. Short. Another long enough sentence here!."
    sentences = SlideChunker.split_sentences(text,min_length=20)
    assert all(len(s) >= 20 for s in sentences)
    assert "Short." not in sentences[0]

def test_metadata_sentences():
    dummy_data = [
        {
            "text":"This is long enough sentence. Another long enough sentence.",
            "number":10,
            "title":"Dummy.pdf"
        }
    ]

    chunker = SlideChunker(dummy_data)
    sentences = chunker.metadata_sentences(dummy_data)

    assert len(sentences) >= 2
    first =  sentences[0]
    assert first['doc_name'] == "Dummy.pdf"
    assert first['page_number'] == 10
    assert "text" in first
    assert "global_idx" in first
    assert isinstance(first['global_idx'],int)

def test_chunk_sentences():
    sentences = [
        {
            {"text":f"Sentence {i} that is long for testing",
            "doc_name": "Dummy.pdf",
            "page_number":i//3,
            "global_idx": i}
            for i in range(10)
        }
    ]
    chunks = SlideChunker.chunk_sentences(sentences,chunk_size=5,overlap=2)
    assert len(chunks) > 0
    first = chunks[0]
    for i in ["text","doc_name","sentence_range","pages","source_sentences","position"]:
        assert i in first        

    sr_start,sr_end = first['sentence_range']
    assert sr_start == first['source_sentences'][0]['global_idx']
    assert sr_end == first['source_sentences'][-1]['global_idx']
    assert first['pages'] == sorted(set(first['pages']))

    #Checking the overlap 
    if len(chunks) > 1:
        first_range = chunks[0]['sentence_range']
        second_range = chunks[0]['sentence_range']
        assert first_range[0] < second_range[0] <= first_range[1]


    
