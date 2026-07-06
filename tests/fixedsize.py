import os
from pathlib import Path

import pytest
from chunkers.fixed_size import FixedSize


DATA_DIR = r"C:\Desktop\LLM Experiment lab\Data\MLM.pdf"

@pytest.fixture(scope="module")
def sample_pdf_path():
    pdf = DATA_DIR
    if not pdf:
        pytest.skip("sample.pdf does not exist")
    return pdf

def test_extract_pages(sample_pdf_path):
    chunker = FixedSize(str(sample_pdf_path))
    pages = chunker.extract_pages()

    assert isinstance(pages,list)
    assert len(pages) > 0

    first = pages[0]
    assert "text" in first
    assert "page_number" in first
    assert "text" in first
    assert isinstance(first["text"],str)


def test_metadata_sentences():
    dummy_data = [
        {
            "text":"This is long enough sentence. Another long enough sentence.",
            "page_number":10,
            "doc_name":"Dummy.pdf"
        }
    ]

    chunker = FixedSize(dummy_data)
    sentences = chunker.words_with_metadata(dummy_data)

    assert len(sentences) >= 2
    first =  sentences[0]
    assert first['doc_name'] == "Dummy.pdf"
    assert first['page_number'] == 10
    assert "text" in first
    assert "global_idx" in first
    assert isinstance(first['global_idx'],int)

def test_chunk_sentences():
    word_units = [
        {
            "text":f"Sentence {i} that is long for testing",
            "doc_name": "Dummy.pdf",
            "page_number":i//3,
            "global_idx": i
        }
        for i in range(10)

    ]
    chunks = FixedSize.chunk_words(word_units,chunk_size=5,overlap=2)
    assert len(chunks) > 0
    first = chunks[0]
    for i in ["text","doc_name","pages","source_unit","position"]:
        assert i in first        



    
