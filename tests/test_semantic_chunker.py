import numpy as np 
import pytest
from chunkers.semantic_chunking import SemanticChunker

class DummyEmbedder:
    def __init__(self,mapping):
        self.mapping = mapping

    def embed_sentences(self,sentences):
        return np.array([self.mapping[s] for s in sentences],dtype=np.float32)

class TestableSemanticChunker(SemanticChunker):
    def __init__(self,sentence_units,embedder):
        super().__init__(pdf_path="dummy.pdf",embedder=embedder)
        self._sentence_units = sentence_units

    def extract_sentence_units(self,min_length=20):
        return self._sentence_units


def make_sentence_units(texts):
    return [
        {
            "text":text,
            "doc_name":"dummy.pdf",
            "page_number":1 if i<3 else 2,
            "global_idx":i,
        }
        for i,text in enumerate(texts)
    ]

def test_compute_adjacent_similarities():
    texts = ["A","B","C"]
    units = make_sentence_units(texts)
    embedder = DummyEmbedder({
            "A": [1.0, 0.0],
            "B": [1.0, 0.0],
            "C": [0.0, 1.0],
    })

    chunker = TestableSemanticChunker(units,embedder)
    sims = chunker.compute_adjacent_similarities(units)
    assert len(sims) == 2
    assert sims[0] == pytest.approx(1.0,rel=1e-5)
    assert sims[1] == pytest.approx(0.0,rel=1e-5)

def test_build_chunks_splits_on_similarity_drop():
    texts = [
        "S1 topic A",
        "S2 topic A",
        "S3 topic A",
        "S4 topic B",
        "S5 topic B",
        "S6 topic B",
    ]
    units = make_sentence_units(texts)

    embedder = DummyEmbedder(
        {
            "S1 topic A": [1.0, 0.0],
            "S2 topic A": [0.95, 0.05],
            "S3 topic A": [0.90, 0.10],
            "S4 topic B": [0.0, 1.0],
            "S5 topic B": [0.05, 0.95],
            "S6 topic B": [0.10, 0.90],
        }
    )

    chunker = TestableSemanticChunker(units, embedder)
    chunks = chunker.build_chunks(
        threshold=0.55,
        min_sentences=2,
        max_sentences=6,
    )

    assert len(chunks) == 2
    assert chunks[0]["sentence_count"] == 3
    assert chunks[1]["sentence_count"] == 3
    assert chunks[0]["start_idx"] == 0
    assert chunks[0]["end_idx"] == 2
    assert chunks[1]["start_idx"] == 3
    assert chunks[1]["end_idx"] == 5
    assert chunks[0]["chunker_name"] == "semantic"
    assert chunks[0]["source_unit"] == "sentences"

def test_build_chunks_forces_split_on_max_sentences():
    texts = [
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
    ]
    units = make_sentence_units(texts)

    embedder = DummyEmbedder(
        {
            "S1": [1.0, 0.0],
            "S2": [0.99, 0.01],
            "S3": [0.98, 0.02],
            "S4": [0.97, 0.03],
            "S5": [0.96, 0.04],
        }
    )

    chunker = TestableSemanticChunker(units, embedder)
    chunks = chunker.build_chunks(
        threshold=0.10,
        min_sentences=1,
        max_sentences=2,
    )

    assert len(chunks) == 3
    assert [c["sentence_count"] for c in chunks] == [2, 2, 1]

def test_small_trailing_chunk_gets_merged():
    texts = [
        "S1 A",
        "S2 A",
        "S3 A",
        "S4 B",
    ]
    units = make_sentence_units(texts)

    embedder = DummyEmbedder(
        {
            "S1 A": [1.0, 0.0],
            "S2 A": [0.95, 0.05],
            "S3 A": [0.90, 0.10],
            "S4 B": [0.0, 1.0],
        }
    )

    chunker = TestableSemanticChunker(units, embedder)
    chunks = chunker.build_chunks(
        threshold=0.55,
        min_sentences=2,
        max_sentences=6,
    )

    assert len(chunks) == 1
    assert chunks[0]["sentence_count"] == 4
    assert chunks[0]["boundary_reason"] == "merged_small_chunk"

def test_single_sentence_returns_full_chunk():
    texts = ["Only one sentence in the document."]
    units = make_sentence_units(texts)

    embedder = DummyEmbedder(
        {
            "Only one sentence in the document.": [1.0, 0.0],
        }
    )

    chunker = TestableSemanticChunker(units, embedder)
    chunks = chunker.build_chunks()

    assert len(chunks) == 1
    assert chunks[0]["position"] == "full"
    assert chunks[0]["sentence_count"] == 1


def test_invalid_parameters_raise():
    texts = ["S1", "S2"]
    units = make_sentence_units(texts)

    embedder = DummyEmbedder(
        {
            "S1": [1.0, 0.0],
            "S2": [0.0, 1.0],
        }
    )

    chunker = TestableSemanticChunker(units, embedder)

    with pytest.raises(ValueError):
        chunker.build_chunks(min_sentences=0)

    with pytest.raises(ValueError):
        chunker.build_chunks(min_sentences=3, max_sentences=2)