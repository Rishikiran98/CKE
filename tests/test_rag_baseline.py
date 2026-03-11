import numpy as np

from cke.retrieval.faiss_index import FaissIndex
from cke.retrieval.rag_baseline import RAGRetriever


class DummyEmbeddingModel:
    def embed_text(self, text: str):
        return self.embed_texts([text])[0]

    def embed_texts(self, texts, batch_size: int = 32):
        vectors = []
        for text in texts:
            t = text.lower()
            vectors.append(
                [
                    1.0 if "redis" in t else 0.0,
                    1.0 if "python" in t else 0.0,
                    1.0 if "resp" in t else 0.0,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


def test_faiss_index_retrieves_relevant_document():
    index = FaissIndex(dimension=3)
    docs = [
        {
            "doc_id": "1",
            "text": "Redis uses RESP",
            "embedding": np.array([1, 0, 1], dtype=np.float32),
        },
        {
            "doc_id": "2",
            "text": "Python is dynamic",
            "embedding": np.array([0, 1, 0], dtype=np.float32),
        },
    ]
    index.build_index(docs)
    hits = index.search(np.array([1, 0, 1], dtype=np.float32), k=1)
    assert hits[0]["doc_id"] == "1"


def test_rag_retriever_returns_documents():
    retriever = RAGRetriever(embedding_model=DummyEmbeddingModel())
    retriever.build_index(
        [
            {"doc_id": "1", "text": "Redis uses RESP"},
            {"doc_id": "2", "text": "Python is dynamic"},
        ]
    )
    results = retriever.retrieve("What does Redis use?", k=1)
    assert len(results) == 1
    assert results[0]["doc_id"] == "1"
