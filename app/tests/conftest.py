import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_embedding_provider, get_vector_index

# ------------------------
# Fake Embedding Provider
# ------------------------
class FakeEmbeddingProvider:
    """Fake embedding provider for testing batch + retry logic."""

    requires_e5_prefix = False

    def __init__(self, dim=8, fail_first_n=0):
        self.dim = dim
        self.fail_first_n = fail_first_n
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        # Simulate transient 429 errors
        if self.fail_first_n > 0:
            self.fail_first_n -= 1
            raise RuntimeError("429 Resource exhausted")
        arr = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=self.dim).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            arr.append(v)
        return np.vstack(arr)


# ------------------------
# Fake Vector Index
# ------------------------
class FakeVectorIndex:
    """Minimal in-memory vector index."""

    def __init__(self):
        self.points = []
        self.payloads = {}

    def ensure_collection(self, dim: int) -> None:
        return None

    def upsert(self, ids, vectors, payloads):
        for pid, v, pl in zip(ids, vectors, payloads):
            self.points.append({"id": pid, "vector": np.array(v), "payload": pl})
            self.payloads[pid] = pl

    def search(self, vector, top_k: int, qfilter=None, hnsw_ef=None):
        res = []
        for p in self.points:
            v = p["vector"]
            score = float(np.dot(v, vector) / (np.linalg.norm(v) * np.linalg.norm(vector) + 1e-12))
            res.append({"id": p["id"], "score": score, "payload": p["payload"]})
        res.sort(key=lambda x: x["score"], reverse=True)
        return res[:top_k]


# ------------------------
# Dependency Overrides
# ------------------------
_fake_provider = FakeEmbeddingProvider()
_fake_index = FakeVectorIndex()

app.dependency_overrides[get_embedding_provider] = lambda: _fake_provider
app.dependency_overrides[get_vector_index] = lambda: _fake_index

@pytest.fixture
def client_fixture():
    return TestClient(app)

@pytest.fixture(autouse=True)
def _reset_fake_index():
    _fake_index.points.clear()
    _fake_index.payloads.clear()
    yield
