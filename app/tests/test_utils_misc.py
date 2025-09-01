# test_utils_misc.py

import numpy as np
import pytest

from app.utils.misc import to_epoch, is_429_error, embed_in_batches

def test_to_epoch_valid_basic():
    # ISO with timezone Z should parse to epoch int
    assert isinstance(to_epoch("2025-08-31T20:20:00Z"), int)

def test_to_epoch_valid_offset():
    # ISO with explicit offset should also parse
    assert isinstance(to_epoch("2025-08-31T20:20:00+00:00"), int)

def test_to_epoch_none_and_malformed():
    assert to_epoch(None) is None
    assert to_epoch("not-a-timestamp") is None

@pytest.mark.parametrize("msg,expected", [
    ("429", True),
    ("Resource exhausted", True),
    ("quota exceeded", True),
    ("random error", False),
])
def test_is_429_error(msg, expected):
    e = RuntimeError(msg)
    assert is_429_error(e) is expected

def test_embed_in_batches_happy_path():
    # No failures: should return stacked array of correct shape
    from app.tests.conftest import FakeEmbeddingProvider
    provider = FakeEmbeddingProvider(dim=16, fail_first_n=0)
    texts = [f"t{i}" for i in range(7)]
    arr = embed_in_batches(texts, provider, batch_size=3, max_retries=2)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (7, 16)
    # Ensure provider called floor(7/3)=2 + 1 = 3 times
    assert provider.calls == 3

def test_embed_in_batches_with_429_retries_then_success():
    # Fail first batch once with 429, then succeed
    from app.tests.conftest import FakeEmbeddingProvider
    provider = FakeEmbeddingProvider(dim=8, fail_first_n=1)
    arr = embed_in_batches(["a", "b", "c", "d"], provider, batch_size=2, max_retries=3)
    assert arr.shape == (4, 8)
    # Should result in an extra call due to retry
    assert provider.calls >= 3

def test_embed_in_batches_non_429_raises():
    from app.tests.conftest import FakeEmbeddingProvider

    class Failer(FakeEmbeddingProvider):
        def embed_texts(self, texts):
            raise RuntimeError("some other error")

    provider = Failer()
    with pytest.raises(RuntimeError):
        embed_in_batches(["x", "y"], provider)

def test_embed_in_batches_empty_texts_returns_zero_rows():
    from app.tests.conftest import FakeEmbeddingProvider
    provider = FakeEmbeddingProvider(dim=12)
    arr = embed_in_batches([], provider)
    assert arr.shape == (0, 12)
