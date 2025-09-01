# test_embeddings_helpers.py

from app.services.embeddings import as_passages, as_queries

def test_as_passages_with_prefix_flag():
    texts = ["alpha", "beta"]
    out_pref = as_passages(texts, prefix_required=True)
    assert out_pref[0].startswith("passage: ")
    out_raw = as_passages(texts, prefix_required=False)
    assert out_raw[0] == "alpha"

def test_as_queries_with_prefix_flag():
    texts = ["who is x", "y z"]
    out_pref = as_queries(texts, prefix_required=True)
    assert out_pref[1].startswith("query: ")
    out_raw = as_queries(texts, prefix_required=False)
    assert out_raw[0] == "who is x"
