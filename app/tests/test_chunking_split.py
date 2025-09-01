# test_chunking_split.py

from app.services.chunking import split_text_by_chars

def test_split_text_by_chars_overlap_and_limits():
    text = " ".join([f"word{i}" for i in range(200)])  # simple long text
    parts = split_text_by_chars(text, max_chars=200, overlap=50)
    # Should produce multiple windows with overlap
    assert len(parts) > 1
    # Overlap implies last token of part[i] appears in part[i+1]
    assert parts[0][-20:] in parts[1]
