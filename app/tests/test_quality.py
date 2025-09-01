# test_quality.py

from app.ingestion.quality import clean_text, is_boilerplate, quality_score, should_index

RAW = """
Aceptar cookies para continuar
POLÍTICA DE COOKIES
Inicio | Buscar
Este es el primer párrafo real de la noticia. Contiene contenido informativo y sustantivo.
Publicidad
Este es el segundo párrafo que también es importante. Tiene más de 25 caracteres.
"""

def test_clean_text_removes_boilerplate_but_keeps_prose():
    cleaned = clean_text(RAW)
    # Should keep informative sentences and drop boilerplate
    assert "primer párrafo real" in cleaned
    assert "POLÍTICA DE COOKIES" not in cleaned
    assert "Publicidad" not in cleaned
    # Should be reasonably compact (no triple newlines)
    assert "\n\n\n" not in cleaned

def test_is_boilerplate_detects_multiple_noise_markers():
    noisy = "Acepta cookies. Ver más. Configura tus preferencias."
    assert is_boilerplate(noisy) is True

def test_quality_score_increases_with_text_and_punctuation():
    short = "texto breve sin señales"
    long = "Este texto tiene oraciones. También nuevas líneas.\n\nMás contenido."
    assert quality_score(long) > quality_score(short)

def test_should_index_thresholds():
    ok, reason = should_index("Un texto muy corto.", min_chars=200, min_score=10_000)
    assert ok is False
    assert "too_short" in reason or "low_score" in reason

    long_text = "Frase válida. " * 200  # long with punctuation
    ok2, reason2 = should_index(long_text, min_chars=200, min_score=500)
    assert ok2 is True
    assert reason2 == "ok"
