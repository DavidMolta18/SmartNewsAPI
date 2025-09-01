# test_text_utils.py

from app.utils.text_utils import first_clean_sentence
from app.utils.patterns import SNIPPET_NOISE

def test_first_clean_sentence_skips_noise_and_short():
    txt = (
        "Suscríbete a nuestro boletín. "
        "Haga clic aquí. "
        "Esta es la primera oración informativa que debe salir en el snippet. "
        "Luego viene otra oración relevante."
    )
    snip = first_clean_sentence(txt)
    assert "primera oración informativa" in snip
    # Ensure noise patterns would match earlier lines
    assert SNIPPET_NOISE.search("Suscríbete a nuestro boletín")
