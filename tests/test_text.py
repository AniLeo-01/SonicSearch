from app.text import content_tokens, tokens


def test_tokens_lowercase_and_drop_apostrophes():
  assert tokens("We don’t fly NASA's WB-57.") == ["we", "dont", "fly", "nasa", "wb", "57"]


def test_content_tokens_drop_stop_words_short_tokens_and_numbers():
  assert content_tokens("We don’t fly NASA's WB-57 to El Paso, uh, really.") == ["dont", "fly", "nasa", "paso"]
