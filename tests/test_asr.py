from app.asr import to_words


def groq(text, words, seg_ends):
  """A Groq verbose_json response: word i starts at i seconds and lasts 0.3 s."""
  return {"text": text, "words": [{"word": w, "start": i, "end": i + 0.3} for i, w in enumerate(words)],
          "segments": [{"end": e} for e in seg_ends]}


def test_words_take_punctuation_and_glued_pieces_merge():
  r = groq(" We don't fly the WB-57, it's 3.5 tons. Right?",
           ["We", "don", "'t", "fly", "the", "WB", "-57", "it", "'s", "3.5", "tons", "Right"], [10.5, 12.0])
  words, seg_ends = to_words(r)
  assert [w.text for w in words] == ["We", "don't", "fly", "the", "WB-57,", "it's", "3.5", "tons.", "Right?"]
  assert (words[1].start, words[1].end) == (1.0, 2.3)  # "don" + "'t" spans both pieces
  assert seg_ends == {7, 8}  # the last word of each segment


def test_starts_never_go_backwards_and_words_last_at_least_20_ms():
  words, _ = to_words({"text": "a b", "words": [{"word": "a", "start": 1.0, "end": 1.0}, {"word": "b", "start": 0.5, "end": 0.5}]})
  assert [(w.start, round(w.end, 2)) for w in words] == [(1.0, 1.02), (1.0, 1.02)]


def test_words_missing_from_the_text_are_kept_as_sent():
  words, _ = to_words({"text": "", "words": [{"word": " hello ", "start": 0, "end": 1}]})
  assert [w.text for w in words] == ["hello"]
