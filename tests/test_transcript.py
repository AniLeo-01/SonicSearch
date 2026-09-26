from app.models import Speaker, Transcript, Utterance, Word
from app.transcript import build_utterances, ends_sentence, infer_roles


def w(text, start, speaker="SPEAKER_00"):
  return Word(text, start, start + 0.3, speaker)


def test_ends_sentence_ignores_abbreviations():
  assert ends_sentence("Right?") and ends_sentence('said."')
  assert not ends_sentence("Dr.") and not ends_sentence("U.S.") and not ends_sentence("and")


def test_utterances_split_on_speaker_change_sentence_end_and_pause():
  words = [w("Hello", 0), w("there", 0.4), w("Hi.", 1, "SPEAKER_01"), w("Yes", 1.5, "SPEAKER_01"), w("again.", 3.5, "SPEAKER_01")]
  utts = build_utterances(words)
  assert [(u.idx, u.speaker, u.text) for u in utts] == [
    (0, "SPEAKER_00", "Hello there"), (1, "SPEAKER_01", "Hi."), (2, "SPEAKER_01", "Yes"), (3, "SPEAKER_01", "again.")]
  assert utts[0].words[1] == ["there", 0.4, 0.7]


def test_long_monologue_splits_at_the_last_comma_in_the_second_half():
  words = [w(f"w{i}," if i == 39 else f"w{i}", i * 0.3) for i in range(60)]
  assert [len(u.words) for u in build_utterances(words)] == [40, 20]


def test_host_asks_the_questions_and_names_attach_when_confident():
  utts = [Utterance(0, "A", 0, 1, "What got you into space?"), Utterance(1, "B", 1, 9, "Well " * 40 + "it started early."),
          Utterance(2, "A", 9, 10, "And then?"), Utterance(3, "B", 10, 20, "Then " * 40 + "I joined.")]
  host, guest = infer_roles(utts, "Hana Host", "Gus Guest")
  assert (host.label, host.role, host.name) == ("A", "host", "Hana Host")
  assert (guest.label, guest.role, guest.name) == ("B", "guest", "Gus Guest")


def test_no_names_when_the_roles_are_unclear():
  utts = [Utterance(0, "A", 0, 1, "one two three"), Utterance(1, "B", 1, 2, "one two")]
  speakers = infer_roles(utts, "Hana Host", "Gus Guest")
  assert [s.role for s in speakers] == ["host", "guest"] and [s.name for s in speakers] == [None, None]


def test_transcript_round_trips_through_json(tmp_path):
  t = Transcript("f", 12.5, [Utterance(0, "A", 0.0, 1.0, "Hi.", [["Hi.", 0.0, 1.0]])], [Speaker("A", "host", "Hana")])
  t.save(tmp_path / "t" / "f.json")
  assert Transcript.load(tmp_path / "t" / "f.json") == t
