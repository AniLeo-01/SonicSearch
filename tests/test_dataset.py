from app.dataset import load_manifest

ENTRIES = load_manifest()

def test_num_files():
  assert 5 <= len(ENTRIES) <= 6

def test_every_speaker_appears_in_one_file():
  names = [name for e in ENTRIES for name in (e.host, e.guest)]
  assert None not in names, "every file needs a host and a guest"
  assert len(set(names)) == len(names) # a speaker appear more than 1 file

