from app.library import new_file_id


def test_new_file_id_is_path_safe_and_unused():
  assert new_file_id("../../etc/Pass Wd.MP3", set()) == "pass_wd"  # no directories, no dots, lowercase
  assert new_file_id("Ep 1!.wav", {"ep_1"}) == "ep_1_2"
  assert new_file_id("Ep 1!.wav", {"ep_1", "ep_1_2"}) == "ep_1_3"
  assert new_file_id("....mp3", set()) == "recording"
