import unittest

from orchid.expansion import HoldoutFilter


class ExpansionTests(unittest.TestCase):
    def test_excludes_holdout_copy_inserted_in_long_poem_and_short_fragment(self):
        parts = "白日依山盡 黃河入海流 欲窮千里目 更上一層樓 風送遠帆去 春色滿江流".split()
        rule = HoldoutFilter([{"lines":parts[:4], "body": "".join(parts[:4])}])
        self.assertEqual(rule.reason(parts, "".join(parts)), "tang_holdout_body_in_song")
        long_only = HoldoutFilter([{"lines":parts, "body":"".join(parts)}])
        self.assertEqual(long_only.reason(parts[:4],"".join(parts[:4])), "body_in_tang_holdout")
        alternate = ["白日依山高",parts[1],"欲登千里目",parts[3]]
        self.assertEqual(rule.reason(alternate,"".join(alternate)), "heldout_edit_distance_leq_2")
        unrelated = ["青嶂連雲起","行人何處歸","日長山路遠","暮鼓送餘暉"]
        self.assertIsNone(rule.reason(unrelated,"".join(unrelated)))
