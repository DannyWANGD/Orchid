import tempfile
import unittest
from pathlib import Path

import torch

from orchid.generation import Sampling
from orchid.inference import Poet
from orchid.metrics import judge_format_keywords
from orchid.model import GPT, GPTConfig
from orchid.prosody_decode import separated_repeats
from orchid.supervision import Task
from orchid.tokenizer import CharacterTokenizer
from scripts.create_delivery import public_files


class ProsodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        vocab = CharacterTokenizer.fit(["白日依山盡，黃河入海流。欲窮千里目，更上一層樓。秋雨孤舟霜葉"])
        model = GPT(GPTConfig(len(vocab), n_layer=1, d_model=16, n_head=2, context_length=40)).eval()
        task = Task("wujue", 4, 5, {"min": 1, "max": 3, "min_chars": 1, "max_chars": 2}, "source_traditional")
        cls.poet = Poet(model, vocab, task, {}, torch.device("cpu"), "", 0)

    def test_constrained_poem_ends_with_different_rhyme_characters(self):
        keywords = ["霜", "霜葉", "孤舟"]
        for temp in (0, .8):
            spec = Sampling(temperature=temp, top_k=1, top_p=.5, max_new_tokens=25)
            result = self.poet.tokens(keywords, spec, 19, constrain_rhyme=True, constrain_keywords=True)
            self.assertEqual(result["stop"], "eos")
            self.assertEqual(result["generated_steps"], 25)
            self.assertTrue(judge_format_keywords(result["generated"], keywords)["joint_ok"])
            self.assertEqual(result["rhyme"], "pass")
            self.assertFalse(result["repeated_rhyme_char"])
            self.assertNotEqual(*result["rhyme_characters"])
            self.assertEqual(len(result["ping_matches"]), 1)

    def test_rejects_short_budget_and_oov(self):
        with self.assertRaisesRegex(ValueError, "at least 25"):
            self.poet.tokens(["秋"], Sampling(max_new_tokens=24), 3, constrain_rhyme=True)
        with self.assertRaisesRegex(ValueError, "outside the training vocabulary"):
            self.poet.tokens(["龘"], Sampling(), 3, constrain_rhyme=True)

    def test_repeat_aware_ranking_preserves_rhyme_and_keywords(self):
        self.assertEqual(separated_repeats(["煙波春渺渺", "春雨夜來雨"]), 1)
        opts = {"constrain_rhyme": True, "constrain_keywords": True, "ranking": "repeat-aware"}
        result = self.poet.tokens(["秋雨", "孤舟"], Sampling(max_new_tokens=25), 11, **opts)
        self.assertTrue(judge_format_keywords(result["generated"], ["秋雨", "孤舟"])["joint_ok"])
        self.assertEqual(result["rhyme"], "pass")
        self.assertFalse(result["repeated_rhyme_char"])

    def test_release_excludes_private_blind_review_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nested = root / "nested"
            nested.mkdir()
            for name in ("reviewer_1.csv", "reviewer_2.csv", "reviewer_1_ai_completed.csv",
                         "private_key.json", "ratings_status.json", "meter_review_completed.md",
                         "cover.local.tex"):
                (nested / name).touch()
            (root / "linked.json").symlink_to(nested / "private_key.json")
            self.assertEqual(list(public_files(root, (".csv", ".json", ".md", ".tex"))),
                             [nested / "ratings_status.json"])
