import tempfile
import unittest
from pathlib import Path

import torch

from orchid.blind import DIMENSIONS, write_csv
from orchid.generation import Sampling
from orchid.inference import Poet
from orchid.metrics import judge_format_keywords
from orchid.model import GPT, GPTConfig
from orchid.supervision import Task
from orchid.tokenizer import CharacterTokenizer
from scripts.create_delivery import blank_template


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

    def test_only_blank_review_snapshots_are_eligible_for_course_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reviewer_2.csv"
            fields = dict(anonymous_id="sample", pair_id="pair", display_order="1",
                          keywords="霜", output="秋霜", notes="", **dict.fromkeys(DIMENSIONS, ""))
            write_csv(path, [fields])
            self.assertEqual(blank_template(path), path.read_bytes())
            write_csv(path, [{**fields, "notes": "review comment"}])
            self.assertIsNone(blank_template(path))
            write_csv(path, [{**fields, "relevance": "3"}])
            self.assertIsNone(blank_template(path))
