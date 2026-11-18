import unittest

import numpy as np

from orchid.corpus import assign_splits, canonical_poem, duplicate_edges, is_wujue, source_lines
from orchid.data import IGNORE, TokenWindows, collate_responses, prompt_ids, response_item
from orchid.tokenizer import BOS, EOS, PAD, UNK, CharacterTokenizer


class DataTests(unittest.TestCase):
    def test_split_connects_edit_distance_and_embedded_poem(self):
        base, _ = source_lines(["白日依山盡，黃河入海流。", "欲窮千里目，更上一層樓。"])
        variant = (*base[:3], "又上高臺望")
        displaced = ("日依山盡黃", "河入海流白", *base[2:])
        other = (*base, "遠山含翠色")
        rows = [
            dict(lines=ls, body="".join(ls), form="wujue" if is_wujue(ls) else "other",
                 poem_id=str(i), body_sha256=str(i))
            for i, ls in enumerate((base, variant, displaced, other))
        ]
        edges = list(duplicate_edges(rows))
        self.assertIn((0, 1, "three_aligned_lines"), edges)
        self.assertIn((0, 3, "embedded_four_lines"), edges)
        self.assertIn((0, 2, "edit_distance_leq_2"), edges)
        assign_splits(rows, edges, {"train": .5, "validation": .25, "test": .25}, 2026)
        self.assertEqual({r["group_id"] for r in rows}, {"0"})
        self.assertEqual(len({r["split"] for r in rows}), 1)
        self.assertEqual(source_lines(["白□依山盡"])[1], "missing_glyph")

    def test_token_boundaries_unknown_and_tail(self):
        text = "青山。"
        tok = CharacterTokenizer.fit([text])
        self.assertEqual(tok.decode(tok.encode(text)), text)
        self.assertEqual(tok.encode("霜"), [UNK])
        self.assertEqual(tok.encode("e\u0301"), [UNK])
        with self.assertRaises(ValueError):
            tok.encode("<eos>")
        sequence = [BOS, *tok.encode(text), EOS, BOS, *tok.encode(text), EOS]
        ds = TokenWindows(np.array(sequence, dtype=np.int32), 4)
        self.assertEqual(len(ds), 3)
        self.assertEqual([y for _, row in ds for y in row.tolist() if y != IGNORE], sequence[1:])
        x, y = ds[2]
        self.assertEqual(x.tolist()[1:], [PAD, PAD, PAD])
        self.assertEqual(y.tolist(), [EOS, IGNORE, IGNORE, IGNORE])

    def test_response_mask_preserves_first_char_and_eos(self):
        ls = "白日依山盡 黃河入海流 欲窮千里目 更上一層樓".split()
        poem = canonical_poem(ls)
        tok = CharacterTokenizer.fit([poem])
        items = [response_item(tok, kws, poem, strict=True) for kws in (["山"], ["山", "黃河"])]
        x, y = collate_responses(items)
        for row, kws in zip(y, (["山"], ["山", "黃河"])):
            p = len(prompt_ids(tok, kws))
            self.assertTrue((row[:p - 1] == IGNORE).all())
            self.assertEqual(row[p - 1].item(), tok.ids["白"])
            self.assertEqual(row[row != IGNORE].tolist(), [*tok.encode(poem), EOS])
        self.assertEqual(x[0, -1].item(), PAD)


if __name__ == "__main__":
    unittest.main()
