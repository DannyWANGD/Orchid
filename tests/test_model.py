from dataclasses import replace
from io import BytesIO
import math
import unittest

import torch

from orchid.model import GPT, GPTConfig, cross_entropy_sum


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(123)
        self.cfg = GPTConfig(23, n_layer=2, d_model=32, n_head=4, context_length=12)
        self.model = GPT(self.cfg).eval()

    def test_future_changes_do_not_change_prefix_and_backends_agree(self):
        x = torch.randint(23, (2, 8))
        change = x.clone()
        change[:, 4:] = (x[:, 4:] + 1) % 23
        manual = GPT(replace(self.cfg, attention="manual")).eval()
        manual.load_state_dict(self.model.state_dict())
        keep = torch.tensor([[False, True, False, True, True, True, True, False],
                             [True, True, True, True, True, False, False, False]])
        with torch.inference_mode():
            self.assertEqual(self.model(x).shape, (2, 8, 23))
            torch.testing.assert_close(self.model(x)[:, :4], self.model(change)[:, :4], rtol=0, atol=0)
            torch.testing.assert_close(self.model(x, valid_keys=keep), manual(x, valid_keys=keep),
                                       rtol=1e-5, atol=1e-7)
            replacement = x.masked_fill(~keep, 0)
            torch.testing.assert_close(self.model(x, valid_keys=keep)[keep],
                                       self.model(replacement, valid_keys=keep)[keep], rtol=0, atol=0)
            torch.testing.assert_close(self.model(x, last_only=True), self.model(x)[:, -1:])
            with self.assertRaises(ValueError):
                self.model(torch.zeros((1, 13), dtype=torch.long))

    def test_tied_parameter_and_state_dict_roundtrip(self):
        self.assertIs(self.model.head.weight, self.model.token_embedding.weight)
        self.assertEqual(self.model.parameter_count(), self.cfg.expected_parameters())
        state = BytesIO()
        torch.save(self.model.state_dict(), state)
        state.seek(0)
        restored = GPT(self.cfg).eval()
        restored.load_state_dict(torch.load(state, weights_only=True))
        ids = torch.randint(23, (1, 6))
        torch.testing.assert_close(self.model(ids), restored(ids), rtol=0, atol=0)

    def test_sum_of_manual_nlls_ignores_padding(self):
        logits = torch.tensor([[[math.log(2), 0.0], [0.0, 0.0], [0.0, math.log(3)]]],
                              dtype=torch.float64, requires_grad=True)
        loss = cross_entropy_sum(logits, torch.tensor([[0, -100, 0]]))
        self.assertAlmostEqual(loss.item(), -math.log(2/3) - math.log(1/4), places=12)
        loss.backward()
        self.assertTrue((logits.grad[0, 1] == 0).all())


if __name__ == "__main__":
    unittest.main()
