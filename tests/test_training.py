import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from orchid.io import read_jsonl, sha256_file, write_json
from orchid.model import GPT, GPTConfig
from orchid.runner import train
from orchid.tokenizer import CharacterTokenizer
from orchid.training import TrainConfig, build_adamw, load_checkpoint, optimize, schedule


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_uneven_accumulation_matches_single_batch(self):
        torch.manual_seed(29)
        a = GPT(GPTConfig(11, n_layer=1, d_model=16, n_head=2, context_length=6, dropout=0)).double()
        b = deepcopy(a)
        device = torch.device("cpu")
        cfg = TrainConfig(accumulation=2, grad_clip=10)
        xs = torch.randint(0, 11, (3, 6))
        ys = (xs + 1) % 11
        ys[2, 1:] = -100
        oa, ob = (build_adamw(m, cfg, device) for m in (a, b))
        ra = optimize(a, [(xs[:2], ys[:2]), (xs[2:], ys[2:])], oa, schedule(oa, cfg, 4), cfg, device, "fp32")
        rb = optimize(b, [(xs, ys)], ob, schedule(ob, cfg, 4), cfg, device, "fp32")
        self.assertEqual(ra[1], 13)
        self.assertAlmostEqual(ra[0], rb[0], places=12)
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb, rtol=1e-11, atol=1e-12)

    def test_resume_mid_epoch_matches_uninterrupted_training(self):
        with TemporaryDirectory() as root, redirect_stdout(StringIO()):
            root = Path(root)
            tokens = CharacterTokenizer.fit(["青山雲水月。　"])
            file = root / "tokenizer.json"
            token_sha = tokens.save(file)
            raw, split = root / "data_manifest.json", root / "split_manifest.json"
            write_json(raw, {"test_fixture": True})
            write_json(split, {"test_fixture": True})
            proc = root / "tokens"
            proc.mkdir()
            splits = {}
            for k, n in (("train", 100), ("validation", 21)):
                path = proc / f"{k}.npy"
                text = "青山雲水。月雲" * n
                values = tokens.encode(text, strict=True)[:n]
                np.save(path, np.array(values, dtype=np.int32), allow_pickle=False)
                splits[k] = {"token_file_sha256": sha256_file(path)}
            write_json(root / "token_stats.json", {
                "tokenizer_sha256": token_sha, "data_manifest_sha256": sha256_file(raw),
                "split_manifest_sha256": sha256_file(split), "splits": splits,
            })
            model, data, task, cfg = (root / f"{k}.yaml" for k in ("model", "data", "task", "train"))
            write_json(model, {"n_layer": 1, "d_model": 16, "n_head": 2, "context_length": 6, "dropout": 0.2})
            write_json(data, {"paths": {"processed": str(root), "artifacts": str(root)}})
            write_json(task, {"form": "wujue"})
            write_json(cfg, {
                "model": str(model), "data": str(data), "task": str(task), "seed": 135,
                "training": {"epochs": 2, "micro_batch": 2, "accumulation": 2},
                "runtime": {"device": "cpu", "precision": "fp32", "deterministic": True, "cpu_threads": 2},
            })

            def args(run_id, stop_after=None, resume=None):
                return argparse.Namespace(config=cfg, seed=None, device=None, precision=None, micro_batch=None,
                                          init=None, stop_after=stop_after, resume=resume, run_id=run_id,
                                          checkpoints=root / "checkpoints", runs=root / "runs")

            full = train("pretrain", args("uninterrupted"))
            cut = train("pretrain", args("interrupted", stop_after=3))
            self.assertEqual(cut["cursor"]["step"], 3)
            self.assertGreater(cut["cursor"]["rows"], 0)
            resumed = train("pretrain", args("interrupted", resume=Path(cut["last_checkpoint"])))
            a, b = [load_checkpoint(Path(r["last_checkpoint"])) for r in (full, resumed)]
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual(a["quality"], b["quality"])
            for name in ("epoch", "rows", "step", "target_tokens"):
                self.assertEqual(a["cursor"][name], b["cursor"][name])
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
            for p, state in a["optimizer"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, b["optimizer"]["state"][p][key], rtol=0, atol=0)
            log = list(read_jsonl(root / "runs/interrupted/metrics.jsonl"))
            self.assertEqual([r["step"] for r in log if r["event"] == "train"], list(range(1, a["cursor"]["step"] + 1)))
            self.assertEqual(log[-1]["step"], a["cursor"]["step"])


if __name__ == "__main__":
    unittest.main()
