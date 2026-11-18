from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from orchid.blind import make_packets, score_reviews
from orchid.evaluation import sample_mean_delta
from orchid.io import write_json, write_jsonl
from orchid.metrics import TrainMatches, judge_format_keywords, rhyme_status


class EvalTests(unittest.TestCase):
    def test_rejects_contamination_and_counts_within_sentence(self):
        poem = "白日依山盡，黃河入海流。\n欲窮千里目，更上一層樓。"
        self.assertTrue(judge_format_keywords(poem, ["黃河"])["joint_ok"])
        self.assertFalse(judge_format_keywords(poem, ["盡黃"])["keywords_ok"])
        for invalid in ("題："+poem, poem.replace("，","，，",1), poem+"<pad>", poem.replace("黃","□")):
            self.assertFalse(judge_format_keywords(invalid, ["黃河"])["parsed"])
        self.assertEqual(judge_format_keywords(poem[:-1], ["黃河"])["format_ok"], True)

    def test_rhyme_ambiguity_and_index(self):
        lines = ("白日依山盡","黃河入海流","欲窮千里目","更上一層樓")
        book = {"流": {("ping","尤")}, "樓": {("ping","尤")}, "庚": {("ping","庚")},
                "行": {("ping","庚"),("ze","漾")}}
        self.assertEqual(rhyme_status(lines,book)["rhyme"],"pass")
        self.assertEqual(rhyme_status((*lines[:3],"更上一層庚"),book)["rhyme"],"fail")
        self.assertEqual(rhyme_status((lines[0],"黃河入海行",lines[2],"更上一層庚"),book)["rhyme_cause"],
                         "ambiguous_reading")
        self.assertEqual(rhyme_status((*lines[:3],"更上一層〇"),book)["rhyme_cause"],"missing_form")
        rows=[(lines,"short"),((*lines,"隔山花爛漫"),"long")]
        indexed=TrainMatches(dict(lines=list(ls),body="".join(ls),poem_id=i) for ls,i in rows)
        self.assertEqual(indexed.find(lines)["exact"],"short")
        self.assertEqual(indexed.find(lines)["longer_poem_fragments"],[{"poem_id":"long","start_line":1}])
        altered=("白日依山盡","黃河入海春","欲窮千里日","更上一層樓")
        self.assertEqual(indexed.find(altered)["near"],[{"poem_id":"short","reason":"edit_distance_leq_2"}])

    def test_paired_bootstrap_groups_repeated_seeds(self):
        r=lambda i,s,v:{"prompt_id":i,"sampling_seed":s,"metric":v}
        a=[r("a",1,False),r("a",2,False),r("b",1,True)]
        b=[r("a",1,True),r("a",2,True),r("b",1,False)]
        self.assertEqual(sample_mean_delta(a,b,"metric")["difference"],0)

    def test_blank_review_forms_are_not_scores(self):
        with TemporaryDirectory() as td:
            p=Path(td)
            write_json(p/"prompts.json",{"blind":{"prompt_ids":["M1"],"sample_seed":11}})
            write_jsonl(p/"output.jsonl",
                        [{"suite":"main","prompt_id":"M1","sampling_seed":11,"keywords":["山"],
                          "model":m,"generated":"青山繞碧波。"} for m in ("before","after")])
            folder=p/"blind"
            mock=make_packets(p/"output.jsonl",p/"prompts.json",folder,["before","after"])
            self.assertFalse("before" in (folder/mock[0]).read_text())
            with self.assertRaises(ValueError):
                score_reviews(folder/"private_key.json",[folder/f for f in mock],p/"scores.json")
            self.assertFalse((p/"scores.json").exists())


if __name__=="__main__":
    unittest.main()
