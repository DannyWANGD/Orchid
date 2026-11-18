"""Prepare blinded poetry rating materials or aggregate two returned forms."""

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.blind import export_reviewer_zips, make_packets, score_reviews


def main():
    p=argparse.ArgumentParser(description=__doc__)
    cmd=p.add_subparsers(dest="action",required=True)
    make=cmd.add_parser("make")
    make.add_argument("--generations",type=Path,required=True)
    make.add_argument("--protocol",type=Path,default=Path("artifacts/eval_prompts.json"))
    make.add_argument("--output",type=Path,required=True)
    make.add_argument("--models",nargs=2,default=["R1","R2-2027"])
    distribute=cmd.add_parser("distribute")
    distribute.add_argument("--rubric",type=Path,default=Path("report/blind_review/reviewer_guide.md"))
    aggregate=cmd.add_parser("aggregate")
    aggregate.add_argument("--ratings",type=Path,nargs=2,required=True)
    for action in (distribute, aggregate):
        action.add_argument("--generations",type=Path,required=True)
        action.add_argument("--protocol",type=Path,default=Path("artifacts/eval_prompts.json"))
        action.add_argument("--key",type=Path,required=True)
        action.add_argument("--templates",type=Path,nargs=2,required=True)
        action.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try:
        if a.action=="make":
            print(make_packets(a.generations,a.protocol,a.output,a.models))
        elif a.action=="distribute":
            print(export_reviewer_zips(a.generations,a.protocol,a.key,a.templates,a.rubric,a.output))
        else:
            print(score_reviews(a.generations,a.protocol,a.key,a.templates,a.ratings,a.output)
                  ["means_and_distributions"])
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__=="__main__":
    main()
