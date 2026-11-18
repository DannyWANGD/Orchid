"""Export blinded poetry rating forms or aggregate two completed forms."""

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.blind import make_packets, score_reviews


def main():
    p=argparse.ArgumentParser(description=__doc__)
    cmd=p.add_subparsers(dest="action",required=True)
    make=cmd.add_parser("make")
    make.add_argument("--generations",type=Path,required=True)
    make.add_argument("--protocol",type=Path,default=Path("artifacts/eval_prompts.json"))
    make.add_argument("--output",type=Path,required=True)
    make.add_argument("--models",nargs=2,default=["R1","R2-2027"])
    aggregate=cmd.add_parser("aggregate")
    aggregate.add_argument("--key",type=Path,required=True)
    aggregate.add_argument("--ratings",type=Path,nargs=2,required=True)
    aggregate.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.action=="make":
        print(make_packets(a.generations,a.protocol,a.output,a.models))
    else:
        print(score_reviews(a.key,a.ratings,a.output)["means_and_distributions"])


if __name__=="__main__":
    main()
