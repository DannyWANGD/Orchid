"""Prepare blinded poetry rating materials and validate returned forms."""

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.blind import export_reviewer_zips, make_packets, score_reviews, summarize_review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    cmd=p.add_subparsers(dest="action",required=True)
    make=cmd.add_parser("make")
    make.add_argument("--generations",type=Path,required=True)
    make.add_argument("--protocol",type=Path,default=Path("artifacts/eval_prompts.json"))
    make.add_argument("--output",type=Path,required=True)
    make.add_argument("--models",nargs=2,default=["R1","R2-2027"])
    make.add_argument("--rubric-version", choices=("v1", "v2"), default="v1",
                      help="Use v2 for new reviews; v1 reproduces historical packets")
    distribute=cmd.add_parser("distribute")
    distribute.add_argument("--rubric",type=Path,help="Defaults to the guide frozen with the organizer key")
    aggregate=cmd.add_parser("aggregate")
    aggregate.add_argument("--ratings",type=Path,nargs=2,required=True)
    aggregate.add_argument("--reviewer-kinds", nargs=2, choices=("human", "ai"), default=("human", "human"),
                           help="Reviewer types in the same order as --templates and --ratings")
    summarize=cmd.add_parser("summarize")
    summarize.add_argument("--rating", type=Path, required=True)
    summarize.add_argument("--reviewer", type=int, choices=(1, 2), required=True)
    summarize.add_argument("--reviewer-kind", choices=("human", "ai"), required=True)
    for action in (distribute, aggregate, summarize):
        action.add_argument("--generations",type=Path,required=True)
        action.add_argument("--protocol",type=Path,default=Path("artifacts/eval_prompts.json"))
        action.add_argument("--key",type=Path,required=True)
        action.add_argument("--templates",type=Path,nargs=2,required=True)
        action.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try:
        if a.action=="make":
            print(make_packets(a.generations,a.protocol,a.output,a.models,rubric_version=a.rubric_version))
        elif a.action=="distribute":
            print(export_reviewer_zips(a.generations,a.protocol,a.key,a.templates,a.rubric,a.output))
        elif a.action=="summarize":
            result = summarize_review(a.generations, a.protocol, a.key, a.templates, a.rating, a.output,
                                      reviewer=a.reviewer, kind=a.reviewer_kind)
            print(result["means_and_distributions"])
        else:
            results = score_reviews(a.generations, a.protocol, a.key, a.templates, a.ratings, a.output,
                                    reviewer_kinds=a.reviewer_kinds)
            if "means_and_distributions" in results:
                print(results["means_and_distributions"])
            else:
                for item in results["ratings_by_reviewer"]:
                    print(f"Reviewer {item['reviewer']} ({item['kind']}): {item['means_and_distributions']}")
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__=="__main__":
    main()
