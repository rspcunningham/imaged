"""uv run python -m figures {compute,render} {figure,all}"""

import argparse

from figures import apertures, architecture, hero, mtf, optics

FIGURES = {
    module.NAME: module for module in (hero, optics, apertures, architecture, mtf)
}

parser = argparse.ArgumentParser(prog="figures")
parser.add_argument("stage", choices=["compute", "render"])
parser.add_argument("figure", choices=[*FIGURES, "all"])
args = parser.parse_args()

for name in FIGURES if args.figure == "all" else [args.figure]:
    getattr(FIGURES[name], args.stage)()
    print(f"{args.stage} {name}")
