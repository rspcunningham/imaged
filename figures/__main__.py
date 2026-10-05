"""uv run python -m figures {compute,render} {figure,all}"""

import argparse

from figures import hero

FIGURES = {module.NAME: module for module in (hero,)}

parser = argparse.ArgumentParser(prog="figures")
parser.add_argument("stage", choices=["compute", "render"])
parser.add_argument("figure", choices=[*FIGURES, "all"])
args = parser.parse_args()

for name in FIGURES if args.figure == "all" else [args.figure]:
    getattr(FIGURES[name], args.stage)()
    print(f"{args.stage} {name}")
