"""
training/datagen.py

    python -m training.datagen --config configs/gpt2_small.yaml
    python -m training.datagen --config configs/gpt2_small.yaml --stage explain
    python -m training.datagen --set data.num_samples=64 --set data.output_dir=data/smoke
"""

import argparse

from dotenv import load_dotenv

from nla.datagen import build
from nla.utils import load_config, parse_overrides, resolve_device, set_seed, utf8_stdio

load_dotenv()


def main():
    utf8_stdio()
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/gpt2_small.yaml")
    p.add_argument("--stage", choices=["extract", "explain", "all"], default="all")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    set_seed(cfg["experiment"]["seed"], deterministic=False)
    build(cfg, resolve_device(cfg), stage=args.stage)


if __name__ == "__main__":
    main()
