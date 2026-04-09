#!/usr/bin/env python3
"""Sampling script for QA review of arxiv pipeline outputs.

Usage:
  python scripts/sample_arxiv.py random    --input stage1/kept -n 100
  python scripts/sample_arxiv.py stratified --input stage2/kept --key metadata.primary_category --per-stratum 10
  python scripts/sample_arxiv.py rule-hit  --input stage2/rejected --rule latex_residual -n 50
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dq.runner.shard import read_shards


def _get_nested(doc: dict, key: str):
    """Get a nested field: 'metadata.primary_category' -> doc['metadata']['primary_category']."""
    parts = key.split(".")
    val = doc
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return None
    return val


def _write(docs: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Wrote {len(docs)} docs to {path}")


def cmd_random(args):
    docs = list(read_shards(Path(args.input)))
    rng = random.Random(args.seed)
    n = min(args.n, len(docs))
    _write(rng.sample(docs, n) if n < len(docs) else docs, args.output)


def cmd_stratified(args):
    groups: dict[str, list[dict]] = defaultdict(list)
    for doc in read_shards(Path(args.input)):
        key = str(_get_nested(doc, args.key) or "unknown")
        groups[key].append(doc)

    rng = random.Random(args.seed)
    sampled = []
    for _, group_docs in sorted(groups.items()):
        n = min(args.per_stratum, len(group_docs))
        sampled.extend(rng.sample(group_docs, n))
    _write(sampled, args.output)


def cmd_rule_hit(args):
    sampled = []
    for doc in read_shards(Path(args.input)):
        for rej in doc.get("__dq_rejections", []):
            if args.rule in rej.get("rule", ""):
                sampled.append(doc)
                break
        if len(sampled) >= args.n:
            break
    _write(sampled, args.output)


def main():
    parser = argparse.ArgumentParser(description="Sample docs from pipeline outputs")
    parser.add_argument("--seed", type=int, default=42)
    sub = parser.add_subparsers(dest="strategy", required=True)

    p = sub.add_parser("random")
    p.add_argument("--input", required=True)
    p.add_argument("-n", type=int, default=100)
    p.add_argument("-o", "--output", default="samples_random.jsonl")

    p = sub.add_parser("stratified")
    p.add_argument("--input", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--per-stratum", type=int, default=10)
    p.add_argument("-o", "--output", default="samples_stratified.jsonl")

    p = sub.add_parser("rule-hit")
    p.add_argument("--input", required=True)
    p.add_argument("--rule", required=True)
    p.add_argument("-n", type=int, default=50)
    p.add_argument("-o", "--output", default="samples_rule_hit.jsonl")

    args = parser.parse_args()
    {
        "random": cmd_random,
        "stratified": cmd_stratified,
        "rule-hit": cmd_rule_hit,
    }[args.strategy](args)


if __name__ == "__main__":
    main()
