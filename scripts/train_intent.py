#!/usr/bin/env python3
from __future__ import annotations

import argparse
from vision_agent.training import train_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the lightweight Vision Agent intent classifier")
    parser.add_argument("--dataset", default="data/interface_actions.jsonl")
    parser.add_argument("--output", default="data/models/intent_model.json")
    args = parser.parse_args()
    model = train_file(args.dataset, args.output)
    print(f"trained {len(model.labels)} labels on {sum(model.totals.values())} tokens; saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
