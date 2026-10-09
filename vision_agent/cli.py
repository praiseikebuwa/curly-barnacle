from __future__ import annotations

import argparse
import json

from .agent import VisionAgent
from .controls import ActionExecutor
from .dataset import clean_dataset, load_dataset, save_dataset
from .dashboard import create_app
from .llm import MultimodalReasoner
from .memory import MemoryStore
from .perception import ScreenObserver
from .training import train_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Vision Agent: screenshot/OCR desktop automation with recovery and training")
    parser.add_argument("--db", default="vision_agent.db", help="SQLite memory database")
    sub = parser.add_subparsers(dest="command", required=True)
    obs = sub.add_parser("observe", help="Capture a screenshot")
    obs.add_argument("--ocr", action="store_true", help="Attempt OCR")
    run = sub.add_parser("run", help="Plan and execute a task")
    run.add_argument("task")
    run.add_argument("--execute", action="store_true", help="Actually control the mouse or keyboard (default is dry-run)")
    run.add_argument("--observe", action="store_true", help="Capture a screenshot before planning")
    run.add_argument("--multimodal", action="store_true", help="Use the configured OpenAI-compatible vision model")
    dash = sub.add_parser("dashboard", help="Run the live monitoring dashboard")
    dash.add_argument("--host", default="127.0.0.1")
    dash.add_argument("--port", type=int, default=5000)
    train = sub.add_parser("train", help="Clean and train the intent classifier")
    train.add_argument("--dataset", default="data/interface_actions.jsonl")
    train.add_argument("--output", default="data/models/intent_model.json")
    validate = sub.add_parser("validate-dataset", help="Validate and deduplicate a JSONL dataset")
    validate.add_argument("dataset")
    validate.add_argument("--output")
    args = parser.parse_args()
    if args.command == "observe":
        observation = ScreenObserver().capture(args.ocr)
        print(json.dumps({"image": observation.image_path, "size": [observation.width, observation.height], "elements": [e.__dict__ for e in observation.elements]}, indent=2))
    elif args.command == "run":
        reasoner = MultimodalReasoner() if args.multimodal else None
        agent = VisionAgent(memory=MemoryStore(args.db), executor=ActionExecutor(dry_run=not args.execute), reasoner=reasoner)
        result = agent.run(args.task, observe=args.observe or args.multimodal)
        print(json.dumps({"success": result.success, "message": result.message, "actions": [a.__dict__ for a in result.executed]}, indent=2))
        return 0 if result.success else 1
    elif args.command == "dashboard":
        create_app().run(host=args.host, port=args.port, debug=False)
    elif args.command == "train":
        model = train_file(args.dataset, args.output)
        print(json.dumps({"labels": model.labels, "tokens": sum(model.totals.values()), "output": args.output}, indent=2))
    elif args.command == "validate-dataset":
        clean = clean_dataset(load_dataset(args.dataset))
        if args.output:
            save_dataset(args.output, clean)
        print(json.dumps({"examples": len(clean), "labels": sorted({e.label for e in clean}), "output": args.output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
