from __future__ import annotations

import argparse
import json

from .agent import VisionAgent
from .controls import ActionExecutor
from .memory import MemoryStore
from .perception import ScreenObserver


def main() -> int:
    parser = argparse.ArgumentParser(description="Vision Agent: safe screenshot/OCR desktop automation MVP")
    parser.add_argument("--db", default="vision_agent.db", help="SQLite memory database")
    sub = parser.add_subparsers(dest="command", required=True)
    obs = sub.add_parser("observe", help="Capture a screenshot")
    obs.add_argument("--ocr", action="store_true", help="Attempt OCR")
    run = sub.add_parser("run", help="Plan and execute a task")
    run.add_argument("task")
    run.add_argument("--execute", action="store_true", help="Actually control the mouse or keyboard (default is dry-run)")
    args = parser.parse_args()
    if args.command == "observe":
        observation = ScreenObserver().capture(args.ocr)
        print(json.dumps({"image": observation.image_path, "size": [observation.width, observation.height], "elements": [e.__dict__ for e in observation.elements]}, indent=2))
        return 0
    agent = VisionAgent(memory=MemoryStore(args.db), executor=ActionExecutor(dry_run=not args.execute))
    result = agent.run(args.task)
    print(json.dumps({"success": result.success, "message": result.message, "actions": [a.__dict__ for a in result.executed]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
