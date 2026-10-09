from __future__ import annotations
import argparse
import uvicorn

def main() -> int:
    parser = argparse.ArgumentParser(description="Run the VisionOS AI FastAPI backend")
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(); uvicorn.run("vision_agent.api:create_api", host=args.host, port=args.port, factory=True); return 0
if __name__ == "__main__": raise SystemExit(main())
