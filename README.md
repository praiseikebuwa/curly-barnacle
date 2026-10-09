# Vision Agent

A practical Python computer-using assistant MVP that combines screenshot capture, OCR, multimodal reasoning, safe action execution, retries, memory, training data, and a live dashboard.

## What is implemented

- **Observe:** screen capture with `mss`/PIL and optional Tesseract OCR.
- **Reason:** optional OpenAI-compatible multimodal planner with structured JSON output and base64 screenshots. It uses `gemini-3-flash-preview` by default for vision and follows model-family token rules.
- **Plan and act:** conservative local planner plus dry-run-first executor for click/type/press/wait.
- **Recover:** exponential-backoff retry runner, action failure boundaries, recovery events, and graceful failed-task results.
- **Remember:** SQLite observations, actions, workflows, and verified outcomes.
- **Monitor:** Flask dashboard at `/` with `/api/events`, `/api/state`, and an SSE endpoint.
- **Learn:** cleaned JSONL interface-action dataset and a reproducible pure-Python multinomial Naive Bayes intent classifier. This is memory/intent learning, not training a giant vision model.

## Install and test

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
python -m pytest -q
```

Desktop support:

```bash
pip install -e '.[desktop]'
vision-agent observe --ocr
```

Optional multimodal reasoning:

```bash
pip install -e '.[llm]'
export OPENAI_API_KEY=...
export OPENAI_API_BASE=...
vision-agent run "search for Python tutorials"
```

The LLM planner is available to Python callers as `MultimodalReasoner`; the default CLI remains local and dry-run-first.

## Dataset and training

```bash
vision-agent validate-dataset data/interface_actions.jsonl --output data/interface_actions.cleaned.jsonl
vision-agent train --dataset data/interface_actions.jsonl --output data/models/intent_model.json
```

The starter dataset is intentionally small and curated. Add verified demonstrations as JSONL rows with `text` and one of the labels `type`, `wait`, `open`, `press`, or `click`. Validate before training.

## Dashboard

```bash
vision-agent dashboard --host 127.0.0.1 --port 5000
```

For a live run, pass the same `EventBus` to `VisionAgent` and `create_app(event_bus)` from Python. The current CLI dashboard is a standalone monitor endpoint; production deployment should use authentication and HTTPS.

## Safety

This is not a fully autonomous system and deliberately refuses to invent coordinates. Keep dry-run mode enabled while developing. Require human review for passwords, payments, account security, deletion, or other irreversible actions.
