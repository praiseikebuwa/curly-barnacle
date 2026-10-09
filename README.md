# Vision Agent

A practical Python MVP for a computer-using assistant. It combines screenshot capture, OCR, visual-change verification, safe action execution, and SQLite memory. The design is intentionally adapter-based: desktop integrations are optional, while planning and tests work in headless environments.

## Features

- **Observe:** capture the screen and extract text regions with OCR when available.
- **Plan:** turn a small set of natural-language intents into explicit, reviewable actions.
- **Act safely:** click/type/keypress/wait through an executor with dry-run mode and an allowlist of action kinds.
- **Verify:** compare before/after screenshots using a perceptual difference score.
- **Remember:** persist observations, actions, outcomes, and verified workflows in SQLite.
- **Windows-friendly:** `mss`, `pyautogui`, and Tesseract are optional dependencies; Windows UI Automation can be added as a perception adapter.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
python -m vision_agent.cli --help
python -m pytest
```

For desktop capture and control:

```bash
pip install -e '.[desktop]'
vision-agent observe --ocr
vision-agent run "open the browser and search for Python tutorials" --dry-run
```

Without `--dry-run`, the agent will request an explicit confirmation before executing a plan. Start with harmless tasks and keep the emergency stop behavior of your OS available. Never use it for passwords, payments, account security, or irreversible actions without human review.

## Architecture

`ScreenObserver` -> `Planner` -> `ActionExecutor` -> `ResultVerifier`, with `MemoryStore` alongside the loop. `vision_agent.adapters` are small protocols, making it straightforward to plug in a vision-language model or Windows UI Automation later.

This is an MVP, not an autonomous system that understands every application. The built-in planner deliberately produces conservative plans for a few common intents and refuses to guess coordinates.
