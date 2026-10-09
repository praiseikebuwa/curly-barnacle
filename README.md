# Vision Agent — from-scratch local training milestone

This repository is now local-first and does **not** integrate hosted AI providers, external model APIs, pretrained checkpoints, or pretrained OCR/vision models. The current trained capability is deliberately narrow: a small randomly initialized NumPy visual classifier learns three synthetic UI-like classes (`button`, `text_field`, `panel`) and reports reproducible held-out accuracy. Existing screenshot capture, OCR adapter, safety controls, task state, recovery, API, dashboard, dataset, and action infrastructure remain available.

## Train our own first visual model

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
vision-agent-research hardware
vision-agent train-vision --epochs 25
vision-agent evaluate-vision
```

Training creates a synthetic manifest under `data/synthetic_ui/`, a randomly initialized model checkpoint under `checkpoints/`, and a JSON training report under `logs/`. The report includes seed, epoch losses, validation accuracy, duration, and checkpoint path. The model uses NumPy operations and manually implemented gradients; no pretrained weights are loaded.

The synthetic dataset is a scaffold for the first measurable phase, not a claim of general screen understanding. To grow capability, replace or augment it with reviewed screenshots and annotations collected through the dataset tools.

## Existing local platform

Start the local FastAPI backend:

```bash
visionos-api --host 127.0.0.1 --port 8000
```

Start the React dashboard:

```bash
cd frontend && npm install && npm run dev
```

The backend exposes real health, task, screenshot-capture, secure-image-import, and event routes. The development server is localhost-only by default. Desktop input remains disabled until explicit safety policy/Windows adapter integration.

## Tests

```bash
python -m pytest -q
cd frontend && npm run build
```

## Architecture decision

The previous hosted multimodal integration was removed to satisfy the from-scratch requirement. The system now has separate data generation, training, checkpoint, evaluation, inference, action, and control boundaries. A small local classifier is used because it can be trained and measured in this environment; a competitive general-purpose multimodal model would require far more licensed data and compute than an ordinary laptop.

## Current limitations and next milestone

- The first model classifies synthetic UI-like images; it does not yet detect arbitrary real UI elements or read text.
- OCR remains an optional software adapter and is not presented as a model trained from scratch.
- Windows UI Automation, Playwright, and real desktop-control validation require a Windows test environment.
- Next: reviewed screenshot annotation studio, real UI element dataset, trained text-region/OCR component, and action demonstrations with held-out task evaluation.
