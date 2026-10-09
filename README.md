# Vision Agent — local-first from-scratch computer-use platform

Vision Agent is a local-first research platform for learning narrow computer-vision and action capabilities without hosted AI providers or pretrained model weights. It includes screenshot/OCR adapters, dataset tools, safe action execution, task state/recovery, FastAPI APIs, a React dashboard, and a reproducible from-scratch NumPy CNN pipeline.

## From-scratch visual model

The current model is `DeepVisionCNN`: two convolution layers, ReLU activations, max pooling, a dense classifier, random initialization, and manually implemented NumPy backpropagation. It trains on six synthetic UI-like classes: `button`, `text_field`, `panel`, `checkbox`, `menu`, and `window`.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
vision-agent hardware
vision-agent train-vision --epochs 25
vision-agent evaluate-vision
```

Training produces a synthetic manifest under `data/synthetic_ui/`, a checkpoint under `checkpoints/`, and a JSON report under `logs/`. Reports include the architecture, seed, epoch losses, held-out accuracy, duration, and checkpoint path. Generated artifacts are ignored by git. The model is a measurable narrow experiment, not general screen understanding.

## Dashboard and web model demo

Start the backend:

```bash
visionos-api --host 127.0.0.1 --port 8000
```

Start the frontend separately:

```bash
cd frontend
npm install
npm run dev
```

The dashboard reads the real model state from `GET /api/v1/models/vision/status`. Before training it reports `unavailable`; after training it shows the actual architecture, epoch count, and held-out validation accuracy. It never displays fabricated metrics.

## Safe computer control

Dry-run is the default. Real mouse and keyboard operations require both `--execute` and the explicit `--allow-desktop-control` flag, and the executor checks the safety policy before each action:

```bash
vision-agent run "search for Python tutorials"
vision-agent run "search for Python tutorials" --execute --allow-desktop-control
```

Use real control only on a controlled Windows test environment with approved applications. The sandbox cannot honestly validate Windows UI Automation or real input devices. Do not use the system for payments, passwords, account security, deletion, or other consequential actions.

## Automation and skill

GitHub Actions is configured in `.github/workflows/ci.yml`. Every push and pull request runs Python tests, from-scratch model training/evaluation, and the frontend production build. The reusable Manus skill is available at `skills/vision-agent-local-training/SKILL.md` and is also installed at `/home/ubuntu/skills/vision-agent-local-training/`.

## Tests

```bash
python -m pytest -q
cd frontend && npm run build
```

The suite covers the CNN architecture, random initialization, checkpointing, held-out evaluation, API routes, safety policy, action validation, retries, memory, dataset export, secure image ingestion, and dashboard behavior.

## Architecture and limits

The system separates data generation, training, checkpoints, inference, action execution, safety, verification, task state, memory, APIs, and UI. OCR is an optional software adapter and is not presented as a trained project model. Windows UI Automation, Playwright, real desktop-control tests, a text transformer, and a trained OCR recognizer remain future milestones requiring reviewed data and a Windows test environment. No external AI provider is required or configured.
