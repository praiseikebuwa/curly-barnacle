---
name: vision-agent-local-training
description: Local-first Vision Agent development with from-scratch NumPy model training, synthetic UI datasets, checkpoint evaluation, safe desktop-control testing, and GitHub CI. Use when extending, training, evaluating, or operating the Vision Agent without hosted AI providers or pretrained model weights.
---

# Vision Agent local training

- Do not add hosted AI APIs, provider SDKs, API keys, remote inference, or pretrained checkpoints.
- Keep desktop control disabled by default. Require consent, bounds checks, action limits, and an emergency stop.
- Report actual dataset size, training state, losses, held-out metrics, checkpoint path, and limitations.
- Run `python -m pytest -q` before and after changes.
- Train with `vision-agent train-vision --epochs 25`; evaluate with `vision-agent evaluate-vision`.
- Use `DeepVisionCNN` only as a narrow six-class synthetic UI experiment, never as general screen understanding.
- Keep raw data, manifests, checkpoints, and reports separate; do not commit generated artifacts.
- Require a regression test for each new safety, training, schema, or API behavior.
- Run the frontend build and API/CLI smoke tests before commit.
- CI must test, train, evaluate, and build on pushes and pull requests.
