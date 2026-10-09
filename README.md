# VisionOS AI

VisionOS AI is an extensible, local-first Python computer-use platform foundation. It separates API, orchestration, observation, perception, model providers, action safety, verification data, memory, datasets, recovery, and the dashboard instead of treating automation as a screenshot-and-click script.

## Implemented platform foundation

- **FastAPI backend** with Pydantic schemas and versioned routes under `/api/v1`.
- **React + TypeScript + Vite dashboard** showing real backend health and persisted in-process task records.
- **Canonical interface-element schema** with versioning, bounding-box convention, source tracking, nullable evidence, and validation.
- **Typed action schema** that rejects arbitrary code/shell parameters.
- **Explicit task state machine** with legal transitions and cancellation.
- **Restrictive safety policy** with desktop-control consent, action limits, HTTPS domain allowlists, and emergency stop.
- **Observation API** that captures only after explicit request and persists metadata through SQLite memory.
- **Secure image ingestion** for local images and HTTPS URLs with size limits, DNS/IP checks, redirect revalidation, content-type checks, generated filenames, and format verification.
- **Dataset lab primitives** for annotation validation, deterministic train/validation/test splits, and COCO export.
- **Provider-neutral model gateway** with capability registration and actual health checks.
- **Verified experience memory** that rejects unverified sources and supports search/staleness.
- Existing OCR, screenshot, multimodal reasoning, recovery, dry-run action executor, SQLite workflow memory, and intent training from the previous milestones.

## Start it

Backend:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
visionos-api --host 127.0.0.1 --port 8000
```

Frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL` if the backend is not on `http://127.0.0.1:8000`. The frontend defaults to a local-only backend and does not contain provider secrets.

Useful routes:

- `GET /api/v1/health`
- `POST /api/v1/tasks`
- `GET /api/v1/tasks`
- `POST /api/v1/tasks/{task_id}/cancel`
- `POST /api/v1/observations/capture?ocr=true`
- `POST /api/v1/images/url` with `{ "url": "https://..." }`
- `GET /api/v1/events`

## Tests and training

```bash
python -m pytest -q
python -m vision_agent.cli train --dataset data/interface_actions.jsonl --output /tmp/intent_model.json
cd frontend && npm run build
```

The test suite covers schema validation, state transitions, safety policy, recovery, OCR/multimodal payload construction, API lifecycle, secure ingestion rejection, dataset splitting/COCO export, memory, and dashboard behavior.

## Optional integrations

- Desktop: `pip install -e '.[desktop]'`; Windows UI Automation and Playwright are designed as adapters but are not enabled by default.
- Hosted multimodal model: `pip install -e '.[llm]'` and configure `OPENAI_API_KEY` plus `OPENAI_API_BASE` server-side. The health endpoint reports availability only when credentials are present; it does not claim inference health without a provider check.
- Tesseract must be installed separately for OCR on Windows (for example, install Tesseract and configure `pytesseract.pytesseract.tesseract_cmd`).

## Security and limitations

The backend defaults to localhost. No authentication is enabled in this development scaffold, so do not expose it publicly. Screenshots and imported images remain local and should be protected by the host filesystem. Real desktop control is disabled by default and requires an explicit safety-policy instance; consequential actions need human approval. The Linux sandbox cannot honestly validate Windows UI Automation, real mouse/keyboard input, or a Windows screenshot device, so those remain guarded adapters rather than fabricated support.

The React dashboard is a first milestone shell, not a finished production control plane. PostgreSQL/Alembic, authenticated multi-user roles, browser DOM adapters, and full annotation-editor UX are intentionally staged for later phases; the API/data interfaces are separated so they can be added without replacing the core.
