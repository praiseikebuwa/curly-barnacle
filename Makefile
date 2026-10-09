.PHONY: test api frontend train

test:
	python -m pytest -q

api:
	visionos-api

frontend:
	cd frontend && npm install && npm run dev

train:
	vision-agent train --dataset data/interface_actions.jsonl --output data/models/intent_model.json
