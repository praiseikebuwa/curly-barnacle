#!/usr/bin/env python3
from pathlib import Path
from tempfile import TemporaryDirectory
from vision_agent.from_scratch import evaluate_vision, train_vision
with TemporaryDirectory(prefix='vision-agent-smoke-') as tmp:
    root=Path(tmp); print(train_vision(root/'data',root/'model.npz',root/'report.json',epochs=3)); print(evaluate_vision(root/'data',root/'model.npz'))
