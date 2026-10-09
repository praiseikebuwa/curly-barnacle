from pathlib import Path
import numpy as np
from vision_agent.from_scratch import train_vision, evaluate_vision
from vision_agent.hardware import assess
from vision_agent.vision_model import TinyVisionClassifier

def test_random_models_start_differently():
    a=TinyVisionClassifier(seed=1); b=TinyVisionClassifier(seed=2); assert not np.array_equal(a.w1,b.w1)

def test_from_scratch_training_checkpoint_and_eval(tmp_path: Path):
    result=train_vision(tmp_path/'data',tmp_path/'model.npz',tmp_path/'report.json',epochs=8)
    assert result['trained_from_random_initialization'] is True
    assert Path(result['checkpoint']).exists() and Path(result['report']).exists()
    evaluated=evaluate_vision(tmp_path/'data',tmp_path/'model.npz'); assert 0 <= evaluated['accuracy'] <= 1

def test_hardware_assessment_is_real():
    info=assess(); assert info['cpu_count'] and info['os']
