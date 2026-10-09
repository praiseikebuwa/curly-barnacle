from pathlib import Path
import numpy as np
from vision_agent.from_scratch import train_vision,evaluate_vision
from vision_agent.model_status import vision_status
from vision_agent.vision_model import DeepVisionCNN

def test_deep_cnn_has_two_convolution_layers():
 m=DeepVisionCNN(seed=1); assert m.w1.shape==(3,3,1,4); assert m.w2.shape==(3,3,4,8); assert m.w3.shape[1]==6

def test_deep_cnn_trains_and_saves(tmp_path:Path):
 r=train_vision(tmp_path/'data',tmp_path/'model.npz',tmp_path/'report.json',epochs=3); assert r['architecture']=='deep_numpy_cnn'; assert Path(r['checkpoint']).exists(); assert 0<=evaluate_vision(tmp_path/'data',tmp_path/'model.npz')['accuracy']<=1

def test_model_status_is_unavailable_without_artifacts(tmp_path:Path):
 assert vision_status(tmp_path/'no.npz',tmp_path/'no.json')['trained'] is False
