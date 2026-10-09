from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .synthetic_data import generate_dataset, load_dataset, LABELS
from .vision_model import TinyVisionClassifier, save_report

def train_vision(data_dir='data/synthetic_ui', checkpoint='checkpoints/tiny_vision.npz', report='logs/tiny_vision.json', seed=7, epochs=25):
    root=generate_dataset(data_dir, seed=seed); x,y,_=load_dataset(root); rng=np.random.default_rng(seed); order=rng.permutation(len(y)); split=max(1,int(len(y)*.8)); train, val=order[:split],order[split:]; model=TinyVisionClassifier(input_dim=x.shape[1], classes=len(LABELS), seed=seed); result=model.fit(x[train],y[train],x[val],y[val],epochs=epochs,checkpoint=checkpoint); save_report(result, report); return {'labels':LABELS,'samples':len(y),'train_samples':len(train),'validation_samples':len(val),'final_validation_accuracy':result.validation_accuracy[-1],'checkpoint':result.checkpoint,'report':report,'trained_from_random_initialization':True}

def evaluate_vision(data_dir='data/synthetic_ui', checkpoint='checkpoints/tiny_vision.npz'):
    x,y,_=load_dataset(data_dir); model=TinyVisionClassifier.load(checkpoint); return {'samples':len(y),'accuracy':model.loss_accuracy(x,y)[1],'checkpoint':checkpoint,'labels':LABELS}
