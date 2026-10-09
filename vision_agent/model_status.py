from __future__ import annotations
import json
from pathlib import Path

def vision_status(checkpoint='checkpoints/tiny_vision.npz',report='logs/tiny_vision.json'):
 checkpoint,report=Path(checkpoint),Path(report)
 if not checkpoint.exists() or not report.exists(): return {'trained':False,'status':'unavailable','reason':'No local checkpoint and report found','checkpoint':str(checkpoint),'report':str(report)}
 data=json.loads(report.read_text(encoding='utf-8'));return {'trained':True,'status':'trained','architecture':data.get('architecture'),'seed':data.get('seed'),'epochs':data.get('epochs'),'validation_accuracy':data.get('validation_accuracy',[-1])[-1],'checkpoint':str(checkpoint),'report':str(report),'limitations':'Synthetic UI classes only; no claim of general screen understanding.'}
