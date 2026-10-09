from __future__ import annotations
import argparse, json
from .agent import VisionAgent
from .controls import ActionExecutor
from .dataset import clean_dataset, load_dataset, save_dataset
from .dashboard import create_app
from .from_scratch import evaluate_vision, train_vision
from .hardware import assess, recommendation
from .memory import MemoryStore
from .perception import ScreenObserver
from .training import train_file

def main() -> int:
    parser=argparse.ArgumentParser(description='Vision Agent: local-first computer automation and from-scratch training'); parser.add_argument('--db',default='vision_agent.db'); sub=parser.add_subparsers(dest='command',required=True)
    obs=sub.add_parser('observe'); obs.add_argument('--ocr',action='store_true')
    run=sub.add_parser('run'); run.add_argument('task'); run.add_argument('--execute',action='store_true'); run.add_argument('--observe',action='store_true')
    dash=sub.add_parser('dashboard'); dash.add_argument('--host',default='127.0.0.1'); dash.add_argument('--port',type=int,default=5000)
    train=sub.add_parser('train'); train.add_argument('--dataset',default='data/interface_actions.jsonl'); train.add_argument('--output',default='data/models/intent_model.json')
    tv=sub.add_parser('train-vision'); tv.add_argument('--data',default='data/synthetic_ui'); tv.add_argument('--checkpoint',default='checkpoints/tiny_vision.npz'); tv.add_argument('--report',default='logs/tiny_vision.json'); tv.add_argument('--epochs',type=int,default=25)
    ev=sub.add_parser('evaluate-vision'); ev.add_argument('--data',default='data/synthetic_ui'); ev.add_argument('--checkpoint',default='checkpoints/tiny_vision.npz')
    sub.add_parser('hardware'); validate=sub.add_parser('validate-dataset'); validate.add_argument('dataset'); validate.add_argument('--output')
    args=parser.parse_args()
    if args.command=='observe':
        o=ScreenObserver().capture(args.ocr); print(json.dumps({'image':o.image_path,'size':[o.width,o.height],'elements':[e.__dict__ for e in o.elements]},indent=2))
    elif args.command=='run':
        r=VisionAgent(memory=MemoryStore(args.db),executor=ActionExecutor(dry_run=not args.execute)).run(args.task,args.observe); print(json.dumps({'success':r.success,'message':r.message,'actions':[a.__dict__ for a in r.executed]},indent=2)); return 0 if r.success else 1
    elif args.command=='dashboard': create_app().run(host=args.host,port=args.port,debug=False)
    elif args.command=='train':
        m=train_file(args.dataset,args.output); print(json.dumps({'labels':m.labels,'tokens':sum(m.totals.values()),'output':args.output},indent=2))
    elif args.command=='train-vision': print(json.dumps(train_vision(args.data,args.checkpoint,args.report,epochs=args.epochs),indent=2))
    elif args.command=='evaluate-vision': print(json.dumps(evaluate_vision(args.data,args.checkpoint),indent=2))
    elif args.command=='hardware': print(json.dumps({'hardware':assess(),'recommendation':recommendation(assess())},indent=2))
    elif args.command=='validate-dataset':
        clean=clean_dataset(load_dataset(args.dataset)); save_dataset(args.output,clean) if args.output else None; print(json.dumps({'examples':len(clean),'labels':sorted({e.label for e in clean}),'output':args.output},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
