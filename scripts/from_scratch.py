from __future__ import annotations
import argparse, json
from vision_agent.hardware import assess, recommendation
from vision_agent.from_scratch import train_vision, evaluate_vision

def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    h=sub.add_parser('hardware'); t=sub.add_parser('train-vision'); t.add_argument('--data',default='data/synthetic_ui'); t.add_argument('--checkpoint',default='checkpoints/tiny_vision.npz'); t.add_argument('--report',default='logs/tiny_vision.json'); t.add_argument('--epochs',type=int,default=25)
    e=sub.add_parser('evaluate-vision'); e.add_argument('--data',default='data/synthetic_ui'); e.add_argument('--checkpoint',default='checkpoints/tiny_vision.npz')
    args=parser.parse_args()
    if args.command=='hardware': info=assess(); print(json.dumps({'hardware':info,'recommendation':recommendation(info)},indent=2))
    elif args.command=='train-vision': print(json.dumps(train_vision(args.data,args.checkpoint,args.report,epochs=args.epochs),indent=2))
    else: print(json.dumps(evaluate_vision(args.data,args.checkpoint),indent=2))
if __name__=='__main__': main()
