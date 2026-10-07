"""Default training entry: PyTorch YOLOv1-ResNet34, not the archived native run.py."""
import argparse
from pathlib import Path
from yolo_baseline.config import ROOT,DATA,WEIGHTS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=DATA)
    parser.add_argument('--pretrained',type=Path,default=WEIGHTS)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/resnet34')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--micro-batch',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=135)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--split',choices=['full','dev'],default='full')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--eval-batch',type=int,default=4)
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.workers<0 or args.eval_batch<1 or args.iterations<1:parser.error('Invalid workers/evaluation/timing setting')
    from yolo_baseline.training import train
    train(args)


if __name__=='__main__':main()
