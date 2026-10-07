import argparse
import json
from pathlib import Path
import torch
from yolo_baseline.config import ROOT,DATA,Config,write_json,digest
from yolo_baseline.model import YOLOv1,load_checkpoint
from yolo_baseline.data import splits
from yolo_baseline.evaluation import evaluate,benchmark
from yolo_baseline.runtime import seed_all,environment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',type=Path,default=ROOT/'runs/resnet34/last.pt')
    parser.add_argument('--data-root',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/resnet34/re_evaluation')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--iterations',type=int,default=100)
    args=parser.parse_args()
    if args.batch<1 or args.workers<0 or args.iterations<1:parser.error('Invalid evaluation arguments')
    state=load_checkpoint(args.weights);cfg=Config(**state['config']);cfg.validate();seed_all(cfg.seed)
    initialization=state['initialization'];selected_epoch=state['completed_epochs'];source_hashes=state['source_hashes']
    device=torch.device(args.device);model=YOLOv1(cfg).to(device);model.load_state_dict(state['model']);del state
    rows=splits(args.data_root,cfg.split)[2]
    result=evaluate(model,rows,device,cfg,args.output,args.batch,args.workers)
    result.update({'model':cfg.name,'checkpoint':str(args.weights.resolve()),'config':cfg.dictionary(),
                   'checkpoint_sha256':digest(args.weights),'selected_epoch':selected_epoch,
                   'initialization':initialization,'source_hashes':source_hashes,
                   'timing':benchmark(model,rows,device,cfg,args.iterations),'environment':environment()})
    write_json(args.output/'metrics.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
