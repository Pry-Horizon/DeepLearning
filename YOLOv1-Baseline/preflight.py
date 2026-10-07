"""Read-only shared-environment and VOC/pretrained checks, without GPU allocation."""
import argparse
import importlib
import json
from pathlib import Path
from yolo_baseline.config import DATA,WEIGHTS,digest,PRETRAINED_SHA256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,default=DATA)
    parser.add_argument('--pretrained',type=Path,default=WEIGHTS);args=parser.parse_args()
    issues=[];result={}
    for name in ('torch','torchvision','numpy','PIL'):
        try:result[name]=importlib.import_module(name).__version__
        except Exception as exc:issues.append(f'{name}: {exc}')
    minimum={'torch':(1,13),'torchvision':(0,14),'numpy':(1,24),'PIL':(9,1)}
    import re
    for name,required in minimum.items():
        if name in result:
            version=tuple(map(int,re.findall(r'\d+',str(result[name]))[:2]))
            if version<required:issues.append(f'{name} {result[name]} is below required {required}')
    if not issues:
        import torch
        try:
            from torchvision.ops import nms
            nms(torch.tensor([[0.,0.,1.,1.]]),torch.tensor([.9]),.5)
            from yolo_baseline.data import splits
            rows=splits(args.data_root);result['data']={name:len(value) for name,value in zip(('train','val','test'),rows)}
            result['cuda_available']=torch.cuda.is_available()
            if not result['cuda_available']:issues.append('CUDA runtime/device unavailable')
        except Exception as exc:issues.append(str(exc))
    try:
        fingerprint=digest(args.pretrained)
        if fingerprint!=PRETRAINED_SHA256:raise ValueError('Wrong pretrained weight hash')
        result['pretrained_sha256']=fingerprint
    except Exception as exc:issues.append(f'pretrained: {exc}')
    result.update({'ready':not issues,'issues':issues})
    print(json.dumps(result,indent=2));raise SystemExit(1 if issues else 0)


if __name__=='__main__':main()
