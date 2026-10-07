"""One real pretrained VOC micro-batch SGD update; no formal checkpoint or metrics."""
import argparse
import json
import torch
from yolo_baseline.config import ROOT,DATA,WEIGHTS,Config,write_json
from yolo_baseline.data import records,VOCDataset
from yolo_baseline.model import YOLOv1
from yolo_baseline.loss import YOLOLoss
from yolo_baseline.runtime import seed_all,environment
from yolo_baseline.evaluation import benchmark


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0');parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--check-timing',action='store_true');args=parser.parse_args()
    if args.batch<1:parser.error('--batch must be positive')
    seed_all(2026);device=torch.device(args.device)
    cfg=Config();model=YOLOv1(cfg,WEIGHTS).to(device);model.train()
    dataset=VOCDataset(records(DATA,'2007','train')[:args.batch],training=True)
    samples=[dataset[i] for i in range(len(dataset))]
    image=torch.stack([item[0] for item in samples]);target=torch.stack([item[1] for item in samples])
    optimizer=torch.optim.SGD(model.parameters(),lr=cfg.warmup_start,momentum=.9,foreach=False)
    tracked=model.head[-1].weight.detach().clone()
    prediction=model(image.to(device));loss=YOLOLoss()(prediction,target.to(device))
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite smoke loss')
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise FloatingPointError('Nonfinite gradients')
    optimizer.step()
    changed=not torch.equal(tracked,model.head[-1].weight)
    if not changed:raise AssertionError('SGD did not update head weights')
    result={'type':'one micro-batch smoke only, not training metrics','output_shape':list(prediction.shape),
            'loss':float(loss.detach()),'head_updated':changed,'parameters':sum(p.numel() for p in model.parameters()),
            'initialization':model.initialization,'peak_cuda_bytes':torch.cuda.max_memory_allocated(device),
            'environment':environment()}
    if args.check_timing:result['untrained_timing_probe']=benchmark(model,records(DATA,'2007','test')[:16],device,cfg,4)
    write_json(ROOT/'runs/resnet34-smoke.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
