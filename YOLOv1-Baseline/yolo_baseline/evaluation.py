import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from torch.utils.data import DataLoader
from .config import CLASSES,write_json
from .data import VOCDataset,preprocess
from .model import unpack,box_corners


def decode(prediction,threshold=.001,nms_iou=.5):
    if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite detector predictions')
    classes,confidence,boxes=unpack(prediction)
    scores=(classes.clamp(0,1).unsqueeze(-2)*confidence.clamp(0,1).unsqueeze(-1)).reshape(len(prediction),98,20)
    corners=box_corners(boxes).clamp(0,1).reshape(len(prediction),98,4)
    result=[]
    for image_boxes,image_scores in zip(corners.cpu(),scores.cpu()):
        image=[]
        valid=(image_boxes[:,2:]>image_boxes[:,:2]).all(-1)
        for cls in range(20):
            keep=(image_scores[:,cls]>=threshold)&valid
            b,s=image_boxes[keep],image_scores[keep,cls]
            if not len(b):continue
            indices=nms(b,s,nms_iou)
            image.extend((cls,float(s[i]),b[i].tolist()) for i in indices)
        result.append(image)
    return result


def voc_iou(box,boxes):
    box=np.asarray(box,dtype=np.float64);boxes=np.asarray(boxes,dtype=np.float64).reshape(-1,4)
    intersection=np.maximum(np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2])+1,0).prod(1)
    area=np.maximum(box[2:]-box[:2]+1,0).prod()
    union=area+np.maximum(boxes[:,2:]-boxes[:,:2]+1,0).prod(1)-intersection
    return intersection/np.maximum(union,1e-12)


def score_class(ground_truth,records,cls):
    gt={key:[dict(o,matched=False) for o in objects if o['class']==cls] for key,objects in ground_truth.items()}
    positives=sum(not o['difficult'] for objects in gt.values() for o in objects)
    records=sorted(records,key=lambda item:-item[1]);tp=np.zeros(len(records));fp=np.zeros(len(records))
    for index,(key,score,box) in enumerate(records):
        if key not in gt:raise ValueError('Detection outside evaluated split')
        objects=gt[key]
        if not objects:fp[index]=1;continue
        overlaps=voc_iou(box,[o['bbox'] for o in objects]);best=int(overlaps.argmax());obj=objects[best]
        if overlaps[best]>.5:
            if obj['difficult']:continue
            if obj['matched']:fp[index]=1
            else:tp[index]=1;obj['matched']=True
        else:fp[index]=1
    cumulative_tp,cumulative_fp=tp.cumsum(),fp.cumsum()
    recall=cumulative_tp/max(positives,1);precision=cumulative_tp/np.maximum(cumulative_tp+cumulative_fp,1e-12)
    ap=sum(float(precision[recall>=t].max()) if np.any(recall>=t) else 0 for t in np.linspace(0,1,11))/11
    return {'AP':ap if positives else None,'GT':positives,'TP':int(tp.sum()),'FP':int(fp.sum()),
            'precision':float(precision[-1]) if len(precision) else 0.,'recall':float(recall[-1]) if len(recall) else 0.}


@torch.no_grad()
def evaluate(model,rows,device,cfg,output,batch_size=4,workers=2,label='VOC2007 test'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    dataset=VOCDataset(rows);gt={key:objects for key,(objects,_,_) in dataset.annotations.items()}
    files=[(output/f'detections_{name}.txt').open('w',encoding='utf-8') for name in CLASSES]
    loader=DataLoader(dataset,batch_size=batch_size,num_workers=workers,shuffle=False,pin_memory=device.type=='cuda')
    model.eval();count=0
    try:
        for images,_,keys,sizes,_ in loader:
            predictions=model(images.to(device,non_blocking=True))
            for key,size,detections in zip(keys,sizes.tolist(),decode(predictions,cfg.score_threshold,cfg.nms_iou)):
                width,height=size
                for cls,score,box in detections:
                    x1,y1,x2,y2=box
                    # Inverse of XML inclusive -> zero-based half-open conversion.
                    values=(min(width,x1*width+1),min(height,y1*height+1),max(1,x2*width),max(1,y2*height))
                    files[cls].write(key+' '+str(score)+' '+' '.join(map(str,values))+'\n')
            count+=len(images)
            if count%200<len(images):print(f'{label}: {count}/{len(rows)}',flush=True)
    finally:
        for stream in files:stream.close()
    if count!=len(rows):raise ValueError('Incomplete evaluation')
    per_class={}
    for cls,name in enumerate(CLASSES):
        records=[]
        with (output/f'detections_{name}.txt').open() as stream:
            for line in stream:
                fields=line.split();records.append((fields[0],float(fields[1]),list(map(float,fields[2:]))))
        per_class[name]=score_class(gt,records,cls)
    aps=[v['AP'] for v in per_class.values() if v['AP'] is not None]
    if len(aps)!=20:raise ValueError('Evaluated split has missing VOC class')
    result={'dataset':label,'images':count,'mAP50_voc2007_11point':float(np.mean(aps)),
            'mAP50_percent':float(np.mean(aps)*100),
            'per_class':per_class,'score_threshold':cfg.score_threshold,'nms_iou':cfg.nms_iou,
            'difficult':'ignored in AP; excluded in training','iou':'strict >0.5, inclusive VOC coordinates'}
    write_json(output/'metrics.json',result);return result


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark(model,rows,device,cfg,iterations=100):
    if iterations<1:raise ValueError('iterations must be positive')
    model.eval();paths=[row[1] for row in rows[:16]];inputs=[]
    for path in paths:
        with Image.open(path) as image:inputs.append(preprocess(image.convert('RGB')).unsqueeze(0).to(device))
    if not inputs:raise ValueError('No timing images')
    for i in range(20):model(inputs[i%len(inputs)])
    synchronize(device);network=[];pipeline=[]
    for i in range(iterations):
        synchronize(device);start=time.perf_counter()
        model(inputs[i%len(inputs)]);synchronize(device)
        network.append(time.perf_counter()-start)
    # Warm file cache and CPU postprocessing independently of the measured loop.
    for tensor in inputs:decode(model(tensor),cfg.score_threshold,cfg.nms_iou)
    for i in range(iterations):
        synchronize(device);start=time.perf_counter()
        with Image.open(paths[i%len(paths)]) as image:tensor=preprocess(image.convert('RGB')).unsqueeze(0).to(device)
        decode(model(tensor),cfg.score_threshold,cfg.nms_iou);synchronize(device)
        pipeline.append(time.perf_counter()-start)
    return {'batch':1,'precision':'FP32','warmup':20,'iterations':iterations,'distinct_images':len(inputs),
            'network_fps':len(network)/sum(network),'pipeline_fps':len(pipeline)/sum(pipeline),
            'network_mean_ms':float(np.mean(network)*1000),'network_p50_ms':float(np.percentile(network,50)*1000),
            'network_p95_ms':float(np.percentile(network,95)*1000),'pipeline_mean_ms':float(np.mean(pipeline)*1000),
            'network_scope':'resident GPU tensor to raw output; excludes transfers/preprocessing/NMS',
            'pipeline_scope':'cached-file open/decode, resize, normalization, H2D, forward, D2H, class NMS; excludes drawing',
            'device':str(device),'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'cuda_synchronized':device.type=='cuda'}
