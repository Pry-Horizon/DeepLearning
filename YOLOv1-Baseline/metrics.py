"""VOC2007 11-point AP on original Darknet result files; no replacement network."""
import argparse
import json
from pathlib import Path
import numpy as np
from protocol import DATA,RUN,BUILD,CLASSES,TEST_IMAGES
from prepare_voc import annotation,samples


def voc_iou(box,boxes):
    box=np.asarray(box,dtype=np.float64);boxes=np.asarray(boxes,dtype=np.float64).reshape(-1,4)
    intersection=np.maximum(np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2])+1,0).prod(1)
    union=np.maximum(box[2:]-box[:2]+1,0).prod()+np.maximum(boxes[:,2:]-boxes[:,:2]+1,0).prod(1)-intersection
    return intersection/np.maximum(union,1e-12)


def score_class(ground_truth,records,cls):
    gt={key:[dict(o,matched=False) for o in objects if o['class']==cls] for key,objects in ground_truth.items()}
    positives=sum(not o['difficult'] for objects in gt.values() for o in objects)
    records=sorted(records,key=lambda item:-item[1])
    tp,fp=np.zeros(len(records)),np.zeros(len(records))
    for index,(key,score,box) in enumerate(records):
        if key not in gt:raise ValueError(f'Prediction outside test split: {key}')
        objects=gt[key]
        if not objects:fp[index]=1;continue
        ious=voc_iou(box,[o['bbox'] for o in objects]);best=int(ious.argmax());obj=objects[best]
        if ious[best]>.5:
            if obj['difficult']:continue
            if obj['matched']:fp[index]=1
            else:tp[index]=1;obj['matched']=True
        else:fp[index]=1
    cum_tp,cum_fp=tp.cumsum(),fp.cumsum()
    recall=cum_tp/max(positives,1);precision=cum_tp/np.maximum(cum_tp+cum_fp,1e-12)
    ap=sum(float(precision[recall>=t].max()) if np.any(recall>=t) else 0 for t in np.linspace(0,1,11))/11
    return {'AP':ap if positives else None,'GT':positives,'TP':int(tp.sum()),'FP':int(fp.sum()),
            'recall':float(recall[-1]) if len(recall) else 0.,'precision':float(precision[-1]) if len(precision) else 0.}


def evaluate(results=BUILD/'results',data=DATA,output=RUN/'metrics.json',timing=None):
    test=samples('2007','test',data)
    if len(test)!=TEST_IMAGES:raise ValueError('All 4952 VOC2007 test images required')
    gt={key:annotation(xml)[0] for _,key,_,xml in test}
    per_class={}
    for cls,name in enumerate(CLASSES):
        path=Path(results)/f'comp4_det_test_{name}.txt'
        records=[]
        for line in path.read_text().splitlines():
            if not line.strip():continue
            fields=line.split()
            if len(fields)!=6:raise ValueError(f'Malformed detection: {path}')
            values=[float(x) for x in fields[1:]]
            if not np.isfinite(values).all():raise ValueError('Nonfinite detections')
            records.append((fields[0],values[0],values[1:]))
        per_class[name]=score_class(gt,records,cls)
    aps=[v['AP'] for v in per_class.values()]
    if any(ap is None for ap in aps):raise ValueError('Missing test ground-truth class')
    result={'dataset':'VOC2007 test','test_images':len(test),'mAP50_voc2007_11point':float(np.mean(aps)),
            'per_class':per_class,'iou':'strict >0.5, inclusive VOC coordinates',
            'score_threshold':.001,'nms_threshold':.5,'test_model_selection':False}
    if timing:result['timing']=json.loads(Path(timing).read_text())
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=BUILD/'results')
    p.add_argument('--data-root',type=Path,default=DATA);p.add_argument('--output',type=Path,default=RUN/'metrics.json')
    p.add_argument('--timing',type=Path);a=p.parse_args();evaluate(a.results,a.data_root,a.output,a.timing)
