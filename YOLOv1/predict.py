"""Standalone original YOLOv1 image prediction. No training or VOC dataset needed."""
import argparse
import inspect
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
import torch
from torch import nn
from torchvision import transforms as T
from torchvision.ops import nms
ROOT=Path(__file__).resolve().parent
IMAGE_EXTENSIONS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}


CLASSES=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

ARCHITECTURE=[(64,7,2),'M',(192,3,1),'M',(128,1,1),(256,3,1),
              (256,1,1),(512,3,1),'M']
ARCHITECTURE += [(256,1,1),(512,3,1)]*4
ARCHITECTURE += [(512,1,1),(1024,3,1),'M']
ARCHITECTURE += [(512,1,1),(1024,3,1)]*2
ARCHITECTURE += [(1024,3,1),(1024,3,2),(1024,3,1),(1024,3,1)]


def pack_output(raw):
    classes=raw[:,:980].reshape(-1,7,7,20)
    confidence=raw[:,980:1078].reshape(-1,7,7,2)
    boxes=raw[:,1078:].reshape(-1,7,7,8)
    return torch.cat((classes,confidence,boxes),-1)

def unpack(prediction):
    return prediction[...,:20],prediction[...,20:22],prediction[...,22:].reshape(*prediction.shape[:-1],2,4)

def box_corners(boxes):
    device,dtype=boxes.device,boxes.dtype
    yy,xx=torch.meshgrid(torch.arange(7,device=device),torch.arange(7,device=device),indexing='ij')
    grid=torch.stack((xx,yy),-1).to(dtype).reshape(1,7,7,1,2)
    center=(boxes[...,:2]+grid)/7
    size=boxes[...,2:].square()
    return torch.cat((center-size/2,center+size/2),-1)

class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[];channels=3
        for spec in ARCHITECTURE:
            if spec=='M':layers.append(nn.MaxPool2d(2,2))
            else:
                out,kernel,stride=spec
                layers += [nn.Conv2d(channels,out,kernel,stride,kernel//2),nn.LeakyReLU(.1,inplace=True)]
                channels=out
        self.features=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(7*7*1024,4096),
                               nn.LeakyReLU(.1,inplace=True),nn.Dropout(.5),nn.Linear(4096,1470))
    def forward(self,x):
        return pack_output(self.head(self.features(x)))


def preprocess(image):
    image=image.resize((448,448),Image.Resampling.BILINEAR)
    return T.functional.normalize(T.functional.to_tensor(image),(.5,.5,.5),(.5,.5,.5))


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


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

def image_paths(source):
    source=Path(source)
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:raise ValueError('Source must be a supported image')
        return [source]
    if source.is_dir():
        paths=sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if paths:return paths
        raise ValueError('Source folder has no supported images')
    raise FileNotFoundError(f'Image or folder not found: {source}')

def draw_detections(image,detections):
    canvas=image.copy();draw=ImageDraw.Draw(canvas);width,height=canvas.size
    font_size=max(12,min(32,round(min(width,height)/35)))
    font=None
    for name in ('arial.ttf','DejaVuSans.ttf'):
        try:font=ImageFont.truetype(name,font_size);break
        except OSError:pass
    if font is None:font=ImageFont.load_default()
    palette=('#ff5050','#28d78e','#50aaff','#ffcd32','#d17aff','#ff8d35')
    records=[]
    for cls,score,box in sorted(detections,key=lambda d:d[1]):
        x1,y1,x2,y2=box
        bbox=[max(0,min(width-1,x1*width)),max(0,min(height-1,y1*height)),
              max(0,min(width-1,x2*width)),max(0,min(height-1,y2*height))]
        color=palette[cls%len(palette)];draw.rectangle(bbox,outline=color,width=max(2,font_size//7))
        label=f'{CLASSES[cls]} {score:.2f}';text_box=draw.textbbox((0,0),label,font=font)
        tw,th=text_box[2]-text_box[0],text_box[3]-text_box[1]
        tx=max(0,min(bbox[0],width-tw-6));ty=max(0,bbox[1]-th-6)
        draw.rectangle((tx,ty,tx+tw+6,ty+th+6),fill=color)
        draw.text((tx+3,ty+3-text_box[1]),label,font=font,fill='black')
        records.append({'class_id':cls,'class':CLASSES[cls],'confidence':score,'box_xyxy_pixels':bbox})
    return canvas,sorted(records,key=lambda r:-r['confidence'])

@torch.no_grad()
def predict(model,paths,device,output,confidence,nms_iou):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);model.eval();results=[]
    for index,path in enumerate(paths,1):
        with Image.open(path) as stream:image=ImageOps.exif_transpose(stream).convert('RGB')
        predictions=model(preprocess(image).unsqueeze(0).to(device))
        detections=decode(predictions,confidence,nms_iou)[0]
        canvas,records=draw_detections(image,detections)
        destination=output/f'{index:04d}_{path.stem}.png'
        if destination.resolve()==path.resolve():raise ValueError('Output would overwrite the input image')
        canvas.save(destination)
        results.append({'source':str(path.resolve()),'annotated_image':str(destination.resolve()),
                        'width':image.width,'height':image.height,'detections':records})
        print(f'{path.name}: {len(records)} detections -> {destination}',flush=True)
    return results

def main():
    parser=argparse.ArgumentParser(description='Load raw.pt and predict images with original YOLOv1.')
    parser.add_argument('--weights',type=Path,default=ROOT/'checkpoints'/'raw.pt')
    parser.add_argument('--source',type=Path,default=ROOT/'images')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--conf',type=float,default=.2)
    parser.add_argument('--nms-iou',type=float,default=.5)
    parser.add_argument('--device',default='auto')
    args=parser.parse_args()
    if not 0<=args.conf<=1 or not 0<args.nms_iou<=1:parser.error('Invalid confidence/NMS threshold')
    if not args.weights.is_file():parser.error(f'Download the final raw.pt from the server to: {args.weights}')
    paths=image_paths(args.source)
    kwargs={'map_location':'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:kwargs['weights_only']=False
    state=torch.load(args.weights,**kwargs)
    cfg=state.get('config',{})
    required={'name':'YOLOv1-Original-Scratch','image_size':448,'grid':7,'boxes':2,'classes':20,'hidden':4096,'dropout':.5}
    if any(cfg.get(k)!=v for k,v in required.items()):raise ValueError('Weights do not belong to this original YOLOv1 architecture')
    if state.get('completed_epochs',0)<1:raise ValueError('Checkpoint has no trained epochs')
    device=torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    model=YOLOv1().to(device);model.load_state_dict(state['model'],strict=True);model.eval()
    metadata={'weights':str(args.weights.resolve()),'weights_kind':state.get('weights_kind','checkpoint'),
              'completed_epochs':state['completed_epochs'],'total_training_epochs':state.get('total_training_epochs'),
              'device':str(device),'display_threshold':args.conf,'nms_iou':args.nms_iou}
    del state
    metadata['results']=predict(model,paths,device,args.output,args.conf,args.nms_iou)
    write_json(args.output/'predictions.json',metadata)
    print(f'Finished. Open: {args.output.resolve()}')


if __name__=='__main__':main()
