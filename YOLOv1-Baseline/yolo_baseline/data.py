import math
import random
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler
from PIL import Image,ImageEnhance
from torchvision.transforms import functional as TF
from .config import CLASSES,DATA

MEAN=(.485,.456,.406);STD=(.229,.224,.225)


def annotation(path):
    xml=ET.parse(path).getroot()
    width,height=int(xml.findtext('size/width')),int(xml.findtext('size/height'))
    objects=[]
    for obj in xml.findall('object'):
        name=obj.findtext('name')
        if name not in CLASSES:continue
        bbox=[float(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
        objects.append({'class':CLASSES.index(name),'bbox':bbox,'difficult':int(obj.findtext('difficult','0'))})
    return objects,width,height


def records(root,year,split):
    base=Path(root)/('VOC'+year)
    ids=(base/'ImageSets/Main'/f'{split}.txt').read_text().split()
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate VOC IDs')
    rows=[]
    for key in ids:
        image,xml=base/'JPEGImages'/f'{key}.jpg',base/'Annotations'/f'{key}.xml'
        if not image.is_file() or not xml.is_file():raise FileNotFoundError(f'{image} / {xml}')
        rows.append((f'{year}_{key}',image,xml))
    return rows


def splits(root=DATA,mode='full'):
    train=records(root,'2007','trainval' if mode=='full' else 'train')+records(root,'2012','trainval')
    val=records(root,'2007','val') if mode=='dev' else []
    test=records(root,'2007','test')
    expected=16551 if mode=='full' else 14041
    if len(train)!=expected or len(test)!=4952 or (mode=='dev' and len(val)!=2510):raise ValueError('Official full VOC splits required')
    sets=[{r[0] for r in rows} for rows in (train,val,test)]
    if any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2))):raise ValueError('Split leakage')
    return train,val,test


def preprocess(image):
    image=image.resize((448,448),Image.Resampling.BILINEAR)
    return TF.normalize(TF.to_tensor(image),MEAN,STD)


def augment(image,boxes,rng):
    width,height=image.size
    sx,sy=rng.uniform(.8,1.2),rng.uniform(.8,1.2)
    tx,ty=rng.uniform(-.2,.2)*width,rng.uniform(-.2,.2)*height
    image=image.transform(image.size,Image.Transform.AFFINE,(1/sx,0,-tx/sx,0,1/sy,-ty/sy),
                          resample=Image.Resampling.BILINEAR,fillcolor=(124,116,104))
    boxes=boxes.copy()
    if len(boxes):
        boxes[:,[0,2]]=boxes[:,[0,2]]*sx+tx;boxes[:,[1,3]]=boxes[:,[1,3]]*sy+ty
        boxes[:,[0,2]]=boxes[:,[0,2]].clip(0,width);boxes[:,[1,3]]=boxes[:,[1,3]].clip(0,height)
    if rng.random()<.5:
        image=TF.hflip(image)
        if len(boxes):boxes[:,[0,2]]=width-boxes[:,[2,0]]
    image=ImageEnhance.Color(image).enhance(rng.uniform(1/1.5,1.5))
    image=ImageEnhance.Brightness(image).enhance(rng.uniform(1/1.5,1.5))
    return image,boxes


def encode(boxes,labels,width,height):
    target=torch.zeros(7,7,25);collisions=0
    if not len(boxes):return target,collisions
    boxes=np.asarray(boxes,dtype=np.float32)/np.array([width,height,width,height],dtype=np.float32)
    sizes=boxes[:,2:]-boxes[:,:2];centers=(boxes[:,:2]+boxes[:,2:])/2
    # One class and one ground-truth box per cell: retain largest object deterministically.
    for index in np.argsort(-(sizes[:,0]*sizes[:,1]),kind='stable'):
        if np.any(sizes[index]<=0):continue
        x,y=np.clip(centers[index],0,1-1e-7);col,row=int(x*7),int(y*7)
        if target[row,col,20]:collisions+=1;continue
        target[row,col,int(labels[index])]=1;target[row,col,20]=1
        target[row,col,21:25]=torch.tensor([x*7-col,y*7-row,*sizes[index]])
    return target,collisions


class VOCDataset(Dataset):
    def __init__(self,rows,training=False,seed=2026):
        self.rows=rows;self.training=training;self.seed=seed;self.epoch=0
        self.annotations={key:annotation(xml) for key,_,xml in rows}
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        key,path,_=self.rows[index]
        with Image.open(path) as stream:image=stream.convert('RGB')
        objects,width,height=self.annotations[key]
        if image.size!=(width,height):raise ValueError(f'XML/image dimension mismatch: {path}')
        objects=[o for o in objects if not o['difficult']]
        boxes=np.array([[o['bbox'][0]-1,o['bbox'][1]-1,o['bbox'][2],o['bbox'][3]] for o in objects],dtype=np.float32).reshape(-1,4)
        labels=[o['class'] for o in objects]
        if self.training:image,boxes=augment(image,boxes,random.Random(self.seed+self.epoch*1000003+index))
        target,collisions=encode(boxes,labels,width,height)
        return preprocess(image),target,key,torch.tensor([width,height]),collisions


class FullBatchSampler(Sampler):
    """Shuffle full dataset, pad final effective batch with first shuffled images."""
    def __init__(self,size,micro_batch,effective_batch,seed,epoch):
        self.size=size;self.micro=micro_batch;self.batch=effective_batch;self.seed=seed;self.epoch=epoch
    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch)
        indices=torch.randperm(self.size,generator=generator).tolist()
        count=math.ceil(self.size/self.batch)*self.batch
        indices+=(indices*math.ceil((count-self.size)/self.size))[:count-self.size]
        for start in range(0,count,self.micro):yield indices[start:start+self.micro]
    def __len__(self):return math.ceil(self.size/self.batch)*self.batch//self.micro
