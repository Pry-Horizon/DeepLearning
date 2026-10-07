from pathlib import Path
import inspect
import torch
from torch import nn
from torchvision.models import resnet34
from .config import Config,WEIGHTS,digest,PRETRAINED_SHA256


def load_checkpoint(path,map_location='cpu'):
    kwargs={'map_location':map_location}
    if 'weights_only' in inspect.signature(torch.load).parameters:kwargs['weights_only']=False
    return torch.load(path,**kwargs)


class YOLOv1(nn.Module):
    def __init__(self,cfg=None,pretrained=None):
        super().__init__();self.cfg=cfg or Config()
        backbone=resnet34(weights=None)
        self.initialization=None
        if pretrained is not None:
            path=Path(pretrained)
            fingerprint=digest(path)
            if fingerprint!=PRETRAINED_SHA256:raise ValueError('Not the pinned torchvision ResNet34 ImageNet V1 weights')
            backbone.load_state_dict(load_checkpoint(path),strict=True)
            self.initialization={'kind':'torchvision.ResNet34_Weights.IMAGENET1K_V1','sha256':fingerprint}
        self.backbone=nn.Sequential(*list(backbone.children())[:-2])
        layers=[];channels=512
        # 14x14 backbone map -> four detector convs -> 7x7.
        for stride in (1,1,2,1):
            layers += [nn.Conv2d(channels,1024,3,stride,1),nn.LeakyReLU(.1,inplace=True)]
            channels=1024
        self.detector=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(7*7*1024,self.cfg.hidden),
                               nn.LeakyReLU(.1,inplace=True),nn.Dropout(self.cfg.dropout),
                               nn.Linear(self.cfg.hidden,7*7*30))
        for layer in list(self.detector.modules())+list(self.head.modules()):
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight,a=.1,nonlinearity='leaky_relu');nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight,std=.01);nn.init.zeros_(layer.bias)
    def train(self,mode=True):
        super().train(mode)
        if mode and self.cfg.freeze_bn_statistics:
            for layer in self.backbone.modules():
                if isinstance(layer,nn.BatchNorm2d):layer.eval()
        return self
    def forward(self,x):
        features=self.detector(self.backbone(x))
        return self.head(features).reshape(-1,7,7,30)


def unpack(prediction):
    return prediction[...,:20],prediction[...,20:22],prediction[...,22:].reshape(*prediction.shape[:-1],2,4)


def box_corners(boxes):
    device,dtype=boxes.device,boxes.dtype
    yy,xx=torch.meshgrid(torch.arange(7,device=device),torch.arange(7,device=device),indexing='ij')
    grid=torch.stack((xx,yy),-1).to(dtype).reshape(1,7,7,1,2)
    center=(boxes[...,:2]+grid)/7
    size=boxes[...,2:].square()
    return torch.cat((center-size/2,center+size/2),-1)


def iou(a,b):
    intersection=(torch.minimum(a[...,2:],b[...,2:])-torch.maximum(a[...,:2],b[...,:2])).clamp(min=0).prod(-1)
    area_a=(a[...,2:]-a[...,:2]).clamp(min=0).prod(-1)
    area_b=(b[...,2:]-b[...,:2]).clamp(min=0).prod(-1)
    return intersection/(area_a+area_b-intersection).clamp(min=1e-9)
