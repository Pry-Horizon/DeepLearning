import torch
from torch import nn
from .model import unpack,box_corners,iou


class YOLOLoss(nn.Module):
    """Linear class/confidence/xy outputs; raw width-height outputs represent sqrt(w,h)."""
    def forward(self,pred,target,reduction='mean'):
        classes,confidence,boxes=unpack(pred)
        obj=target[...,20].bool()
        truth=target[...,21:25]
        truth_sqrt=torch.cat((truth[...,:2],truth[...,2:].clamp(min=0).sqrt()),-1).unsqueeze(-2)
        overlaps=iou(box_corners(boxes),box_corners(truth_sqrt)).detach()
        responsible=torch.nn.functional.one_hot(overlaps.argmax(-1),2).bool() & obj.unsqueeze(-1)
        xy=((boxes[...,:2]-truth_sqrt[...,:2]).square().sum(-1)*responsible).sum((1,2,3))
        wh=((boxes[...,2:]-truth_sqrt[...,2:]).square().sum(-1)*responsible).sum((1,2,3))
        positive=((confidence-overlaps).square()*responsible).sum((1,2,3))
        negative=(confidence.square()*(~responsible)).sum((1,2,3))
        classification=((classes-target[...,:20]).square().sum(-1)*obj).sum((1,2))
        total=5*(xy+wh)+positive+.5*negative+classification
        if reduction=='sum':return total.sum()
        if reduction=='none':return total
        return total.mean()
