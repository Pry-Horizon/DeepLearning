from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data/VOCdevkit'
WEIGHTS=ROOT/'data/pretrained/resnet34-b627a593.pth'
PRETRAINED_SHA256='b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f'
CLASSES=('aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
         'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor')


@dataclass
class Config:
    name: str='YOLOv1-ResNet34-FC'
    schema: int=1
    image_size: int=448
    grid: int=7
    boxes: int=2
    classes: int=20
    hidden: int=4096
    dropout: float=.5
    epochs: int=135
    effective_batch: int=64
    micro_batch: int=4
    lr: float=.001
    warmup_epochs: int=5
    warmup_start: float=.0001
    momentum: float=.9
    decay: float=.0005
    seed: int=2026
    split: str='full'
    freeze_bn_statistics: bool=True
    score_threshold: float=.001
    nms_iou: float=.5
    def validate(self):
        if self.micro_batch<1 or self.effective_batch%self.micro_batch:
            raise ValueError('micro-batch must divide effective batch 64')
        if self.epochs<1 or self.split not in ('full','dev'):raise ValueError('Invalid epochs/split')
    def dictionary(self):return asdict(self)
    def learning_rate(self,epoch):
        if epoch<self.warmup_epochs:
            ratio=epoch/max(self.warmup_epochs-1,1)
            return self.warmup_start+(self.lr-self.warmup_start)*ratio
        return self.lr*(.1 if epoch>=75 else 1)*(.1 if epoch>=105 else 1)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def sources():return {p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.json.part')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    tmp.replace(path)
