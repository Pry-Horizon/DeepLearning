"""Paper takes precedence over the archived author's cfg. No PyTorch model port."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT/'darknet'
BUILD = ROOT/'build/darknet'
DATA = ROOT/'data/VOCdevkit'
RUN = ROOT/'runs/paper'
EPOCHS = 135
BATCH = 64
TRAIN_IMAGES = 16551
TEST_IMAGES = 4952
# The paper gives approximate epochs, not how to round 16551/64.
# Round cumulative boundaries, not each epoch; the last full batch overshoots by 47 images.
TOTAL_STEPS = math.ceil(EPOCHS*TRAIN_IMAGES/BATCH)
LR_STEPS = (200, 400, 600, math.ceil(75*TRAIN_IMAGES/BATCH), math.ceil(105*TRAIN_IMAGES/BATCH))
CLASSES = ('aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
           'cow','diningtable','dog','horse','motorbike','person','pottedplant',
           'sheep','sofa','train','tvmonitor')
COMMIT = '42ba5d4585a252b344cc737420e46ad93f005dbe'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_upstream():
    lock=json.loads((ROOT/'upstream-lock.json').read_text(encoding='utf-8'))
    if lock['commit']!=COMMIT:raise ValueError('Upstream commit changed')
    for relative,expected in lock['files'].items():
        path=UPSTREAM/relative
        if not path.is_file() or sha256(path)!=expected:
            raise ValueError(f'Upstream file changed: {relative}')
    return lock


def recipe():
    return {'basis':'paper first; original GitHub C/CUDA implementation',
            'upstream_commit':COMMIT,'epochs':EPOCHS,'batch':BATCH,
            'subdivisions':64,'micro_batch':1,
            'total_updates':TOTAL_STEPS,'train_images':TRAIN_IMAGES,'test_images':TEST_IMAGES,
            'lr':.001,'lr_steps':list(LR_STEPS),'lr_scales':[2.5,2.,2.,.1,.1],
            'momentum':.9,'decay':.0005,'input':448,'precision':'FP32',
            'paper_unspecified':{
                'warmup':'Author source: steps 200/400/600, included in first 75 reported epochs; paper has no precise warmup duration',
                'epoch_rounding':'Cumulative ceil(epoch*16551/64); 34913 full batches; final overshoot 47 images; native sampling with replacement',
                'random_state':'Native srand(time(0))/data_seed; the historical random state was not published'},
            'not_claimed':'Bitwise identical historical run or guaranteed 63.4 mAP / 45 FPS'}


def cfg_sections(text):
    sections=[]
    for line in text.splitlines():
        line=line.split('#',1)[0].strip()
        if not line:continue
        if line.startswith('['):sections.append((line[1:-1],{}))
        elif '=' in line:
            key,value=line.split('=',1); sections[-1][1][key.strip()]=value.strip()
    return sections


def paper_cfg():
    verify_upstream()
    text=(UPSTREAM/'cfg/yolo.cfg').read_text()
    replacements={'steps=200,400,600,20000,30000':'steps='+','.join(map(str,LR_STEPS)),
                  'max_batches = 40000':f'max_batches = {TOTAL_STEPS}'}
    for before,after in replacements.items():
        if text.count(before)!=1:raise ValueError(f'Unexpected cfg: {before}')
        text=text.replace(before,after)
    return text
