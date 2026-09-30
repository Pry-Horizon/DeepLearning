"""Author VOC label conversion, Python 3 adaptation and verified train/test lists."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from protocol import ROOT,DATA,BUILD,RUN,CLASSES,TRAIN_IMAGES,TEST_IMAGES


def annotation(path):
    xml=ET.parse(path).getroot()
    w,h=int(xml.findtext('size/width')),int(xml.findtext('size/height'))
    objects=[]
    for obj in xml.findall('object'):
        name=obj.findtext('name')
        if name not in CLASSES:continue
        objects.append({'class':CLASSES.index(name),'difficult':int(obj.findtext('difficult','0')),
                        'bbox':[float(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]})
    return objects,w,h


def samples(year,split,data=DATA):
    base=Path(data)/('VOC'+year)
    ids=(base/'ImageSets/Main'/f'{split}.txt').read_text().split()
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate split IDs')
    result=[]
    for image_id in ids:
        image,xml=base/'JPEGImages'/f'{image_id}.jpg',base/'Annotations'/f'{image_id}.xml'
        if not image.is_file() or not xml.is_file():raise FileNotFoundError(f'{image} / {xml}')
        result.append((year,image_id,image,xml))
    return result


def convert_label(xml):
    objects,w,h=annotation(xml)
    lines=[]
    for obj in objects:
        if obj['difficult']:continue
        x1,y1,x2,y2=obj['bbox']
        # Exact equations from upstream scripts/voc_label.py: no xmin-1 adjustment.
        values=((x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h)
        lines.append(str(obj['class'])+' '+' '.join(map(str,values)))
    return '\n'.join(lines)+ ('\n' if lines else '')


def prepare(data=DATA):
    train=samples('2007','trainval',data)+samples('2012','trainval',data)
    test=samples('2007','test',data)
    if len(train)!=TRAIN_IMAGES or len(test)!=TEST_IMAGES:raise ValueError('Full official VOC splits required')
    if {(y,i) for y,i,*_ in train}&{(y,i) for y,i,*_ in test}:raise ValueError('Train/test overlap')
    for year,image_id,image,xml in train+test:
        label=image.parent.parent/'labels'/f'{image_id}.txt'
        label.parent.mkdir(parents=True,exist_ok=True)
        text=convert_label(xml)
        if label.exists() and label.read_text()!=text:raise ValueError(f'Existing label differs: {label}')
        label.write_text(text,encoding='utf-8')
    (BUILD/'data').mkdir(parents=True,exist_ok=True)
    for name,records in [('voc.0712.trainval',train),('voc.2007.test',test)]:
        (BUILD/'data'/name).write_text(''.join(image.resolve().as_posix()+'\n' for _,_,image,_ in records),encoding='utf-8')
    RUN.mkdir(parents=True,exist_ok=True)
    info={'train':len(train),'test':len(test),'overlap':0,'data_root':str(Path(data).resolve()),
          'label_source':'darknet/scripts/voc_label.py','difficult_training':'excluded'}
    (RUN/'data.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info),flush=True)
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=DATA)
    prepare(p.parse_args().data_root)
