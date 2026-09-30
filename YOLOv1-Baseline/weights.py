"""Download/audit the GitHub mirror; require all 20 convs before paper training."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
from protocol import ROOT,UPSTREAM,cfg_sections,sha256

URL='https://raw.githubusercontent.com/frankzhangrui/Darknet-Yolo/1398cf31d9f048558b32d568cd0dba5b140d92be/extraction.conv.weights'
MIRROR_SHA256='048f832fbaddec03256f2c1cf380e5694d6b580f24639515587daa358ff32e9f'


def conv_sizes():
    channels=3;sizes=[]
    for name,values in cfg_sections((UPSTREAM/'cfg/extraction.cfg').read_text()):
        if name=='convolutional':
            if int(values.get('batch_normalize','0')):raise ValueError('BN backbone incompatible with paper')
            out,k=int(values['filters']),int(values['size'])
            sizes.append((out+out*channels*k*k)*4);channels=out
    if len(sizes)!=20:raise ValueError('Expected 20 ImageNet convolution layers')
    return sizes


def audit(path):
    path=Path(path)
    with path.open('rb') as f:header=f.read(16)
    if len(header)!=16:raise ValueError('Truncated header')
    # This pinned 2015 parser always reads a 16-byte header, unlike later Darknet.
    seen=struct.unpack('<i',header[12:])[0]
    size=path.stat().st_size;offset=16;layers=0
    for n in conv_sizes():
        if offset+n>size:break
        offset+=n;layers+=1
    return {'path':str(path.resolve()),'bytes':size,'sha256':sha256(path),'seen':seen,
            'complete_convolutions':layers,'required_convolutions':20,
            'expected_bytes':16+sum(conv_sizes()),'complete_first20':size==16+sum(conv_sizes())}


def require_pretrained(path):
    report=audit(path)
    if not report['complete_first20'] or report['seen']!=0:
        raise ValueError('Paper initialization requires all 20 pretrained convs, 16-byte header, seen=0; incomplete mirror rejected')
    sidecar=Path(path).with_suffix('.provenance.json')
    metadata=json.loads(sidecar.read_text())
    if metadata.get('sha256')!=report['sha256'] or metadata.get('kind')!='imagenet_first20' or not metadata.get('source'):
        raise ValueError('Missing/mismatched ImageNet provenance')
    return dict(report,provenance=metadata)


def main():
    p=argparse.ArgumentParser();p.add_argument('--download-github',action='store_true');p.add_argument('--weights',type=Path)
    a=p.parse_args()
    if a.download_github:
        a.weights=ROOT/'data/pretrained/github-extraction.conv.weights';a.weights.parent.mkdir(parents=True,exist_ok=True)
        if not a.weights.exists():
            with urllib.request.urlopen(URL,timeout=60) as response,a.weights.with_suffix('.part').open('wb') as out:
                while block:=response.read(1024*1024):out.write(block)
            a.weights.with_suffix('.part').replace(a.weights)
        if sha256(a.weights)!=MIRROR_SHA256:raise ValueError('GitHub mirror checksum mismatch')
    if not a.weights:p.error('--weights or --download-github required')
    report=audit(a.weights)
    if a.download_github:report['github_url']=URL
    a.weights.with_suffix('.audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
