"""Download official torchvision ImageNet V1 weights into this project only."""
import argparse
from pathlib import Path
import urllib.request
from yolo_baseline.config import WEIGHTS,digest,write_json,PRETRAINED_SHA256

URL='https://download.pytorch.org/models/resnet34-b627a593.pth'


def download(path=WEIGHTS):
    path=Path(path)
    if path.exists():
        fingerprint=digest(path)
        if fingerprint!=PRETRAINED_SHA256:raise ValueError('Existing file does not match pinned torchvision weight hash')
    else:
        path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix('.pth.part')
        with urllib.request.urlopen(URL,timeout=60) as response,part.open('wb') as output:
            total=0
            while block:=response.read(1024*1024):
                output.write(block);total+=len(block)
                if total%(10*1024*1024)==0:print(f'{total//1024//1024} MiB',flush=True)
        fingerprint=digest(part)
        if fingerprint!=PRETRAINED_SHA256:raise ValueError('Downloaded ImageNet weight checksum mismatch')
        part.replace(path)
    write_json(path.with_suffix('.provenance.json'),{'kind':'torchvision.ResNet34_Weights.IMAGENET1K_V1',
               'source':URL,'sha256':fingerprint,'bytes':path.stat().st_size})
    print(f'Verified: {path} ({path.stat().st_size} bytes)',flush=True)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=WEIGHTS)
    download(parser.parse_args().output)
