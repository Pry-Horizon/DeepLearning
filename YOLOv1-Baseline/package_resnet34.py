"""Portable source bundle: excludes native Darknet, old runs and datasets."""
import json
import zipfile
import argparse
from yolo_baseline.config import ROOT,digest,PRETRAINED_SHA256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--include-weights',action='store_true');args=parser.parse_args()
    names=['README.md','SERVER.md','requirements.txt','train.py','evaluate.py','preflight.py','server.py','package_server.py',
           'prepare_resnet34.py','prepare_data.py','package_resnet34.py','test_baseline.py','smoke_baseline.py']
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'yolo_baseline').glob('*.py'))]
    if args.include_weights:
        if digest(ROOT/'data/pretrained/resnet34-b627a593.pth')!=PRETRAINED_SHA256:raise ValueError('Pretrained weight changed')
        names+=['data/pretrained/resnet34-b627a593.pth','data/pretrained/resnet34-b627a593.provenance.json']
    target=ROOT/'dist/yolov1-resnet34-server.zip';target.parent.mkdir(exist_ok=True)
    manifest={name:digest(ROOT/name) for name in names}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:archive.write(ROOT/name,name)
        archive.writestr('server-manifest.json',json.dumps(manifest,indent=2))
    print(json.dumps({'path':str(target),'bytes':target.stat().st_size,'sha256':digest(target)},indent=2))


if __name__=='__main__':main()
