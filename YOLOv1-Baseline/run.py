"""Only native author Darknet executes the model. Hyperparameters cannot be overridden."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
from protocol import *
from prepare_voc import prepare,samples
from weights import require_pretrained

FINAL_BYTES=1086814216
FINAL_WEIGHTS=BUILD/'backup/paper_final.weights'


def native_binary():
    exe=BUILD/'darknet'
    if not exe.is_file():raise FileNotFoundError('Native executable missing: run python build.py on Linux with CUDA toolkit')
    info=json.loads((BUILD/'build-info.json').read_text())
    if not info.get('gpu'):raise ValueError('CPU build is diagnostic only: original color augmentation runs in GPU crop layer')
    if info.get('executable_sha256')!=sha256(exe):raise ValueError('Native binary changed since build')
    if info.get('cfg_sha256')!=sha256(BUILD/'cfg/paper.cfg'):raise ValueError('Config changed since build')
    for rel,digest in info['compiled_sources'].items():
        if sha256(BUILD/rel)!=digest:raise ValueError(f'Compiled source changed: {rel}')
    if (BUILD/'cfg/paper.cfg').read_text()!=paper_cfg():raise ValueError('Paper training cfg has been modified')
    return exe


def check_final(path):
    if path.stat().st_size!=FINAL_BYTES:raise ValueError('Not a complete original YOLOv1 detector checkpoint')
    with path.open('rb') as f:
        f.seek(12);seen=struct.unpack('<i',f.read(4))[0]
    if seen!=TOTAL_STEPS*BATCH:raise ValueError(f'Expected {TOTAL_STEPS*BATCH} training samples, got {seen}; not the configured final run')
    return {'sha256':sha256(path),'seen':seen,'target_epochs':EPOCHS,
            'dataset_equivalent_passes':seen/TRAIN_IMAGES,'batch':BATCH}


def command_log(command,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as output:
        process=subprocess.Popen([str(x) for x in command],cwd=BUILD,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);output.write(line);output.flush()
            code=process.wait()
            if code:raise subprocess.CalledProcessError(code,command)
        except BaseException:
            process.terminate();process.wait();raise


def environment():
    result={'python':sys.version,'platform':platform.platform(),'cuda_compiler':shutil.which('nvcc'),
            'gcc':shutil.which('gcc'),'make':shutil.which('make')}
    if shutil.which('nvidia-smi'):
        result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    return result


def evaluate_and_time(weights,data,iterations):
    from metrics import evaluate
    exe=native_binary();meta=check_final(weights)
    prepare(data)
    command_log([exe,'yolo','valid','cfg/paper-inference.cfg',weights.resolve()],RUN/'evaluate.log')
    image=samples('2007','test',data)[0][2]
    timing=RUN/'timing.json'
    command_log([exe,'benchmark','cfg/paper-inference.cfg',weights.resolve(),image.resolve(),str(iterations),timing.resolve()],RUN/'benchmark.log')
    result=evaluate(BUILD/'results',data,RUN/'metrics.json',timing)
    result.update({'checkpoint':meta,'protocol':recipe(),'environment':environment()})
    (RUN/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['preflight','train','evaluate'])
    p.add_argument('--pretrained',type=Path)
    p.add_argument('--weights',type=Path,default=FINAL_WEIGHTS)
    p.add_argument('--data-root',type=Path,default=DATA)
    p.add_argument('--iterations',type=int,default=100)
    a=p.parse_args()
    if a.iterations<1:p.error('--iterations must be positive')
    verify_upstream()
    if a.action=='preflight':
        issues=[];result={'environment':environment(),'protocol':recipe()}
        for label,fn in [('data',lambda:prepare(a.data_root)),('binary',native_binary),
                         ('pretraining',lambda:require_pretrained(a.pretrained) if a.pretrained else (_ for _ in ()).throw(ValueError('Complete ImageNet first20 weights not supplied')))]:
            try:
                value=fn();result[label]=str(value) if isinstance(value,Path) else value
            except (OSError,ValueError,KeyError) as exc:issues.append(f'{label}: {exc}')
        result.update({'ready':not issues,'blockers':issues,'final_mAP':None,'final_FPS':None})
        RUN.mkdir(parents=True,exist_ok=True)
        (RUN/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2));return
    if a.action=='evaluate':evaluate_and_time(a.weights,a.data_root,a.iterations);return
    if not a.pretrained:p.error('train requires complete ImageNet --pretrained weights')
    initialization=require_pretrained(a.pretrained)
    exe=native_binary();prepare(a.data_root)
    if FINAL_WEIGHTS.exists() or (BUILD/'backup/paper_last.weights').exists() or (RUN/'train.log').exists():
        raise FileExistsError('Existing run preserved; select a clean checkout for a new strict run')
    metadata={'initialization':initialization,'protocol':recipe(),'environment':environment()}
    (RUN/'training.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    command_log([exe,'yolo','train','cfg/paper.cfg',a.pretrained.resolve()],RUN/'train.log')
    evaluate_and_time(FINAL_WEIGHTS,a.data_root,a.iterations)


if __name__=='__main__':main()
