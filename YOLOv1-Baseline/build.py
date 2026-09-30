"""Generate an audited native build. Vendor source stays byte-identical to GitHub."""
import argparse
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
from protocol import *


def replace_once(text,before,after):
    if text.count(before)!=1:raise ValueError(f'Unexpected source: {before}')
    return text.replace(before,after)


def generate():
    lock=verify_upstream()
    for rel in lock['files']:
        target=BUILD/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(UPSTREAM/rel,target)
    yolo=(UPSTREAM/'src/yolo.c').read_text()
    yolo=replace_once(yolo,'char *backup_directory = "/home/pjreddie/backup/";',
                      'char *backup_directory = "backup";')
    yolo=replace_once(yolo,'list *plist = get_paths("data/voc.2012.test");',
                      'list *plist = get_paths("data/voc.2007.test");')
    # Save only last + final, avoiding ~38 GiB of periodic weights. Training math unchanged.
    yolo=replace_once(yolo,'if(i%1000==0){',f'if((i*64)/{TRAIN_IMAGES} != ((i-1)*64)/{TRAIN_IMAGES}){{')
    yolo=replace_once(yolo,'sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);',
                      'sprintf(buff, "%s/%s_last.weights", backup_directory, base);')
    yolo=replace_once(yolo,'i += 1;',f'i += 1;\n        printf("PAPER_EPOCH %d/{EPOCHS} UPDATE %d/{TOTAL_STEPS}\\n", ((i-1)*64)/{TRAIN_IMAGES}+1, i);\n        fflush(stdout);')
    (BUILD/'src/yolo.c').write_text(yolo,encoding='utf-8')
    main=(UPSTREAM/'src/darknet.c').read_text()
    main=replace_once(main,'extern void run_yolo(int argc, char **argv);',
                      'extern void run_yolo(int argc, char **argv);\nextern void run_benchmark(int argc, char **argv);')
    main=replace_once(main,'if(0==strcmp(argv[1], "imagenet")){',
                      'if(0==strcmp(argv[1], "benchmark")){\n        run_benchmark(argc, argv);\n    } else if(0==strcmp(argv[1], "imagenet")){')
    (BUILD/'src/darknet.c').write_text(main,encoding='utf-8')
    shutil.copyfile(ROOT/'support/benchmark.c',BUILD/'src/benchmark.c')
    make=(UPSTREAM/'Makefile').read_text()
    make=replace_once(make,'OBJ=gemm.o','OBJ=benchmark.o gemm.o')
    make=replace_once(make,'OPTS=-Ofast','OPTS=-Ofast -fcommon')
    (BUILD/'Makefile').write_text(make,encoding='utf-8')
    cfg=paper_cfg()
    (BUILD/'cfg/paper.cfg').write_text(cfg,encoding='utf-8')
    infer=replace_once(replace_once(cfg,'batch=64','batch=1'),'subdivisions=64','subdivisions=1')
    (BUILD/'cfg/paper-inference.cfg').write_text(infer,encoding='utf-8')
    (BUILD/'backup').mkdir(exist_ok=True);(BUILD/'results').mkdir(exist_ok=True)
    patches=[]
    for rel in ('src/yolo.c','src/darknet.c','Makefile'):
        patches.extend(difflib.unified_diff((UPSTREAM/rel).read_text().splitlines(True),
                                            (BUILD/rel).read_text().splitlines(True),fromfile='upstream/'+rel,tofile='build/'+rel))
    patches.extend(difflib.unified_diff((UPSTREAM/'cfg/yolo.cfg').read_text().splitlines(True),
                                        cfg.splitlines(True),fromfile='upstream/cfg/yolo.cfg',tofile='build/cfg/paper.cfg'))
    (ROOT/'support/build.patch').write_text(''.join(patches),encoding='utf-8')
    RUN.mkdir(parents=True,exist_ok=True)
    (RUN/'protocol.json').write_text(json.dumps(recipe(),indent=2),encoding='utf-8')
    return {'source':lock['commit'],'generated':True,'cfg_sha256':sha256(BUILD/'cfg/paper.cfg')}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--generate-only',action='store_true')
    p.add_argument('--cpu',action='store_true',help='Build diagnostics only; paper training requires GPU crop/HSV path')
    p.add_argument('--arch',default='sm_120',help='GPU code; validated against nvcc supported list')
    args=p.parse_args();info=generate()
    if args.generate_only:print(json.dumps(info));return
    if os.name=='nt':raise SystemExit('Native author build requires Linux/WSL with gcc, make and CUDA toolkit; no such toolchain detected here.')
    for name in ('gcc','make'):
        if not shutil.which(name):raise SystemExit(f'Missing build tool: {name}')
    command=['make','-B','-j2',f'GPU={0 if args.cpu else 1}','OPENCV=0']
    if not args.cpu:
        nvcc=shutil.which('nvcc')
        if not nvcc:raise SystemExit('Missing nvcc CUDA toolkit (PyTorch runtime is insufficient)')
        supported=subprocess.check_output([nvcc,'--list-gpu-code'],text=True).split()
        if args.arch not in supported:raise SystemExit(f'{args.arch} not supported by this nvcc')
        command.extend([f'NVCC={nvcc}',f'ARCH=-arch={args.arch}'])
    subprocess.run(command,cwd=BUILD,check=True)
    info.update({'gpu':not args.cpu,'command':command,'executable_sha256':sha256(BUILD/'darknet'),
                 'compiled_sources':{str(p.relative_to(BUILD)).replace('\\','/'):sha256(p)
                                     for p in (BUILD/'src').iterdir() if p.suffix in ('.c','.cu','.h')}})
    (BUILD/'build-info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
