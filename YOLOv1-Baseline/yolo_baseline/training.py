import json
import math
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from .config import ROOT,Config,DATA,WEIGHTS,write_json,sources,digest
from .data import splits,VOCDataset,FullBatchSampler
from .model import YOLOv1,load_checkpoint
from .loss import YOLOLoss
from .evaluation import evaluate,benchmark
from .runtime import seed_all,save_checkpoint,restore_random,environment


def train(args):
    cfg=Config(micro_batch=args.micro_batch,epochs=args.epochs,seed=args.seed,split=args.split);cfg.validate()
    device=torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise ValueError('Formal training requires an available CUDA device')
    seed_all(cfg.seed);source_hashes=sources();output=Path(args.output)
    train_rows,val_rows,test_rows=splits(args.data_root,cfg.split)
    checkpoint=load_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        if 'optimizer' not in checkpoint or 'rng' not in checkpoint:raise ValueError('Resume requires full last.pt, not an inference-only checkpoint')
        if checkpoint.get('config')!=cfg.dictionary():raise ValueError('Resume config differs; use the original epochs, seed, micro-batch and split')
        if checkpoint.get('source_hashes')!=source_hashes:raise ValueError('Resume source differs from saved training implementation')
        if not (output/'run.json').is_file():raise ValueError('Resume must use the original output directory')
    elif output.exists() and any(output.iterdir()):raise FileExistsError('Output directory already has results; use --resume last.pt or a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    model=YOLOv1(cfg,pretrained=None if checkpoint else args.pretrained).to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=cfg.lr,momentum=cfg.momentum,weight_decay=cfg.decay,foreach=False)
    initialization=checkpoint['initialization'] if checkpoint else model.initialization
    start,updates,best_ap=0,0,-1.
    if checkpoint:
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        start=checkpoint['completed_epochs'];updates=checkpoint['updates'];best_ap=checkpoint['best_validation_ap']
        restore_random(checkpoint['rng']);del checkpoint
    if start>cfg.epochs:raise ValueError('Checkpoint exceeds configured epochs')
    protocol={'config':cfg.dictionary(),'train_images':len(train_rows),'val_images':len(val_rows),'test_images':len(test_rows),
              'padded_samples_per_epoch':math.ceil(len(train_rows)/64)*64-len(train_rows),
              'model_selection':'last configured epoch' if cfg.split=='full' else 'best VOC2007 val mAP',
              'initialization':initialization,'source_hashes':source_hashes,'environment':environment(),
              'differences_from_original':['ResNet34 backbone with pretrained BN; backbone BN statistics frozen',
                  'ImageNet mean/std normalization','deterministic shuffle and final batch padding',
                  'explicit 5-epoch warmup to 0.001; no claim of original pretraining recipe'],
              'resume':'epoch-boundary checkpoint restores optimizer and RNG; unfinished epoch must be replayed'}
    if not args.resume:write_json(output/'run.json',protocol)
    dataset=VOCDataset(train_rows,training=True,seed=cfg.seed);criterion=YOLOLoss()
    import hashlib
    fingerprint=hashlib.sha256(json.dumps(dataset.annotations,sort_keys=True).encode()).hexdigest()
    original=json.loads((output/'run.json').read_text(encoding='utf-8'))
    if args.resume and original.get('annotation_sha256')!=fingerprint:raise ValueError('Training annotations changed since the saved run')
    if not args.resume:
        protocol['annotation_sha256']=fingerprint;write_json(output/'run.json',protocol)
    accum=cfg.effective_batch//cfg.micro_batch
    for epoch in range(start,cfg.epochs):
        dataset.epoch=epoch
        sampler=FullBatchSampler(len(dataset),cfg.micro_batch,cfg.effective_batch,cfg.seed,epoch)
        generator=torch.Generator().manual_seed(cfg.seed+epoch)
        loader=DataLoader(dataset,batch_sampler=sampler,num_workers=args.workers,pin_memory=True,generator=generator)
        model.train();optimizer.zero_grad(set_to_none=True)
        lr=cfg.learning_rate(epoch)
        for group in optimizer.param_groups:group['lr']=lr
        total=0.;seen=0;collisions=0;begin=time.perf_counter()
        for step,(images,target,_,_,collision) in enumerate(loader,1):
            images,target=images.to(device,non_blocking=True),target.to(device,non_blocking=True)
            loss=criterion(model(images),target,reduction='sum')/cfg.effective_batch
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss; last completed epoch is preserved')
            loss.backward();total+=float(loss.detach())*cfg.effective_batch;seen+=len(images);collisions+=int(collision.sum())
            if step%accum==0:
                # foreach=False avoids allocating another full model-sized tensor list.
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),foreach=False)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                if updates%20==0:print(f'epoch {epoch+1}/{cfg.epochs} update {updates} loss {total/seen:.5f} lr {lr:g}',flush=True)
        entry={'epoch':epoch+1,'loss':total/seen,'lr':lr,'samples':seen,'updates':updates,
               'same_cell_collisions':collisions,'seconds':time.perf_counter()-begin}
        improved=False
        if val_rows:
            val=evaluate(model,val_rows,device,cfg,output/'validation',args.eval_batch,args.workers,'VOC2007 val')
            ap=val['mAP50_voc2007_11point'];entry['validation_mAP50']=ap
            if ap>best_ap:best_ap=ap;improved=True
        save_checkpoint(output/'last.pt',model,optimizer,cfg,epoch+1,updates,best_ap,initialization,source_hashes)
        if improved:
            # Best checkpoint is inference-only; resume always uses last.pt.
            torch.save({'schema':1,'model':model.state_dict(),'config':cfg.dictionary(),'completed_epochs':epoch+1,
                        'initialization':initialization,'source_hashes':source_hashes},output/'best.pt.part')
            (output/'best.pt.part').replace(output/'best.pt')
        with (output/'history.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(entry,allow_nan=False)+'\n')
        print(json.dumps(entry),flush=True)
    selected=output/('best.pt' if val_rows else 'last.pt')
    if val_rows:
        state=load_checkpoint(selected);model.load_state_dict(state['model']);selected_epoch=state['completed_epochs'];del state
    else:selected_epoch=cfg.epochs
    result=evaluate(model,test_rows,device,cfg,output/'test',args.eval_batch,args.workers)
    result.update({'model':cfg.name,'checkpoint':str(selected.resolve()),'selected_epoch':selected_epoch,
                   'test_used_for_model_selection':False,'config':cfg.dictionary(),'environment':environment(),
                   'checkpoint_sha256':digest(selected),'initialization':initialization,'source_hashes':source_hashes,
                   'train_images':len(train_rows),'validation_images':len(val_rows),
                   'timing':benchmark(model,test_rows,device,cfg,args.iterations)})
    write_json(output/'metrics.json',result);print(json.dumps(result,indent=2),flush=True)
