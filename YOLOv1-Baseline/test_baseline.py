import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from yolo_baseline.config import Config
from yolo_baseline.data import encode,FullBatchSampler,splits
from yolo_baseline.loss import YOLOLoss
from yolo_baseline.evaluation import score_class,decode,evaluate
from yolo_baseline.runtime import save_checkpoint,restore_random,seed_all
from yolo_baseline.model import load_checkpoint


class BaselineTests(unittest.TestCase):
    def test_explicit_lr_warmup_and_decay_boundaries(self):
        cfg=Config()
        self.assertAlmostEqual(cfg.learning_rate(0),.0001)
        self.assertAlmostEqual(cfg.learning_rate(4),.001)
        self.assertAlmostEqual(cfg.learning_rate(74),.001)
        self.assertAlmostEqual(cfg.learning_rate(75),.0001)
        self.assertAlmostEqual(cfg.learning_rate(105),.00001)
    def test_one_cell_largest_object_and_edges(self):
        target,collision=encode([[1,1,9,9],[2,2,4,4],[99,99,100,100]],[14,0,1],100,100)
        self.assertEqual(collision,1);self.assertEqual(target[...,20].sum(),2)
        self.assertEqual(target[0,0,14],1);self.assertEqual(target[6,6,1],1)
        self.assertAlmostEqual(float(target[0,0,23]),.08)

    def test_responsible_box_and_background_gradients(self):
        target=torch.zeros(1,7,7,25);target[0,0,0,0]=1;target[0,0,0,20]=1
        target[0,0,0,21:25]=torch.tensor([.5,.5,.25,.25])
        pred=torch.zeros(1,7,7,30)
        pred[0,0,0,22:26]=torch.tensor([.5,.5,.5,.5])
        pred[0,0,0,26:30]=torch.tensor([2.,2.,.1,.1])
        pred.requires_grad_();loss=YOLOLoss()(pred,target);loss.backward()
        self.assertAlmostEqual(float(loss.detach()),2.,places=5) # class error 1 + responsible confidence error 1
        self.assertAlmostEqual(float(pred.grad[0,0,0,20]),-2.,places=5)
        self.assertEqual(float(pred.grad[0,0,0,21]),0.)
        self.assertTrue(torch.equal(pred.grad[0,0,0,26:30],torch.zeros(4)))

    def test_accumulation_matches_full_batch_loss_gradients(self):
        torch.manual_seed(4);pred=torch.randn(4,7,7,30,requires_grad=True);truth=torch.zeros(4,7,7,25)
        full=YOLOLoss()(pred,truth);full.backward();expected=pred.grad.clone();pred.grad=None
        for start in (0,2):(YOLOLoss()(pred[start:start+2],truth[start:start+2],reduction='sum')/4).backward()
        torch.testing.assert_close(pred.grad,expected)

    def test_padding_full_effective_batch_and_epoch_shuffle(self):
        sampler=FullBatchSampler(16551,4,64,2026,0);batches=list(sampler)
        flat=[i for batch in batches for i in batch]
        self.assertEqual(len(flat),16576);self.assertEqual(len(set(flat)),16551)
        self.assertTrue(all(len(batch)==4 for batch in batches))
        self.assertEqual(list(sampler),batches)
        self.assertNotEqual(list(FullBatchSampler(16551,4,64,2026,1))[0],batches[0])

    def test_AP_difficult_and_duplicate_detections(self):
        gt={'a':[{'class':0,'bbox':[1,1,10,10],'difficult':0}],
            'b':[{'class':0,'bbox':[1,1,10,10],'difficult':1}]}
        records=[('b',1,[1,1,10,10]),('a',.95,[50,50,60,60]),('a',.9,[1,1,10,10]),('a',.8,[1,1,10,10])]
        result=score_class(gt,records,0)
        self.assertEqual((result['TP'],result['FP']),(1,2));self.assertAlmostEqual(result['AP'],.5)
        self.assertEqual(score_class(gt,records,0),result)

    def test_decode_per_class_nms_and_sqrt_width(self):
        pred=torch.zeros(1,7,7,30);pred[0,3,3,0]=.9;pred[0,3,3,20:22]=torch.tensor([.9,.8])
        pred[0,3,3,22:]=torch.tensor([.5,.5,.5,.5,.5,.5,.5,.5])
        result=decode(pred)[0];self.assertEqual(len(result),1)
        self.assertAlmostEqual(result[0][2][2]-result[0][2][0],.25)

    def test_full_evaluation_pipeline_pixel_coordinates_and_all_class_AP(self):
        from yolo_baseline.config import CLASSES
        class PerfectDetector(torch.nn.Module):
            def __init__(self):super().__init__();self.cursor=0
            def forward(self,images):
                pred=torch.zeros(len(images),7,7,30)
                for i in range(len(images)):
                    pred[i,3,3,self.cursor+i]=1;pred[i,3,3,20]=1
                    pred[i,3,3,22:26]=torch.tensor([.5,.5,.5**.5,.5**.5])
                self.cursor+=len(images);return pred
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for cls,name in enumerate(CLASSES):
                image=root/f'{cls}.jpg';xml=root/f'{cls}.xml'
                Image.new('RGB',(100,100)).save(image)
                xml.write_text(f'<annotation><size><width>100</width><height>100</height></size><object><name>{name}</name><difficult>0</difficult><bndbox><xmin>26</xmin><ymin>26</ymin><xmax>75</xmax><ymax>75</ymax></bndbox></object></annotation>')
                rows.append((str(cls),image,xml))
            result=evaluate(PerfectDetector(),rows,torch.device('cpu'),Config(),root/'results',batch_size=4,workers=0,label='synthetic test only')
            self.assertAlmostEqual(result['mAP50_voc2007_11point'],1.)
            self.assertTrue(all(v['TP']==1 and v['FP']==0 for v in result['per_class'].values()))

    def test_checkpoint_restores_momentum_and_rng(self):
        seed_all(2026);model=torch.nn.Linear(2,1);optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
        model(torch.ones(1,2)).sum().backward();optimizer.step();optimizer.zero_grad()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pt';save_checkpoint(path,model,optimizer,Config(),1,2,-1,{}, {})
            reference=torch.rand(5);state=load_checkpoint(path)
            other=torch.nn.Linear(2,1);other.load_state_dict(state['model'])
            opt=torch.optim.SGD(other.parameters(),lr=.5,momentum=.9);opt.load_state_dict(state['optimizer'])
            restore_random(state['rng']);torch.testing.assert_close(torch.rand(5),reference)
            for m,o in ((model,optimizer),(other,opt)):
                m(torch.ones(1,2)).sum().backward();o.step()
            for a,b in zip(model.parameters(),other.parameters()):torch.testing.assert_close(a,b)

    def test_development_split_is_disjoint(self):
        from yolo_baseline.config import DATA
        if not (DATA/'VOC2012').exists():self.skipTest('VOC not installed')
        train,val,test=splits(DATA,'dev')
        self.assertEqual((len(train),len(val),len(test)),(14041,2510,4952))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA integration test')
    def test_interrupted_epoch_boundary_resume_matches_uninterrupted_training(self):
        from yolo_baseline.training import train
        from yolo_baseline import training
        class TinyModel(torch.nn.Module):
            def __init__(self,cfg,pretrained=None):
                super().__init__();self.linear=torch.nn.Linear(3,1470);self.dropout=torch.nn.Dropout(.5)
                self.initialization={'kind':'synthetic-test-only'}
            def forward(self,x):return self.linear(self.dropout(x.mean((2,3)))).reshape(-1,7,7,30)
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self,rows,training=False,seed=2026):
                self.rows=rows;self.annotations={'synthetic':[]};self.epoch=0
            def __len__(self):return len(self.rows)
            def __getitem__(self,i):return torch.full((3,2,2),i/64),torch.zeros(7,7,25),'synthetic',torch.tensor([2,2]),0
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=SimpleNamespace(micro_batch=4,epochs=2,seed=2026,split='full',device='cuda:0',
                                 data_root=root,pretrained=root/'unused',workers=0,eval_batch=4,iterations=1,resume=None,output=root/'whole')
            with patch.object(training,'YOLOv1',TinyModel),patch.object(training,'VOCDataset',SyntheticDataset),\
                 patch.object(training,'splits',return_value=(list(range(64)),[],[0])),\
                 patch.object(training,'evaluate',return_value={'mAP50_voc2007_11point':0.}),\
                 patch.object(training,'benchmark',return_value={'synthetic_test_only':True}):
                train(args)
                expected=load_checkpoint(args.output/'last.pt')['model']
                args.output=root/'resumed'
                real_save=training.save_checkpoint
                def interrupt_after_save(*values,**kwargs):
                    real_save(*values,**kwargs);raise RuntimeError('simulated disconnect after completed epoch')
                with patch.object(training,'save_checkpoint',side_effect=interrupt_after_save):
                    with self.assertRaisesRegex(RuntimeError,'simulated'):train(args)
                args.resume=args.output/'last.pt';train(args)
                actual=load_checkpoint(args.resume)['model']
                for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
