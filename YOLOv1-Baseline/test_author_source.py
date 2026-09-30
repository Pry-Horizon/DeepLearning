import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from protocol import *
from build import generate
from prepare_voc import convert_label
from metrics import score_class,voc_iou
from weights import audit,conv_sizes,require_pretrained


class AuthorSourceTests(unittest.TestCase):
    def test_upstream_exact_commit_and_hashes(self):
        self.assertEqual(verify_upstream()['commit'],COMMIT)

    def test_paper_epochs_batch_and_boundaries(self):
        blocks=cfg_sections(paper_cfg());net=blocks[0][1]
        self.assertEqual(EPOCHS,135)
        self.assertEqual(net['batch'],'64')
        self.assertEqual(net['subdivisions'],'64')
        self.assertEqual(int(net['max_batches']),34913)
        self.assertEqual(LR_STEPS,(200,400,600,19396,27154))
        self.assertEqual(TOTAL_STEPS*BATCH-135*TRAIN_IMAGES,47)
        self.assertEqual(net['scales'],'2.5,2,2,.1,.1')
        self.assertNotIn('40000',paper_cfg())

    def test_architecture_is_entirely_upstream(self):
        original=cfg_sections((UPSTREAM/'cfg/yolo.cfg').read_text())
        actual=cfg_sections(paper_cfg())
        self.assertEqual(original[1:],actual[1:])
        self.assertEqual(sum(k=='convolutional' for k,v in actual),24)
        self.assertEqual([v['output'] for k,v in actual if k=='connected'],['4096','1470'])
        self.assertEqual([v['probability'] for k,v in actual if k=='dropout'],['.5'])
        self.assertFalse(any(v.get('batch_normalize')=='1' for k,v in actual))
        self.assertEqual(actual[-1][1]['num'],'2')

    def test_build_keeps_training_math_unchanged(self):
        generate()
        core=('detection_layer.c','network.c','network_kernels.cu','data.c','image.c',
              'crop_layer.c','crop_layer_kernels.cu','connected_layer.c','convolutional_layer.c',
              'convolutional_kernels.cu','parser.c')
        for name in core:self.assertEqual((UPSTREAM/'src'/name).read_bytes(),(BUILD/'src'/name).read_bytes())
        self.assertEqual((BUILD/'cfg/paper.cfg').read_text(),paper_cfg())
        self.assertIn('data/voc.2007.test',(BUILD/'src/yolo.c').read_text())

    def test_label_equations_and_difficult(self):
        xml='<annotation><size><width>100</width><height>200</height></size>'
        for difficult in (0,1):
            xml+=f'<object><name>person</name><difficult>{difficult}</difficult><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>60</ymax></bndbox></object>'
        xml+='</annotation>'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.xml';path.write_text(xml);lines=convert_label(path).splitlines()
        self.assertEqual(len(lines),1)
        self.assertEqual(lines[0],'14 0.2 0.2 0.2 0.2')

    def test_AP_difficult_duplicate_false_positive(self):
        gt={'a':[{'class':0,'bbox':[1,1,10,10],'difficult':0}],
            'b':[{'class':0,'bbox':[1,1,10,10],'difficult':1}]}
        records=[('b',1.,[1,1,10,10]),('a',.95,[50,50,60,60]),
                 ('a',.9,[1,1,10,10]),('a',.8,[1,1,10,10])]
        actual=score_class(gt,records,0)
        self.assertEqual((actual['TP'],actual['FP']),(1,2))
        self.assertAlmostEqual(actual['AP'],.5)
        self.assertEqual(voc_iou([1,1,1,1],[[1,1,1,1]])[0],1.)
        self.assertEqual(score_class(gt,records,0),actual)

    def test_missing_backbone_layers_are_not_silently_random(self):
        self.assertEqual(16+sum(conv_sizes()),89612560)
        path=ROOT/'data/pretrained/extraction.conv.weights'
        if not path.exists():self.skipTest('Mirror not downloaded')
        report=audit(path)
        self.assertEqual(report['complete_convolutions'],16)
        self.assertFalse(report['complete_first20'])
        with self.assertRaises(ValueError):require_pretrained(path)


if __name__=='__main__':unittest.main()
