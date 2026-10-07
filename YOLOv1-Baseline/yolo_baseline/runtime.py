import os
import platform
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torchvision
from .config import write_json


def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def random_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_random(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path,model,optimizer,cfg,epoch,updates,best_ap,initialization,source_hashes):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state={'schema':1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
           'config':cfg.dictionary(),'completed_epochs':epoch,'updates':updates,'best_validation_ap':best_ap,
           'rng':random_state(),'initialization':initialization,'source_hashes':source_hashes,'environment':environment()}
    temporary=path.with_suffix('.pt.part');torch.save(state,temporary);temporary.replace(path)


def environment():
    return {'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,
            'numpy':np.__version__,'platform':platform.platform(),'cuda_runtime':torch.version.cuda,
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
