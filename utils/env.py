import torch
import numpy as np
import random as rn
from pytorch_lightning import Trainer, seed_everything

def set_seed(seed):
    rn.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True