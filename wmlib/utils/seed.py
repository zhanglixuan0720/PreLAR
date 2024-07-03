import torch
import numpy as np
import random

def set_seed(seed):
    """Set the seed for all random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # no apparent impact on speed
    torch.backends.cudnn.benchmark = True  # faster, increases memory though.., no impact on seed
    return seed