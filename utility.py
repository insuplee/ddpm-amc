import torch
import random
import numpy as np
from datetime import datetime


def fix_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_time_info():
    now_time = datetime.now().strftime("%Y%m%d_%H%M%S").split('_')
    day, hms = now_time[0], now_time[1]
    return day, hms