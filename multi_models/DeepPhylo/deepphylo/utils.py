import numpy as np
import torch
import random

# 设置随机种子
def set_seed(seed_value=42):
    # 设置PyTorch的随机种子
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    # 设置NumPy的随机种子
    np.random.seed(seed_value)

    # 设置Python的随机种子
    random.seed(seed_value)

    # 使结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False