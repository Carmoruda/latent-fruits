import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed (int): Random seed value.
    """

    # Set seed for various libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    # Set environment variable for Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in cuDNN if available
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
