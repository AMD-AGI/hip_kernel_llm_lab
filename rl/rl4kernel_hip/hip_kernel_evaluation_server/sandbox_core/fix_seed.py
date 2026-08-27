# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import argparse
import os
import sys
from random import randint
from glob import glob
import importlib.util
import random
import numpy as np
import torch
from collections import namedtuple

torch.set_printoptions(profile="full")

def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across multiple libraries and configure PyTorch for deterministic behavior.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set seed for PyTorch on all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(seed=42)