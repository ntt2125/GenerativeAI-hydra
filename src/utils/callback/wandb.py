import os
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

class WandbCallback(Callback):
    def __init__(self):
        super().__init__()