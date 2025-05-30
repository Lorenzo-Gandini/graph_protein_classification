# source/config.py
import dataclasses
from typing import Optional
import os

@dataclasses.dataclass
class ModelConfig:
    test_path:  Optional[str] = None
    train_path: Optional[str] = None
    pretrain_paths: Optional[str] = None
    batch_size: int = 24
    hidden_dim: int = 128
    latent_dim: int = 8
    num_classes: int = 6
    epochs: int = 150
    learning_rate: float = 0.00005
    num_cycles: int = 5
    warmup: int = 5
    early_stopping_patience: int = 20
    loss_type: str = "ce"
    ub_lr: int = 1
    @property
    def folder_name(self) -> str:
        """Extract folder name (A, B, C, or D) from test path"""
        files = self.train_path if self.train_path is not None else self.test_path
        db = ''
        for file in files.split(' '):
            db += os.path.basename(os.path.dirname(file))
        return db
    
