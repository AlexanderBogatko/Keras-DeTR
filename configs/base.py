from pathlib import Path
from typing import List
from dataclasses import dataclass


@dataclass
class DataConfBase:
    path: Path


@dataclass
class SizeConfBase:
    width: int
    height: int


@dataclass
class ModelConfBase:
    input_color_mode: str
    input_size: SizeConfBase


@dataclass
class OptimizerConfBase:
    type: str
    learning_rate: float


@dataclass
class TrainConfBase:
    aug: bool
    train_datasets: List[str]
    train_metrics: List[str]
    test_datasets: List[str]
    test_metrics: List[str]
    loss: str
    optimizer: OptimizerConfBase
    batch_size: int
    epochs: int
