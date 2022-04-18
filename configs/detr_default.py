from pathlib import Path
from typing import List
from dataclasses import dataclass, field

from .base import DataConfBase, SizeConfBase, OptimizerConfBase, ModelConfBase, TrainConfBase


@dataclass
class DeTRModelConf(ModelConfBase):
    input_color_mode: str = "RGB"
    input_size: SizeConfBase = SizeConfBase(width=512, height=512)
    classes: List[str] = field(default_factory=lambda: [
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus'
    ])
    backbone: str = "ResNet50"
    freeze_backbone_batch_norm: bool = False
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    transformer_dim: int = 256
    num_queries: int = 100


@dataclass
class DeTRTrainConf(TrainConfBase):
    aug: bool = False
    train_datasets: List[str] = field(default_factory=lambda: [
        'my_dataset_name_1',
        'my_dataset_name_2'
    ])
    train_metrics: List[str] = field(default_factory=lambda: [])
    test_datasets: List[str] = field(default_factory=lambda: [
        'my_test_dataset_name_1',
        'my_test_dataset_name_2',
    ])
    test_metrics: List[str] = field(default_factory=lambda: [
        'mAP @ [IoU = 0.75]',
        'mAP @ [IoU = 0.5]',
    ])
    loss: str = 'detr_loss'
    optimizer: OptimizerConfBase = OptimizerConfBase(type='adam', learning_rate=1e-4)
    batch_size: int = 16
    epochs: int = 300


@dataclass
class DeTRConf:
    data: DataConfBase = DataConfBase(path=Path('my/path/to/data/dir'))
    model: DeTRModelConf = DeTRModelConf()
    train: DeTRTrainConf = DeTRTrainConf()
