import logging
from typing import Optional

import numpy as np
import tensorflow as tf

from configs import DeTRModelConf
from model_base import ModelBase
from .blocks import TransformerBlock, TransformerDecoderBlock
from .backbones import get_available_backbones
from .layers import QuerySampler, PositionEmbedding

log = logging.getLogger(name=__name__)


class DeTR(ModelBase):
    """
        Contains methods to build Detection Transformer Model. Reference: https://arxiv.org/pdf/2005.12872.pdf
    """
    MODES = {'train', 'test'}

    def __init__(self, config: DeTRModelConf, mode: Optional[str] = None):
        assert mode in self.MODES, f"Unsupported mode: {mode}. Expect values: {self.MODES}"

        super().__init__(config=config, mode=mode)
        self.config = config

        self.num_classes = len(self.config.classes)
        self.input_size = (self.config.input_size.width, self.config.input_size.height)
        self.class_name_to_index = {
            class_name: i
            for i, class_name in enumerate(self.config.classes)
        }

    def set_weights(self, weights: dict, missing_names_ok: bool = True) -> None:
        super().set_weights(weights=weights, missing_names_ok=missing_names_ok)

    def _build(self) -> tf.keras.models.Model:

        input_shape = (*self.input_size[::-1], 3)
        input_layer = tf.keras.layers.Input(shape=input_shape, name='input')

        # Construct CNN backbone
        backbone = get_available_backbones()[self.config.backbone](input_shape=input_shape)
        feature_extractor = backbone()

        if self.config.freeze_backbone_batch_norm:
            # Freeze batch-norm layers (as in original paper)
            for layer in feature_extractor.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False

        x = feature_extractor(input_layer)  # (B, H0/32, W0/32, 2048)

        ############# START TRANSFORMER PART #################
        # Projection
        x = tf.keras.layers.Conv2D(filters=self.config.transformer_dim, kernel_size=(1, 1), name='projection')(x)
        x = tf.keras.layers.Reshape((-1, self.config.transformer_dim))(x)  # (B, H*W, model_dim)
        x = tf.keras.layers.Permute((2, 1))(x)  # (B, model_dim, H*W)

        # Add position embedding for Key & Query
        key = query = PositionEmbedding(sequence_length=self.config.transformer_dim, output_dim=x.shape[-1])(x)

        for _ in range(self.config.num_encoder_layers):
            x = TransformerBlock(value=x, key=key, query=query, projection_dim=self.config.transformer_dim,
                                 mlp_ratio=8, num_heads=8)
            key = query = PositionEmbedding(sequence_length=self.config.transformer_dim, output_dim=x.shape[-1])(x)
        encoder_output = x

        # Transformer Decoder
        x = QuerySampler(num_queries=self.config.num_queries, embedding_dim=self.config.transformer_dim)(x)
        key = query = PositionEmbedding(sequence_length=self.config.num_queries, output_dim=x.shape[-1])(x)

        for _ in range(self.config.num_decoder_layers):
            x = TransformerDecoderBlock(
                encoder_output=encoder_output,
                query=query,
                key=key,
                value=x,
                projection_dim=self.config.transformer_dim,
                mlp_ratio=8,
                num_heads=8
            )
            key = query = PositionEmbedding(sequence_length=self.config.num_queries, output_dim=x.shape[-1])(x)
        ############# END of TRANSFORMER PART #################

        # Feed-Forward Network (FFN) for classification
        outputs_class = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='classes')(x)

        # Feed-Forward Network (FFN) for bounding box regression
        for _ in range(2):
            x = tf.keras.layers.Dense(self.config.transformer_dim, activation='relu')(x)
        output_coordinates = tf.keras.layers.Dense(4, activation='sigmoid', name='coordinates')(x)

        outputs = [outputs_class, output_coordinates]
        if self.mode == 'train':
            outputs = tf.keras.layers.Concatenate(axis=-1, name='fused_output')([outputs_class, output_coordinates])

        return tf.keras.Model(input_layer, outputs)

    @staticmethod
    def normalize(data):
        result = np.empty_like(data, dtype=np.float32)
        np.subtract(data, 127.0, out=result)
        np.divide(result, 128.0, out=result)
        return result

    @staticmethod
    def restore_normalized(data):
        result = np.empty_like(data, dtype=np.uint8)
        np.add(data * 128, 127, out=result, casting="unsafe")
        return result
