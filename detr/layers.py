import numpy as np
import tensorflow as tf


class PositionEmbedding(tf.keras.layers.Layer):
    """
        Implements positional embedding for transformers. Reference: https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, sequence_length: int,
                 output_dim: int,
                 num_pos_features: int = 64,
                 temperature: int = 100,
                 normalize: bool = False,
                 scale: float = None, **kwargs):
        super().__init__(**kwargs)
        self.num_pos_features = num_pos_features
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale or 2 * np.pi
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.position_embedding_layer = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim,
            weights=[self._get_position_embedding_matrix(sequence_length=sequence_length, output_dim=output_dim)],
            trainable=False
        )

    def _get_position_embedding_matrix(self, sequence_length: int, output_dim: int, n: int = 10000) -> np.ndarray:
        result = np.zeros(shape=(sequence_length, output_dim), dtype=np.float32)
        for k in range(sequence_length):
            for i in np.arange(int(output_dim / 2)):
                denominator = np.power(n, 2 * i / output_dim)
                result[k, 2 * i] = np.sin(k / denominator)
                result[k, 2 * i + 1] = np.cos(k / denominator)
        return result

    def call(self, x):
        sequence = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_embedding = tf.expand_dims(self.position_embedding_layer(sequence), axis=0)
        position_embedding = tf.tile(position_embedding, [tf.shape(x)[0], 1, 1])
        return tf.keras.layers.Add()([x, position_embedding])

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            'sequence_length': self.sequence_length, 'output_dim': self.output_dim,
            'num_pos_features': self.num_pos_features, 'temperature': self.temperature,
            'normalize': self.normalize, 'scale': self.scale
        })
        return base_config


class QuerySampler(tf.keras.layers.Layer):
    """
        Implements query sampler for Detection Transformer. Paper: https://arxiv.org/pdf/2005.12872.pdf
    """
    def __init__(self, num_queries: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.embedding_dim = embedding_dim
        self.query_embedding = tf.keras.layers.Embedding(input_dim=num_queries, output_dim=embedding_dim)

    def call(self, x):
        queries_set = tf.range(start=0, limit=self.num_queries, delta=1)
        queries = tf.expand_dims(self.query_embedding(queries_set), axis=0)
        return tf.tile(queries, [tf.shape(x)[0], 1, 1])

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'num_queries': self.num_queries, 'embedding_dim': self.embedding_dim})
        return base_config
