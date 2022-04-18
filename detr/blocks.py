import tensorflow as tf

from .layers import PositionEmbedding


def MLPBlock(x: tf.keras.layers.Layer, hidden_units: int, dropout_rate: float) -> tf.keras.layers.Layer:
    """
        Represents two stacked feed-forward networks with dropouts.
        Args:
            x: input layer,
            hidden_units: number of units for feed-forward networks,
            dropout_rate: rate for Dropout (from 0 to 1)
        Returns:
            output layer
    """
    input_channels = x.shape[-1]

    x = tf.keras.layers.Dense(hidden_units, activation=tf.nn.gelu)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Dense(input_channels)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def TransformerBlock(
        value: tf.keras.layers.Layer,
        key: tf.keras.layers.Layer,
        query: tf.keras.layers.Layer,
        projection_dim: int,
        mlp_ratio: float,
        num_heads: int,
        drop_prob: float = 0.
) -> tf.keras.layers.Layer:
    """
        Represents Transformer encoding block.
        Args:
            value: value layer,
            key: key layer,
            query: query layer,
            projection_dim: size of each attention head for query and key,
            mlp_ratio: ratio for mlp block (num_hidden_units = projection_dim * mlp_ratio),
            num_heads: number of heads in multi-head attention,
            drop_prob: dropout probability rate
        Returns:
            output layer
    """
    skip_connection_first = value

    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim,
        dropout=drop_prob
    )(query=query, value=value, key=key)

    skip_connection_second = tf.keras.layers.Add()([x, skip_connection_first])
    x = tf.keras.layers.LayerNormalization()(skip_connection_second)

    x = MLPBlock(x=x, hidden_units=int(projection_dim * mlp_ratio), dropout_rate=drop_prob)
    return tf.keras.layers.Add()([x, skip_connection_second])


def TransformerDecoderBlock(
        encoder_output: tf.keras.layers.Layer,
        query: tf.keras.layers.Layer,
        key: tf.keras.layers.Layer,
        value: tf.keras.layers.Layer,
        projection_dim: int,
        mlp_ratio: float,
        num_heads: int,
        drop_prob: float = 0.
) -> tf.keras.layers.Layer:
    """
        Implements Transformer decoding block.
        Args:
            encoder_output: output layer from encoder,
            query: query layer,
            key: key layer,
            value: value layer,
            projection_dim: size of each attention head for query and key,
            mlp_ratio: ratio for mlp block (num_hidden_units = projection_dim * mlp_ratio),
            num_heads: number of heads in multi-head attention,
            drop_prob: dropout probability rate
        Returns:
            output layer
    """
    skip_connection_first = value

    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim,
        dropout=drop_prob
    )(query=query, value=value, key=key)

    skip_connection_second = tf.keras.layers.Add()([x, skip_connection_first])
    x = tf.keras.layers.LayerNormalization()(skip_connection_second)

    encoder_output_key = PositionEmbedding(
        sequence_length=encoder_output.shape[1],
        output_dim=encoder_output.shape[-1]
    )(encoder_output)

    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim,
        dropout=drop_prob
    )(query=x, value=encoder_output, key=encoder_output_key)

    skip_connection_last = tf.keras.layers.Add()([x, skip_connection_second])
    x = tf.keras.layers.LayerNormalization()(skip_connection_last)

    x = MLPBlock(x=x, hidden_units=int(projection_dim * mlp_ratio), dropout_rate=drop_prob)
    return tf.keras.layers.Add()([x, skip_connection_last])
