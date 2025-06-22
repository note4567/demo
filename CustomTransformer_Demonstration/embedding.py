from typing import Any
import tensorflow as tf  # NOQA # type: ignore
import numpy as np  # NOQA # type: ignore
import math

PAD_ID = 0


class TokenEmbedding(tf.keras.layers.Layer):
    """
    トークン列を Embedded Vector 列に変換
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dtype=tf.float32,
        *args,
        **kwargs,  # NOQA
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size: int = vocab_size
        self.embedding_dim: int = embedding_dim
        self.dtype_: Any = dtype

    def build(self, input_shape: tf.TensorShape) -> None:
        self.lookup_table = self.add_variable(
            name="token_embedding",
            shape=[self.vocab_size, self.embedding_dim],
            dtype=self.dtype_,
            initializer=tf.random_normal_initializer(
                0.0, self.embedding_dim**-0.5
            ),  # NOQA
        )
        super().build(input_shape)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        mask: tf.Tensor = tf.cast(tf.not_equal(input, PAD_ID), tf.float32)
        embedding = tf.nn.embedding_lookup(self.lookup_table, input)
        embedding *= tf.expand_dims(mask, -1)  # 元々 PAD だった部分を0にする
        return embedding * self.embedding_dim**0.5


class AddPositionalEncoding(tf.keras.layers.Layer):
    """
    入力テンソルに対し、位置の情報を付与して返すレイヤー
    PE_{pos, 2i}   = sin(pos / 10000^{2i / d_model})
    PE_{pos, 2i+1} = cos(pos / 10000^{2i / d_model})
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        inputs: [batch_size, length, hidden_dim]
        これは embedding 処理が終了した後の行列
        """
        fl_type: tf.float32 = inputs.dtype
        batch_size, max_length, depth = tf.unstack(tf.shape(inputs))
        depth_counter: tf.Tensor = tf.range(depth) // 2 * 2
        depth_matrix: tf.Tensor = tf.tile(
            tf.expand_dims(depth_counter, 0), [max_length, 1]
        )

        depth_matrix = tf.pow(10000.0, tf.cast(depth_matrix / depth, fl_type))
        np.set_printoptions(suppress=True, precision=5)
        phase: tf.Tensor = (
            tf.cast(tf.range(depth) % 2, fl_type) * math.pi / 2
        )

        phase_matrix: tf.Tensor = tf.tile(
            tf.expand_dims(phase, 0), [max_length, 1]
        )
        pos_counter: tf.Tensor = tf.range(max_length)
        pos_matrix: tf.Tensor = tf.cast(
            tf.tile(tf.expand_dims(pos_counter, 1), [1, depth]), fl_type
        )
        positional_encoding: tf.Tensor = tf.sin(
            pos_matrix / depth_matrix + phase_matrix
        )
        positional_encoding = tf.tile(
            tf.expand_dims(positional_encoding, 0), [batch_size, 1, 1]
        )

        return inputs + positional_encoding
