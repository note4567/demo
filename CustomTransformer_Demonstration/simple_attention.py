from typing import Any
from common import config  # NOQA

import tensorflow as tf  # NOQA # type: ignore


class SimpleAttention(tf.keras.models.Model):
    """
    [Demo] Multi-head ではない単純な Attention
    """

    def __init__(self, depth: int, *args, **kwargs) -> None:
        """
        :param depth: 隠れ層及び出力の次元
        """
        super().__init__(*args, **kwargs)
        self.depth: int = depth

        self.q_dense_layer: Any = tf.keras.layers.Dense(
            depth, use_bias=False, name="q_dense_layer"
        )
        self.k_dense_layer: Any = tf.keras.layers.Dense(
            depth, use_bias=False, name="k_dense_layer"
        )
        self.v_dense_layer: Any = tf.keras.layers.Dense(
            depth, use_bias=False, name="v_dense_layer"
        )
        self.output_dense_layer: Any = tf.keras.layers.Dense(
            depth, use_bias=False, name="output_dense_layer"
        )

    def call(self, input: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        """
        モデルの実行を行う
        :param input: query のテンソル
        :param memory: query に情報を与える memory のテンソル
        """
        # Dense レイヤーを通すことで、入力データは新しい特徴空間に変換される
        q: Any = self.q_dense_layer(input)  # [batch_size, q_length, depth]
        k: Any = self.k_dense_layer(memory)  # [batch_size, m_length, depth]
        v: Any = self.v_dense_layer(memory)

        logit: Any = tf.matmul(q, k, transpose_b=True)
        attention_weight: Any = tf.nn.softmax(logit, name="attention_weight")
        attention_output: Any = tf.matmul(attention_weight, v)

        return self.output_dense_layer(attention_output)


# サイズを指定
batch_size = 2
q_length = 5
depth = 4


# TensorFlowでランダムな値で初期化されたテンソルを作成
input_query: tf.Tensor = tf.random.uniform((batch_size, q_length, depth))
input_memory: tf.Tensor = tf.random.uniform((batch_size, q_length, depth))

at = SimpleAttention(depth)
print(at.call(input_query, input_memory))
