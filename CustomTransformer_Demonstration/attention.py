from typing import Any
from common import config  # NOQA

import tensorflow as tf  # NOQA # type: ignore
import numpy as np  # NOQA # type: ignore


class MultiheadAttention(tf.keras.models.Model):
    """
    Multi-head Attention のモデル
    """

    def __init__(
                self,
                hidden_dim: int,
                head_num: int,
                dropout_rate: float,
                *args,
                **kwargs
    ) -> None:  # fmt: skip

        """
        :param hidden_dim:   隠れ層及び出力の次元
                             head_num の倍数である必要がある。
        :param head_num:     ヘッドの数
        :param dropout_rate: ドロップアウトする確率
        """

        super().__init__(*args, **kwargs)
        self.hidden_dim: int = hidden_dim
        self.head_num: int = head_num
        self.dropout_rate: float = dropout_rate

        self.q_dense_layer: Any = tf.keras.layers.Dense(
            hidden_dim, use_bias=False, name="q_dense_layer"
        )
        self.k_dense_layer: Any = tf.keras.layers.Dense(
            hidden_dim, use_bias=False, name="k_dense_layer"
        )
        self.v_dense_layer: Any = tf.keras.layers.Dense(
            hidden_dim, use_bias=False, name="v_dense_layer"
        )
        self.output_dense_layer: Any = tf.keras.layers.Dense(
            hidden_dim, use_bias=False, name="output_dense_layer"
        )
        self.attention_dropout_layer: Any = tf.keras.layers.Dropout(
            dropout_rate
        )  # fmt: skip

    def call(
        self,
        input: tf.Tensor,
        memory: tf.Tensor,
        attention_mask: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:
        """
        モデルの実行を行う。
        :param input:  query のテンソル
        :param memory: query に情報を与える memory のテンソル
        :param attention_mask: attention weight に適用される mask
            shape = [batch_size, 1, q_length, k_length]
            pad 等無視する部分が True となるようなものを指定する。
        :param training: 学習時か推論時かのフラグ
        """

        q: Any = self.q_dense_layer(input)
        k: Any = self.k_dense_layer(memory)
        v: Any = self.v_dense_layer(memory)

        q = self._split_head(q)
        k = self._split_head(k)
        v = self._split_head(v)

        depth: int = self.hidden_dim // self.head_num
        # for scaled dot production
        q *= depth**-0.5

        # ここで q と k の内積を取ることで query と key の関連度の計算
        logit: tf.Tensor = tf.matmul(q, k, transpose_b=True)
        logit += tf.cast(attention_mask, dtype=tf.float32) * input.dtype.min

        # softmax を取ることで正規化する
        attention_weight: Any = tf.nn.softmax(logit, name="attention_weight")
        attention_weight: Any = self.attention_dropout_layer(
            attention_weight, training=training
        )
        attention_output: Any = tf.matmul(attention_weight, v)
        attention_output = self._combine_head(attention_output)

        return self.output_dense_layer(attention_output)

    def _split_head(self, x: tf.Tensor) -> tf.Tensor:
        """
        入力の tensor の hidden_dim の次元をいくつかのヘッドに分割

        入力 shape: [batch_size, length, hidden_dim] の時
        出力 shape: [batch_size, head_num, length, hidden_dim//head_num]
        """
        with tf.name_scope("split_head"):
            # 入力テンソル x の形状を取得し、その各次元を個別の変数に分解
            (
                batch_size,
                length,
                hidden_dim,
            ) = tf.unstack(tf.shape(x))

            x = tf.reshape(
                x,
                [
                    batch_size,
                    length,
                    self.head_num,
                    self.hidden_dim // self.head_num,
                ],  # NOQA
            )

            return tf.transpose(x, [0, 2, 1, 3])

    def _combine_head(self, x: tf.Tensor) -> tf.Tensor:
        """
        入力の tensor の各ヘッドを結合 _split_head の逆変換

        入力 shape: [batch_size, head_num, length, hidden_dim//head_num]
        出力 shape: [batch_size, length, hidden_dim]
        """
        with tf.name_scope("combine_head"):
            batch_size, _, length, _ = tf.unstack(tf.shape(x))
            x = tf.transpose(x, [0, 2, 1, 3])

            return tf.reshape(x, [batch_size, length, self.hidden_dim])


class SelfAttention(MultiheadAttention):
    def call(  # type: ignore
        self,
        input: tf.Tensor,
        attention_mask: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:

        return super().call(
            input=input,
            memory=input,
            attention_mask=attention_mask,
            training=training,
        )
