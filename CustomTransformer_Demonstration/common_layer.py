import tensorflow as tf  # NOQA # type: ignore


class FeedForwardNetwork(tf.keras.models.Model):
    """
    Transformer 用の Position-wise Feedforward Neural Network
    """

    def __init__(
        self, hidden_dim: int, dropout_rate: float, *args, **kwargs
    ) -> None:  # NOQA

        super().__init__(*args, **kwargs)
        self.hidden_dim: int = hidden_dim
        self.dropout_rate: float = dropout_rate

        self.filter_dense_layer = tf.keras.layers.Dense(
            hidden_dim * 4,
            use_bias=True,
            activation=tf.nn.relu,
            name="filter_layer",  # NOQA
        )
        self.output_dense_layer = tf.keras.layers.Dense(
            hidden_dim, use_bias=True, name="output_layer"
        )
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input: tf.Tensor, training: bool) -> tf.Tensor:
        """
        FeedForwardNetwork を適用
        :param input: shape = [batch_size, length, hidden_dim]
        :return: shape = [batch_size, length, hidden_dim]
        """
        tensor = self.filter_dense_layer(input)
        tensor = self.dropout_layer(tensor, training=training)

        return self.output_dense_layer(tensor)


class ResidualNormalizationWrapper(tf.keras.models.Model):
    """
    - Layer Normalization
    - Dropout
    - Residual Connection
    """

    def __init__(
        self, layer: tf.keras.layers.Layer, dropout_rate: float, *args, **kwargs  # NOQA
    ) -> None:

        super().__init__(*args, **kwargs)
        self.layer = layer
        self.layer_normalization = LayerNormalization()
        self.dropout_layer: tf.keras.layers.Dropout = tf.keras.layers.Dropout(
            dropout_rate
        )

    def build(self, input_shape) -> None:
        super().build(input_shape)

    def call(
        self, input: tf.Tensor, training: bool, *args, **kwargs
    ) -> tf.Tensor:  # NOQA

        tensor: tf.Tensor = self.layer_normalization(input)
        tensor = self.layer(tensor, training=training, *args, **kwargs)
        tensor = self.dropout_layer(tensor, training=training)

        return input + tensor


class LayerNormalization(tf.keras.layers.Layer):
    """
    [レイヤーノーマライゼーション]
    レイヤーの出力が平均 bias, 標準偏差 scale になるように調整
    TensorFlow の Layer クラスを継承して、カスタムの Layer Normalization レイヤーを定義
    build() → call() の順で呼ばれる
    """

    def build(self, input_shape: tf.TensorShape) -> None:
        hidden_dim: int = input_shape[-1]
        self.scale: tf.KerasVariable = self.add_weight(
            name="layer_norm_scale",
            shape=[hidden_dim],
            initializer=tf.ones_initializer(),
        )
        self.bias: tf.KerasVariable = self.add_weight(
            name="layer_norm_bias",
            shape=[hidden_dim],
            initializer=tf.zeros_initializer(),
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor, epsilon: float = 1e-6) -> tf.Tensor:
        mean: tf.Tensor = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance: tf.Tensor = tf.reduce_mean(
            tf.square(x - mean), axis=[-1], keepdims=True
        )
        norm_x: tf.Tensor = (x - mean) / tf.sqrt(variance + epsilon)

        return norm_x * self.scale + self.bias
