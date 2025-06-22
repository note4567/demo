import tensorflow as tf  # NOQA # type: ignore
from typing import List
from common_layer import (
    FeedForwardNetwork,
    ResidualNormalizationWrapper,
    LayerNormalization,
)
from embedding import TokenEmbedding, AddPositionalEncoding
from attention import MultiheadAttention, SelfAttention

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

PAD_ID = 0


class Transformer(tf.keras.models.Model):
    def __init__(
        self,
        vocab_size: int,
        hopping_num: int,
        head_num: int,
        hidden_dim: int,
        dropout_rate: float,
        max_length: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size: int = vocab_size
        self.hopping_num: int = hopping_num
        self.head_num: int = head_num
        self.hidden_dim: int = hidden_dim
        self.dropout_rate: float = dropout_rate
        self.max_length: int = max_length

        self.encoder = Encoder(
            vocab_size=vocab_size,
            hopping_num=hopping_num,
            head_num=head_num,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            max_length=max_length,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            hopping_num=hopping_num,
            head_num=head_num,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            max_length=max_length,
        )

    @classmethod
    def build_from_config(cls, conf):
        # コンフィグからTransformerモデルを再構築するためのロジック
        return cls(**conf)

    def get_config(self):
        conf = super(Transformer, self).get_config()
        conf.update(
            {
                "vocab_size": self.vocab_size,
                "hopping_num": self.hopping_num,
                "head_num": self.head_num,
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate,
                "max_length": self.max_length,
            }
        )
        return conf

    @classmethod
    def from_config(cls, conf):
        return cls(
            vocab_size=conf["vocab_size"],
            hopping_num=conf["hopping_num"],
            head_num=conf["head_num"],
            hidden_dim=conf["hidden_dim"],
            dropout_rate=conf["dropout_rate"],
            max_length=conf["max_length"],
        )

    @tf.function
    def call(self, inputs, training=None):
        training = False
        encoder_input = inputs["encoder_input"]
        decoder_input = inputs["decoder_input"]

        logit = self.transform(
            encoder_input=encoder_input, decoder_input=decoder_input, training=training
        )

        return logit

    def transform(
        self, encoder_input: tf.Tensor, decoder_input: tf.Tensor, training: bool
    ) -> tf.Tensor:
        enc_attention_mask = self._create_enc_attention_mask(encoder_input)
        dec_self_attention_mask = self._create_dec_self_attention_mask(decoder_input)

        encoder_output = self.encoder(
            encoder_input,
            self_attention_mask=enc_attention_mask,
            training=training,
        )
        decoder_output = self.decoder(
            decoder_input,
            encoder_output,
            self_attention_mask=dec_self_attention_mask,
            enc_dec_attention_mask=enc_attention_mask,
            training=training,
        )
        return decoder_output

    def _create_enc_attention_mask(self, encoder_input: tf.Tensor):
        with tf.name_scope("enc_attention_mask"):
            batch_size, length = tf.unstack(tf.shape(encoder_input))
            pad_array = tf.equal(encoder_input, PAD_ID)

            return tf.reshape(pad_array, [batch_size, 1, 1, length])

    def _create_dec_self_attention_mask(self, decoder_input: tf.Tensor):
        with tf.name_scope("dec_self_attention_mask"):
            batch_size, length = tf.unstack(tf.shape(decoder_input))
            pad_array = tf.equal(decoder_input, PAD_ID)
            pad_array = tf.reshape(pad_array, [batch_size, 1, 1, length])

            autoregression_array = tf.logical_not(
                tf.compat.v1.matrix_band_part(
                    tf.ones([length, length], dtype=tf.bool), -1, 0
                )
            )  # 下三角が False
            autoregression_array = tf.reshape(
                autoregression_array, [1, 1, length, length]
            )

            return tf.logical_or(pad_array, autoregression_array)


class Encoder(tf.keras.models.Model):
    def __init__(
        self,
        vocab_size: int,
        hopping_num: int,
        head_num: int,
        hidden_dim: int,
        dropout_rate: float,
        max_length: int,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.hopping_num: int = hopping_num
        self.head_num: int = head_num
        self.hidden_dim: int = hidden_dim
        self.dropout_rate: float = dropout_rate
        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list: List[List[tf.keras.models.Model]] = []
        for _ in range(hopping_num):
            attention_layer = SelfAttention(
                hidden_dim, head_num, dropout_rate, name="self_attention"
            )
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name="ffn")
            self.attention_block_list.append(
                [
                    ResidualNormalizationWrapper(
                        attention_layer, dropout_rate, name="self_attention_wrapper"
                    ),
                    ResidualNormalizationWrapper(
                        ffn_layer, dropout_rate, name="ffn_wrapper"
                    ),
                ]
            )

        self.output_normalization = LayerNormalization()

    def call(
        self,
        input: tf.Tensor,
        self_attention_mask: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:

        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_embedding(embedded_input)

        training = bool(training)
        query = self.input_dropout_layer(embedded_input, training=training)

        for i, layers in enumerate(self.attention_block_list):
            attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f"hopping_{i}"):
                query = attention_layer(
                    query, attention_mask=self_attention_mask, training=training
                )
                query = ffn_layer(query, training=training)

        return self.output_normalization(query)


class Decoder(tf.keras.models.Model):
    def __init__(
        self,
        vocab_size: int,
        hopping_num: int,
        head_num: int,
        hidden_dim: int,
        dropout_rate: float,
        max_length: int,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list: List[List[tf.keras.models.Model]] = []
        for _ in range(hopping_num):
            self_attention_layer = SelfAttention(
                hidden_dim, head_num, dropout_rate, name="self_attention"
            )

            enc_dec_attention_layer = MultiheadAttention(
                hidden_dim, head_num, dropout_rate, name="enc_dec_attention"
            )

            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name="ffn")
            self.attention_block_list.append(
                [
                    ResidualNormalizationWrapper(
                        self_attention_layer,
                        dropout_rate,
                        name="self_attention_wrapper",
                    ),
                    ResidualNormalizationWrapper(
                        enc_dec_attention_layer,
                        dropout_rate,
                        name="enc_dec_attention_wrapper",
                    ),
                    ResidualNormalizationWrapper(
                        ffn_layer, dropout_rate, name="ffn_wrapper"
                    ),
                ]
            )
        self.output_normalization = LayerNormalization()
        self.output_dense_layer = tf.keras.layers.Dense(vocab_size, use_bias=False)

    def call(
        self,
        input: tf.Tensor,
        encoder_output: tf.Tensor,
        self_attention_mask: tf.Tensor,
        enc_dec_attention_mask: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:

        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.input_dropout_layer(embedded_input, training=training)

        for i, layers in enumerate(self.attention_block_list):
            self_attention_layer, enc_dec_attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f"hopping_{i}"):
                query = self_attention_layer(
                    query, attention_mask=self_attention_mask, training=training
                )

                query = enc_dec_attention_layer(
                    query,
                    memory=encoder_output,
                    attention_mask=enc_dec_attention_mask,
                    training=training,
                )

                query = ffn_layer(query, training=training)
        query = self.output_normalization(query)

        return self.output_dense_layer(query)
