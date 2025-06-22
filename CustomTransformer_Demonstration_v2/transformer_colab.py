from tensorflow.keras import layers, saving  # type: ignore
import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore
from transformers import BertJapaneseTokenizer
from transformers import BertTokenizer
from tensorflow.keras.callbacks import Callback  # type: ignore

# ハイパーパラメータ
num_layers = 4  # エンコーダーとデコーダーのレイヤ数
d_model = 40  # モデルの次元
num_heads = 4  # ヘッド数
dff = 500  # フィードフォワードネットワークの次元
max_position_encoding = 34  # 最大位置エンコーディングの長さ
dropout_rate = 0.15
batch_size = 3

# サンプルデータ
input_text: list[str] = ["This is a test.", "How are you?", "Let's train the model."]
target_text: list[str] = [
    "これはテストです。",
    "お元気ですか？",
    "モデルをトレーニングしましょう。",
]
target_text = ["[CLS] " + text + " [SEP]" for text in target_text]

# トークナイザーの準備
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
tokenizer_en = BertTokenizer.from_pretrained("bert-base-uncased")

# 語彙数の算定
input_text = [tokenizer_en.tokenize(text) for text in input_text]
target_text = [tokenizer.tokenize(text) for text in target_text]
all_tokens = set(token for sublist in input_text + target_text for token in sublist)
vocab_size: int = len(all_tokens) + 1


# 符号化とパディング
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(input_text + target_text)

input_sequences = tokenizer.texts_to_sequences(input_text)
target_sequences = tokenizer.texts_to_sequences(target_text)

# 入力とターゲットの最大長を取得
max_length = max(
    max(len(seq) for seq in input_sequences),  # 入力シーケンスの最大長
    max(len(seq) for seq in target_sequences),  # ターゲットシーケンスの最大長
)

# 最大長でパディングを統一
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    input_sequences, maxlen=max_length, padding="post"
)
target_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    target_sequences, maxlen=max_length, padding="post"
)

# TensorFlow に変換
input_tensor = tf.constant(input_sequences, dtype=tf.int32)
target_tensor = tf.constant(target_sequences, dtype=tf.int32)


# Position Encoding の定義
def get_position_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    return tf.constant(pos_encoding, dtype=tf.float32)


# デコーダーの自己アテンション用マスク作成関数
def create_look_ahead_mask(size):
    """自己アテンションのマスクを作成（上三角行列）"""
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


@saving.register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        dff,
        num_layers,
        max_pos_encoding,
        dropout_rate,
        **kwargs,
    ):
        super(Transformer, self).__init__(**kwargs)

        # パラメータの設定
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.max_pos_encoding = max_pos_encoding
        self.dropout_rate = dropout_rate

        # 名前付きEmbeddingレイヤー
        self.embedding = layers.Embedding(vocab_size, d_model, name="embedding")

        # 位置エンコーディング
        self.pos_encoding = get_position_encoding(max_pos_encoding, d_model)

        # エンコーダ層に名前を付けてリスト化
        self.encoder_layers = [
            self.create_encoder_layer(d_model, num_heads, dff, dropout_rate, index)
            for index in range(num_layers)
        ]

        # デコーダ層に名前を付けてリスト化
        self.decoder_layers = [
            self.create_decoder_layer(d_model, num_heads, dff, dropout_rate, index)
            for index in range(num_layers)
        ]

        # 最終出力層に名前を付ける
        self.final_layer = layers.Dense(vocab_size, name="final_layer")

    def get_config(self):
        """推論時に学習後のモデルを使用する際に必要"""

        config = super(Transformer, self).get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "num_layers": self.num_layers,
                "max_pos_encoding": self.max_pos_encoding,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def create_encoder_layer(self, d_model, num_heads, dff, dropout_rate, index):
        # エンコーダの入力に名前を付ける
        inputs = layers.Input(shape=(None, d_model), name=f"encoder_input_{index}")

        embedding = layers.Lambda(
            lambda x: x + self.pos_encoding[: tf.shape(x)[1], :],
            name=f"encoder_positional_encoding_{index}",
        )(inputs)

        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, name=f"encoder_attention_{index}"
        )(embedding, embedding)

        attention_output = layers.Dropout(
            dropout_rate, name=f"encoder_dropout_{index}"
        )(attention_output)

        attention_output = layers.LayerNormalization(
            axis=-1, name=f"encoder_layer_norm_{index}"
        )(embedding + attention_output)

        ffn_output = layers.Dense(
            dff, activation="relu", name=f"encoder_ffn_1_{index}"
        )(attention_output)

        ffn_output = layers.Dense(d_model, name=f"encoder_ffn_2_{index}")(ffn_output)
        ffn_output = layers.Dropout(dropout_rate, name=f"encoder_ffn_dropout_{index}")(
            ffn_output
        )

        output = layers.LayerNormalization(
            axis=-1, name=f"encoder_final_layer_norm_{index}"
        )(attention_output + ffn_output)

        return tf.keras.Model(inputs, output, name=f"encoder_layer_{index}")

    def create_decoder_layer(self, d_model, num_heads, dff, dropout_rate, index):
        # デコーダの入力に名前を付ける
        inputs = layers.Input(shape=(None, d_model), name=f"decoder_input_{index}")
        encoder_output = layers.Input(
            shape=(None, d_model), name=f"encoder_output_{index}"
        )

        embedding = layers.Lambda(
            lambda x: x + self.pos_encoding[: tf.shape(x)[1], :],
            name=f"decoder_positional_encoding_{index}",
        )(inputs)

        # 自己アテンションのマスクを作成
        look_ahead_mask = layers.Lambda(
            lambda x: create_look_ahead_mask(tf.shape(x)[1]),
            name=f"look_ahead_mask_{index}",
        )(inputs)
        # 自己アテンション（マスク付き）
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, name=f"decoder_attention_{index}"
        )(embedding, embedding, attention_mask=look_ahead_mask)

        attention_output = layers.Dropout(
            dropout_rate, name=f"decoder_dropout_{index}"
        )(attention_output)

        attention_output = layers.LayerNormalization(
            axis=-1, name=f"decoder_layer_norm_{index}"
        )(embedding + attention_output)

        # クロスアテンション
        cross_attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
            name=f"decoder_cross_attention_{index}",
        )(attention_output, encoder_output)

        cross_attention_output = layers.Dropout(
            dropout_rate, name=f"decoder_cross_dropout_{index}"
        )(cross_attention_output)

        cross_attention_output = layers.LayerNormalization(
            name=f"decoder_cross_layer_norm_{index}"
        )(attention_output + cross_attention_output)

        ffn_output = layers.Dense(
            dff, activation="relu", name=f"decoder_ffn_1_{index}"
        )(cross_attention_output)

        ffn_output = layers.Dense(d_model, name=f"decoder_ffn_2_{index}")(ffn_output)
        ffn_output = layers.Dropout(dropout_rate, name=f"decoder_ffn_dropout_{index}")(
            ffn_output
        )

        output = layers.LayerNormalization(
            axis=-1, name=f"decoder_final_layer_norm_{index}"
        )(cross_attention_output + ffn_output)

        return tf.keras.Model(
            [inputs, encoder_output], output, name=f"decoder_layer_{index}"
        )

    def call(self, inputs, training=True):
        # 入力を取得
        encoder_input, decoder_input = inputs

        # エンコーダーの処理
        encoder_input = tf.reshape(encoder_input, (tf.shape(encoder_input)[0], -1))
        encoder_output = self.embedding(encoder_input)
        encoder_output = (
            encoder_output + self.pos_encoding[: tf.shape(encoder_output)[1], :]
        )

        # エンコーダ層の順番に処理
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output)

        # デコーダーの処理
        decoder_output = self.embedding(decoder_input)
        decoder_output = (
            decoder_output + self.pos_encoding[: tf.shape(decoder_output)[1], :]
        )

        # デコーダ層の順番に処理
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer([decoder_output, encoder_output])

        # 最後の出力
        final_output = self.final_layer(decoder_output)

        return final_output


# モデルのインスタンス作成
model = Transformer(
    vocab_size, d_model, num_heads, dff, num_layers, max_position_encoding, dropout_rate
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


# モデルのトレーニング
class InputPredictionLogger(Callback):
    def __init__(self, input_tensor, target_tensor):
        super(InputPredictionLogger, self).__init__()
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor

    def on_epoch_end(self, epoch, logs=None):
        for i in range(len(self.input_tensor)):
            predicted_sequence = []
            # i 番目のバッチ
            batch_input = self.input_tensor[i : i + 1]  # type: ignore  # Noqa

            # デコーダの初期入力を設定（<start>トークン）
            start_token = 1
            decoder_input = tf.constant([[start_token]])

            # モデルを使って予測を実行
            predictions = self.model([batch_input, decoder_input], training=False)

            # 最後の予測結果を取得
            predicted_token = tf.argmax(predictions[:, -1, :], axis=-1)
            predicted_sequence.append(predicted_token.numpy()[0])  # 結果をリストに追加

            for _ in range(4):  # 最大4トークンの予測
                # デコーダの入力を更新
                decoder_input = tf.concat(
                    [
                        decoder_input,
                        tf.expand_dims(tf.cast(predicted_token, tf.int32), -1),
                    ],
                    axis=-1,
                )

                output = model([batch_input, decoder_input], training=False)

                # 指数表示の場合に変換
                np.set_printoptions(precision=3, suppress=True)
                output = output.numpy()

                # 最後の単語を予測
                predicted_token = tf.argmax(output[:, -1, :], axis=-1)

                # 結果をリストに追加
                predicted_sequence.append(predicted_token.numpy()[0])

            predicted_text = tokenizer.sequences_to_texts([predicted_sequence])
            print(predicted_text)


# コールバックをインスタンス化
input_prediction_logger = InputPredictionLogger(input_tensor, target_tensor)

# モデルのトレーニング
model.fit(
    [input_tensor, target_tensor],
    target_tensor,
    epochs=500,
    batch_size=batch_size,
    callbacks=[input_prediction_logger],
)

# モデルを保存
model.save("transformer_model.keras")

# 学習した重みをロード
model = tf.keras.models.load_model("transformer_model.keras")

# モデルの状態を表示
print(model.summary())
weights_after_training = model.get_weights()

# 初期デコーダ入力
start_token = tokenizer.word_index.get("<start>", 1)
decoder_input = np.array([[start_token]])

# 推論ループ
predicted_sequence = []
for _ in range(10):
    output = model([input_tensor[0:1], decoder_input], training=False)
    # 最後の単語を予測
    predicted_token = tf.argmax(output[:, -1, :], axis=-1)
    # 結果をリストに追加
    predicted_sequence.append(predicted_token.numpy()[0])
    # デコーダの入力を更新
    decoder_input = tf.concat(
        [decoder_input, tf.expand_dims(predicted_token, -1)], axis=-1
    )

    # もし終了トークンが予測された場合、ループを終了
    if predicted_token == tokenizer.word_index.get("<end>", 2):
        break

# トークンをテキストにデコード
predicted_text = tokenizer.sequences_to_texts([predicted_sequence])
print(predicted_text)
