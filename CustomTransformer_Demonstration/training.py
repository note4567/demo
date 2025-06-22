import tensorflow as tf  # NOQA # type: ignore
from transformer import Transformer
from pathlib import Path
from metrics import padded_cross_entropy_loss, padded_accuracy
from preprocess.batch_generator import BatchGenerator
from transformers import BertJapaneseTokenizer  # type:ignore
import pandas as pd  # type:ignore
import itertools

# [Create Data]
current_path: Path = Path.cwd()
data_path = Path("preprocess/data/data.csv")
absolute_data_path: Path = current_path / data_path

df: pd.DataFrame = pd.read_csv(absolute_data_path, encoding="utf-8")
input_texts = df["context"].fillna("").tolist()
target_texts = df["label"].fillna("").tolist()
text = list(itertools.chain.from_iterable(df.values.tolist()))

# トークナイザーの初期化
tokenizer = BertJapaneseTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
batch_generator = BatchGenerator(
    tokenizer=tokenizer, input_texts=input_texts, target_texts=target_texts
)
batch_size = 5
epochs = 100

# Transformer オブジェクトを作成し、初期化する。
transformer = Transformer(
    vocab_size=32768,
    hopping_num=4,
    head_num=8,
    hidden_dim=512,
    dropout_rate=0.1,
    max_length=500,
)


# カスタム学習率スケジューラー
class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_step, max_learning_rate) -> None:
        super().__init__()
        self.warmup_step = warmup_step
        self.max_learning_rate = max_learning_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        print(step)
        rate = tf.minimum(step**-0.5, step * (self.warmup_step**-1.5)) / (
            self.warmup_step**-0.5
        )
        return self.max_learning_rate * rate


# オプティマイザーとロス関数の設定
optimizer = tf.keras.optimizers.Adam(
    CustomLearningRateSchedule(warmup_step=600, max_learning_rate=1e-4),
    beta_2=0.98,
)


# カスタムロス関数
def loss_function(y_true, y_pred, smoothing=0.05):
    xentropy, weights = padded_cross_entropy_loss(
        y_pred, y_true, smoothing=smoothing, vocab_size=transformer.vocab_size
    )
    return tf.reduce_sum(xentropy) / tf.reduce_sum(weights)


# メトリクスの定義
def accuracy_metric(y_true, y_pred):
    accuracies, weights = padded_accuracy(y_pred, y_true)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(weights)


# モデルのコンパイル
transformer.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy_metric])

save_dir: Path = current_path / Path("temp/learning/transformer/")
log_dir: Path = save_dir / Path("log")
ckpt_path: Path = save_dir / Path("checkpoints/model.ckpt")
log_dir.mkdir(parents=True, exist_ok=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,  # ヒストグラムの更新頻度
    write_graph=False,  # グラフの書き込みを無効化
    update_freq="epoch",  # 'batch'から'epoch'に変更
)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path / Path("20241229_{epoch:02d}_{loss:.2f}.weights.h5"),
    save_weights_only=True,
    save_best_only=True,
    save_freq=30000,
    monitor="loss",
    verbose=1,
)


# バッチデータを格納するリスト
encoder_inputs: list = []
decoder_inputs: list = []
decoder_outputs: list = []

# バッチごとにデータを収集
for batch in batch_generator.get_batch(batch_size=batch_size):
    encoder_input = tf.convert_to_tensor(batch["encoder_input"], dtype=tf.int32)
    decoder_input = tf.convert_to_tensor(batch["decoder_input"], dtype=tf.int32)
    decoder_output = tf.convert_to_tensor(batch["decoder_input"], dtype=tf.int32)

    if encoder_input.shape[0] != batch_size:
        continue  # Skip this batch

    encoder_inputs.append(encoder_input)
    decoder_inputs.append(decoder_input)
    decoder_outputs.append(decoder_output)

dataset = (
    {"encoder_input": encoder_input, "decoder_input": decoder_input},
    decoder_outputs,
)

# 各バッチ内の最大シーケンス長を計算
max_encoder_length = max([encoder_input.shape[1] for encoder_input in encoder_inputs])
max_decoder_length = max([decoder_input.shape[1] for decoder_input in decoder_inputs])

# バッチ内でテンソルの長さを一致させてから、個別に処理
encoder_inputs_padded: list = []
decoder_inputs_padded: list = []
decoder_outputs_padded: list = []

for encoder_input, decoder_input, decoder_output in zip(
    encoder_inputs, decoder_inputs, decoder_outputs
):
    # 各バッチにパディングを施す
    padded_encoder_input = tf.pad(
        encoder_input, [[0, 0], [0, max_encoder_length - encoder_input.shape[1]]]
    )

    padded_decoder_input = tf.pad(
        decoder_input, [[0, 0], [0, max_encoder_length - decoder_input.shape[1]]]
    )
    padded_decoder_output = tf.pad(
        decoder_output, [[0, 0], [0, max_encoder_length - decoder_output.shape[1]]]
    )

    encoder_inputs_padded.append(padded_encoder_input)
    decoder_inputs_padded.append(padded_decoder_input)
    decoder_outputs_padded.append(padded_decoder_output)

dataset = tf.data.Dataset.from_tensor_slices(
    (
        {
            "encoder_input": tf.stack(encoder_inputs_padded),
            "decoder_input": tf.stack(decoder_inputs_padded),
        },
        tf.stack(decoder_outputs_padded),
    )
)


transformer.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
