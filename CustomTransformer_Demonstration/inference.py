import tensorflow as tf  # NOQA # type: ignore
from transformer import Transformer
from transformers import BertJapaneseTokenizer  # NOQA # type: ignore
from pathlib import Path
import numpy as np  # NOQA # type: ignore


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


class TransformerInference:
    def __init__(
        self,
        vocab_size: int = 32768,
        hopping_num: int = 4,
        head_num: int = 8,
        hidden_dim: int = 512,
        dropout_rate: float = 0.0,  # 推論時はドロップアウトを使用しない
        max_length: int = 500,
        checkpoint_path: str = None,
    ) -> None:

        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            "tohoku-nlp/bert-base-japanese-v3"
        )

        self.max_length = max_length
        self.vocab_size = vocab_size

        # 学習済みの重みを読み込む
        if checkpoint_path:
            self.load_weights(checkpoint_path)

    def load_weights(self, checkpoint_path: str):
        """学習済みの重みを読み込む"""
        self.transformer = tf.keras.models.load_model(
            checkpoint_path,
            custom_objects={
                "Transformer": Transformer,
                "CustomLearningRateSchedule": CustomLearningRateSchedule,
            },
        )

    def beam_search(self, encoder_input: tf.Tensor, beam_size: int = 4) -> tf.Tensor:
        """ビームサーチによる生成"""
        # 初期状態の設定
        initial_decoded = tf.constant([[2]], dtype=tf.int32)
        initial_scores = tf.constant([0.0])

        # ビームの初期化
        beam_scores = initial_scores  # 初期スコア
        beam_sequences = initial_decoded  # 初回は [CLS]トークンのみ

        finished_sequences: tf.Tensor = tf.zeros((0, 1), dtype=tf.int32)
        finished_scores: tf.Tensor = tf.zeros(0)

        # 生成ループ
        for i in range(self.max_length - 1):
            predictions = self.transformer(
                {
                    "encoder_input": tf.tile(
                        encoder_input, [tf.shape(beam_sequences)[0], 1]
                    ),
                    "decoder_input": beam_sequences,
                },
                training=False,
            )

            logits = predictions[:, -1, :]
            log_probs = tf.nn.log_softmax(logits)

            # スコアを計算
            if i == 0:
                scores = log_probs[0]
            else:
                scores = beam_scores[:, None] + log_probs

            flattened_scores = tf.reshape(scores, [-1])
            top_scores, top_indices = tf.nn.top_k(flattened_scores, k=beam_size)
            beam_indices = top_indices // self.vocab_size
            token_indices = top_indices % self.vocab_size

            # 新しいシーケンスを作成
            new_sequences = tf.concat(
                [
                    tf.gather(beam_sequences, beam_indices),
                    tf.expand_dims(token_indices, 1),
                ],
                axis=1,
            )

            # 完了したシーケンスを処理
            finished_flags = token_indices == 3
            finished_flags = tf.cast(finished_flags, tf.bool)

            # この step で終了したビーム
            now_finished_seq = tf.boolean_mask(new_sequences, finished_flags)
            padded_beams = tf.pad(
                finished_sequences,
                [
                    [0, 0],
                    [
                        0,
                        tf.shape(now_finished_seq)[1] - tf.shape(finished_sequences)[1],
                    ],
                ],
            )
            finished_sequences = tf.concat(
                [padded_beams, now_finished_seq],
                axis=0,
            )
            finished_scores = tf.concat(
                [finished_scores, tf.boolean_mask(top_scores, finished_flags)], axis=0
            )

            # 未完了のシーケンスを更新
            unfinished_flags: bool = ~finished_flags
            beam_sequences = tf.boolean_mask(new_sequences, unfinished_flags)
            beam_scores = tf.boolean_mask(top_scores, unfinished_flags)

            # すべてのビームが完了したか、または最大長に達した場合は終了
            if tf.size(beam_sequences) == 0:
                break

        # 最終的な結果を選択
        if np.array(tf.shape(finished_sequences)[0]) > 0:
            _, best_index = tf.nn.top_k(finished_scores, k=1)
            return tf.gather(finished_sequences, best_index)[0]
        else:
            _, best_index = tf.nn.top_k(beam_scores, k=1)
            return tf.gather(beam_sequences, best_index)[0]

    def predict(self, input_text: str, beam_size: int = 4) -> str:
        """テキストを入力として受け取り、生成されたテキストを返す"""
        encoder_input = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )["input_ids"]

        encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.int32)
        padded_encoder_input = tf.pad(
            encoder_input,
            [[0, 0], [0, 86 - encoder_input.shape[1]]],
        )

        padded_encoder_input = tf.convert_to_tensor(padded_encoder_input)

        # ビームサーチによる生成
        output_sequence = self.beam_search(padded_encoder_input, beam_size=beam_size)
        output_text = self.tokenizer.convert_ids_to_tokens(output_sequence.numpy())

        return output_text

    def generate_greedy(self, input_text: str) -> str:
        """貪欲法による生成（単純なバージョン）"""
        encoder_input = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )["input_ids"]
        initial_decoded = tf.constant([[2]], dtype=tf.int32)
        decoder_input = initial_decoded  

        # 文章生成ループ
        for _ in range(self.max_length):
            predictions = self.transformer(
                {
                    "encoder_input": encoder_input,
                    "decoder_input": decoder_input,
                },
                training=False,
            )

            last_pred = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(last_pred, axis=-1), tf.int32)
            decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

        return output_text


# 使用例
if __name__ == "__main__":
    current_path: Path = Path.cwd()

    # チェックポイントパスの設定
    checkpoint_path = "checkpoints/transformer_model.keras"

    # 推論モデルの初期化
    inference_model = TransformerInference(
        vocab_size=32768, checkpoint_path=checkpoint_path
    )

    # テスト用の入力テキスト
    input_text = """<0&nbsp;m676-0011兵庫県高砂市荒井町小松原4-1005電話番号079-490-3731営業時間08:00～22:00定休日不定休アクセス電車荒井（兵庫県）駅（山陽電気鉄道）徒歩17分高砂（兵庫県
<営業時間は?"""

    # ビームサーチによる生成
    output_text = inference_model.predict(input_text, beam_size=4)
    print(f"Input: {input_text}")
    print(f"Beam Search Output: {output_text}")

    # 貪欲法による生成
    greedy_output = inference_model.generate_greedy(input_text)
    print(f"Greedy Output: {greedy_output}")

