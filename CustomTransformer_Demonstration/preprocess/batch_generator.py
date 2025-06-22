from typing import Generator
import numpy as np  # type: ignore


class BatchGenerator:
    def __init__(self, tokenizer, input_texts=None, target_texts=None, max_length=500):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length: int = max_length

    def get_batch(self, batch_size: int = 128, shuffle=True) -> Generator:
        # データセットの長さ
        dataset_size: int = len(self.input_texts)
        indices = np.random.permutation(dataset_size)

        for start_idx in range(0, dataset_size, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]  # NOQA 

            # 入力テキストのトークナイズ
            encoder_inputs = self.tokenizer(
                [self.input_texts[i] for i in batch_indices],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )

            # デコーダー入力のトークナイズ
            decoder_inputs = self.tokenizer(
                [self.target_texts[i] for i in batch_indices],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )

            encoder_input_ids = encoder_inputs["input_ids"]
            decoder_input_ids = decoder_inputs["input_ids"]
            yield {
                "encoder_input": encoder_input_ids,
                "decoder_input": decoder_input_ids,
            }
