from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


BYTE_PAD_ID = 0
BYTE_OFFSET = 1
BYTE_VOCAB_SIZE = 257


def normalize_text(text: Any) -> str:
    value = unicodedata.normalize(
        "NFKC",
        str(text),
    )
    return value.strip()


def encode_utf8_bytes(
    text: Any,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = normalize_text(text).lower()
    byte_values = list(
        normalized.encode(
            "utf-8",
            errors="replace",
        )
    )

    byte_ids = [
        value + BYTE_OFFSET
        for value in byte_values[:max_length]
    ]

    attention_mask = [1] * len(byte_ids)

    pad_count = max_length - len(byte_ids)
    byte_ids.extend(
        [BYTE_PAD_ID] * pad_count
    )
    attention_mask.extend(
        [0] * pad_count
    )

    return (
        torch.tensor(
            byte_ids,
            dtype=torch.long,
        ),
        torch.tensor(
            attention_mask,
            dtype=torch.long,
        ),
    )


class MerchantDataset(Dataset):
    def __init__(
        self,
        data_frame: pd.DataFrame,
        max_byte_length: int,
        has_labels: bool = True,
    ):
        required_columns = {
            "id",
            "text",
        }

        missing = required_columns.difference(
            data_frame.columns
        )

        if missing:
            raise KeyError(
                "필수 컬럼이 없습니다: "
                + ", ".join(sorted(missing))
            )

        self.data = data_frame.reset_index(
            drop=True
        ).copy()

        self.max_byte_length = max_byte_length
        self.has_labels = has_labels

        if self.has_labels:
            label_columns = {
                "top_label",
                "mid_label",
                "fine_label",
            }
            label_missing = label_columns.difference(
                self.data.columns
            )
            if label_missing:
                raise KeyError(
                    "Label 컬럼이 없습니다: "
                    + ", ".join(
                        sorted(label_missing)
                    )
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.data.iloc[index]

        byte_ids, byte_attention_mask = (
            encode_utf8_bytes(
                row["text"],
                self.max_byte_length,
            )
        )

        item = {
            "id": row["id"],
            "text": normalize_text(row["text"]),
            "byte_ids": byte_ids,
            "byte_attention_mask": byte_attention_mask,
        }

        if self.has_labels:
            item.update(
                {
                    "top_label": int(
                        row["top_label"]
                    ),
                    "mid_label": int(
                        row["mid_label"]
                    ),
                    "fine_label": int(
                        row["fine_label"]
                    ),
                }
            )

        return item


@dataclass
class HybridCollator:
    tokenizer: PreTrainedTokenizerBase
    max_token_length: int
    has_labels: bool = True

    def __call__(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        texts = [
            item["text"]
            for item in batch
        ]

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt",
        )

        result = {
            "ids": [
                item["id"]
                for item in batch
            ],
            "texts": texts,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized[
                "attention_mask"
            ],
            "byte_ids": torch.stack(
                [
                    item["byte_ids"]
                    for item in batch
                ]
            ),
            "byte_attention_mask": torch.stack(
                [
                    item["byte_attention_mask"]
                    for item in batch
                ]
            ),
        }

        if self.has_labels:
            result.update(
                {
                    "top_labels": torch.tensor(
                        [
                            item["top_label"]
                            for item in batch
                        ],
                        dtype=torch.long,
                    ),
                    "mid_labels": torch.tensor(
                        [
                            item["mid_label"]
                            for item in batch
                        ],
                        dtype=torch.long,
                    ),
                    "fine_labels": torch.tensor(
                        [
                            item["fine_label"]
                            for item in batch
                        ],
                        dtype=torch.long,
                    ),
                }
            )

        return result
