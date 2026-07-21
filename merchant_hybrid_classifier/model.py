from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class MerchantHybridConfig:
    pretrained_name: str = "klue/roberta-base"
    num_top_classes: int = 3
    num_mid_classes: int = 5
    num_fine_classes: int = 5

    byte_vocab_size: int = 257
    max_byte_length: int = 96
    byte_hidden_size: int = 128
    byte_num_layers: int = 2
    byte_num_heads: int = 4

    fusion_size: int = 256
    dropout: float = 0.1
    freeze_text_encoder: bool = False


class MaskedMeanPooling(nn.Module):
    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1)
        mask = mask.to(hidden_state.dtype)

        summed = (
            hidden_state * mask
        ).sum(dim=1)

        counts = mask.sum(dim=1).clamp(
            min=1.0
        )

        return summed / counts


class ByteTinyTransformer(nn.Module):
    def __init__(
        self,
        config: MerchantHybridConfig,
    ):
        super().__init__()

        self.max_byte_length = (
            config.max_byte_length
        )

        self.byte_embedding = nn.Embedding(
            config.byte_vocab_size,
            config.byte_hidden_size,
            padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            config.max_byte_length,
            config.byte_hidden_size,
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=config.byte_hidden_size,
                nhead=config.byte_num_heads,
                dim_feedforward=(
                    config.byte_hidden_size * 4
                ),
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.byte_num_layers,
        )

        self.norm = nn.LayerNorm(
            config.byte_hidden_size
        )

        self.pooling = MaskedMeanPooling()

    def forward(
        self,
        byte_ids: torch.Tensor,
        byte_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = (
            byte_ids.shape
        )

        positions = torch.arange(
            sequence_length,
            device=byte_ids.device,
        ).unsqueeze(0)

        positions = positions.expand(
            batch_size,
            sequence_length,
        )

        hidden_state = (
            self.byte_embedding(byte_ids)
            + self.position_embedding(positions)
        )

        padding_mask = (
            byte_attention_mask == 0
        )

        hidden_state = self.encoder(
            hidden_state,
            src_key_padding_mask=padding_mask,
        )

        hidden_state = self.norm(
            hidden_state
        )

        return self.pooling(
            hidden_state,
            byte_attention_mask,
        )


class MerchantHybridEncoder(nn.Module):
    def __init__(
        self,
        config: MerchantHybridConfig,
    ):
        super().__init__()

        self.config = config

        self.text_encoder = (
            AutoModel.from_pretrained(
                config.pretrained_name
            )
        )

        text_hidden_size = (
            self.text_encoder.config.hidden_size
        )

        self.byte_encoder = (
            ByteTinyTransformer(config)
        )

        self.text_pooling = MaskedMeanPooling()

        self.text_projection = nn.Sequential(
            nn.Linear(
                text_hidden_size,
                config.fusion_size,
            ),
            nn.LayerNorm(
                config.fusion_size
            ),
            nn.GELU(),
        )

        self.byte_projection = nn.Sequential(
            nn.Linear(
                config.byte_hidden_size,
                config.fusion_size,
            ),
            nn.LayerNorm(
                config.fusion_size
            ),
            nn.GELU(),
        )

        self.gate = nn.Linear(
            config.fusion_size * 2,
            config.fusion_size,
        )

        self.fusion_norm = nn.LayerNorm(
            config.fusion_size
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.top_classifier = nn.Linear(
            config.fusion_size,
            config.num_top_classes,
        )

        self.mid_classifier = nn.Linear(
            config.fusion_size,
            config.num_mid_classes,
        )

        self.fine_classifier = nn.Linear(
            config.fusion_size,
            config.num_fine_classes,
        )

        if config.freeze_text_encoder:
            for parameter in (
                self.text_encoder.parameters()
            ):
                parameter.requires_grad = False

    def get_config_dict(self) -> dict:
        return asdict(self.config)

    def _encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.freeze_text_encoder:
            self.text_encoder.eval()

            with torch.no_grad():
                output = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        return self.text_pooling(
            output.last_hidden_state,
            attention_mask,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        byte_ids: torch.Tensor,
        byte_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_feature = self._encode_text(
            input_ids,
            attention_mask,
        )

        byte_feature = self.byte_encoder(
            byte_ids,
            byte_attention_mask,
        )

        text_feature = self.text_projection(
            text_feature
        )

        byte_feature = self.byte_projection(
            byte_feature
        )

        gate_input = torch.cat(
            [
                text_feature,
                byte_feature,
            ],
            dim=-1,
        )

        gate = torch.sigmoid(
            self.gate(gate_input)
        )

        fused_feature = (
            gate * text_feature
            + (1.0 - gate) * byte_feature
        )

        fused_feature = (
            self.fusion_norm(fused_feature)
        )

        fused_feature = self.dropout(
            fused_feature
        )

        return {
            "embedding": fused_feature,
            "gate_mean": gate.mean(
                dim=-1
            ),
            "top_logits": self.top_classifier(
                fused_feature
            ),
            "mid_logits": self.mid_classifier(
                fused_feature
            ),
            "fine_logits": self.fine_classifier(
                fused_feature
            ),
        }
