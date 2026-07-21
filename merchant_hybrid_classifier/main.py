from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from hybrid_dataset import (
    HybridCollator,
    MerchantDataset,
)
from model import (
    MerchantHybridConfig,
    MerchantHybridEncoder,
)
from sample_data import create_sample_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "KLUE-RoBERTa + Byte Transformer "
            "가맹점 업종 분류 데모"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "demo",
            "train",
            "test",
            "inference",
        ],
        default="demo",
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "gpu",
        ],
        default="auto",
    )

    parser.add_argument(
        "--pretrained-name",
        type=str,
        default="klue/roberta-base",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--text-lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--head-lr",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-token-length",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max-byte-length",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--byte-hidden-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--byte-num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--byte-num-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--fusion-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--freeze-text-encoder",
        action="store_true",
    )

    parser.add_argument(
        "--top-loss-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--mid-loss-weight",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--fine-loss-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--train-data-path",
        type=str,
        default="./data/train_data.csv",
    )

    parser.add_argument(
        "--valid-data-path",
        type=str,
        default="./data/valid_data.csv",
    )

    parser.add_argument(
        "--test-data-path",
        type=str,
        default="./data/test_data.csv",
    )

    parser.add_argument(
        "--pred-data-path",
        type=str,
        default="./data/pred_data.csv",
    )

    parser.add_argument(
        "--meta-path",
        type=str,
        default="./meta_data/category_meta.csv",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./model/best_model.pt",
    )

    parser.add_argument(
        "--result-dir",
        type=str,
        default="./result",
    )

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(
    device_option: str,
) -> torch.device:
    if device_option == "cpu":
        return torch.device("cpu")

    if device_option == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU가 선택되었지만 CUDA를 "
                "사용할 수 없습니다."
            )
        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def ensure_directories(
    args: argparse.Namespace,
) -> None:
    Path(args.checkpoint_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(args.result_dir).mkdir(
        parents=True,
        exist_ok=True,
    )


def load_meta(
    meta_path: str,
) -> tuple[
    pd.DataFrame,
    dict[int, str],
    dict[int, str],
    dict[int, str],
]:
    meta = pd.read_csv(meta_path)

    required = {
        "top_label",
        "top_name",
        "mid_label",
        "mid_name",
        "fine_label",
        "fine_name",
    }

    missing = required.difference(
        meta.columns
    )

    if missing:
        raise KeyError(
            "메타데이터 필수 컬럼이 없습니다: "
            + ", ".join(sorted(missing))
        )

    top_names = (
        meta[
            [
                "top_label",
                "top_name",
            ]
        ]
        .drop_duplicates()
        .set_index("top_label")["top_name"]
        .to_dict()
    )

    mid_names = (
        meta[
            [
                "mid_label",
                "mid_name",
            ]
        ]
        .drop_duplicates()
        .set_index("mid_label")["mid_name"]
        .to_dict()
    )

    fine_names = (
        meta[
            [
                "fine_label",
                "fine_name",
            ]
        ]
        .drop_duplicates()
        .set_index("fine_label")["fine_name"]
        .to_dict()
    )

    return (
        meta,
        {
            int(key): value
            for key, value in top_names.items()
        },
        {
            int(key): value
            for key, value in mid_names.items()
        },
        {
            int(key): value
            for key, value in fine_names.items()
        },
    )


def build_config(
    args: argparse.Namespace,
    meta: pd.DataFrame,
) -> MerchantHybridConfig:
    return MerchantHybridConfig(
        pretrained_name=args.pretrained_name,
        num_top_classes=(
            int(meta["top_label"].max()) + 1
        ),
        num_mid_classes=(
            int(meta["mid_label"].max()) + 1
        ),
        num_fine_classes=(
            int(meta["fine_label"].max()) + 1
        ),
        max_byte_length=args.max_byte_length,
        byte_hidden_size=args.byte_hidden_size,
        byte_num_layers=args.byte_num_layers,
        byte_num_heads=args.byte_num_heads,
        fusion_size=args.fusion_size,
        dropout=args.dropout,
        freeze_text_encoder=(
            args.freeze_text_encoder
        ),
    )


def build_loader(
    data_frame: pd.DataFrame,
    tokenizer,
    args: argparse.Namespace,
    has_labels: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = MerchantDataset(
        data_frame,
        max_byte_length=args.max_byte_length,
        has_labels=has_labels,
    )

    collator = HybridCollator(
        tokenizer=tokenizer,
        max_token_length=(
            args.max_token_length
        ),
        has_labels=has_labels,
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(
    batch: dict[str, Any],
    device: torch.device,
    has_labels: bool,
) -> dict[str, Any]:
    tensor_keys = [
        "input_ids",
        "attention_mask",
        "byte_ids",
        "byte_attention_mask",
    ]

    if has_labels:
        tensor_keys.extend(
            [
                "top_labels",
                "mid_labels",
                "fine_labels",
            ]
        )

    for key in tensor_keys:
        batch[key] = batch[key].to(
            device,
            non_blocking=True,
        )

    return batch


def calculate_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[
    torch.Tensor,
    dict[str, float],
]:
    top_loss = F.cross_entropy(
        output["top_logits"],
        batch["top_labels"],
    )

    mid_loss = F.cross_entropy(
        output["mid_logits"],
        batch["mid_labels"],
    )

    fine_loss = F.cross_entropy(
        output["fine_logits"],
        batch["fine_labels"],
    )

    total_loss = (
        args.top_loss_weight * top_loss
        + args.mid_loss_weight * mid_loss
        + args.fine_loss_weight * fine_loss
    )

    return (
        total_loss,
        {
            "top_loss": float(
                top_loss.detach().item()
            ),
            "mid_loss": float(
                mid_loss.detach().item()
            ),
            "fine_loss": float(
                fine_loss.detach().item()
            ),
        },
    )


def build_optimizer(
    model: MerchantHybridEncoder,
    args: argparse.Namespace,
) -> AdamW:
    text_parameters = []
    other_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        if name.startswith("text_encoder."):
            text_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    parameter_groups = []

    if text_parameters:
        parameter_groups.append(
            {
                "params": text_parameters,
                "lr": args.text_lr,
                "weight_decay": (
                    args.weight_decay
                ),
            }
        )

    if other_parameters:
        parameter_groups.append(
            {
                "params": other_parameters,
                "lr": args.head_lr,
                "weight_decay": (
                    args.weight_decay
                ),
            }
        )

    return AdamW(parameter_groups)


def forward_model(
    model: MerchantHybridEncoder,
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch[
            "attention_mask"
        ],
        byte_ids=batch["byte_ids"],
        byte_attention_mask=batch[
            "byte_attention_mask"
        ],
    )


@torch.no_grad()
def evaluate(
    model: MerchantHybridEncoder,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_count = 0

    top_true = []
    top_pred = []
    mid_true = []
    mid_pred = []
    fine_true = []
    fine_pred = []
    gate_values = []

    for batch in loader:
        batch = move_batch(
            batch,
            device,
            has_labels=True,
        )

        output = forward_model(
            model,
            batch,
        )

        loss, _ = calculate_loss(
            output,
            batch,
            args,
        )

        batch_size = batch[
            "fine_labels"
        ].size(0)

        total_loss += (
            loss.item() * batch_size
        )
        total_count += batch_size

        top_true.extend(
            batch["top_labels"]
            .detach()
            .cpu()
            .tolist()
        )
        mid_true.extend(
            batch["mid_labels"]
            .detach()
            .cpu()
            .tolist()
        )
        fine_true.extend(
            batch["fine_labels"]
            .detach()
            .cpu()
            .tolist()
        )

        top_pred.extend(
            output["top_logits"]
            .argmax(dim=-1)
            .detach()
            .cpu()
            .tolist()
        )
        mid_pred.extend(
            output["mid_logits"]
            .argmax(dim=-1)
            .detach()
            .cpu()
            .tolist()
        )
        fine_pred.extend(
            output["fine_logits"]
            .argmax(dim=-1)
            .detach()
            .cpu()
            .tolist()
        )

        gate_values.extend(
            output["gate_mean"]
            .detach()
            .cpu()
            .tolist()
        )

    return {
        "loss": (
            total_loss / max(total_count, 1)
        ),
        "top_accuracy": accuracy_score(
            top_true,
            top_pred,
        ),
        "mid_accuracy": accuracy_score(
            mid_true,
            mid_pred,
        ),
        "fine_accuracy": accuracy_score(
            fine_true,
            fine_pred,
        ),
        "fine_macro_f1": f1_score(
            fine_true,
            fine_pred,
            average="macro",
            zero_division=0,
        ),
        "fine_true": fine_true,
        "fine_pred": fine_pred,
        "gate_mean": float(
            np.mean(gate_values)
            if gate_values
            else 0.0
        ),
    }


def save_checkpoint(
    model: MerchantHybridEncoder,
    tokenizer,
    args: argparse.Namespace,
    best_metric: float,
) -> None:
    checkpoint = {
        "model_state_dict": (
            model.state_dict()
        ),
        "model_config": (
            model.get_config_dict()
        ),
        "max_token_length": (
            args.max_token_length
        ),
        "best_metric": best_metric,
    }

    torch.save(
        checkpoint,
        args.checkpoint_path,
    )

    tokenizer_dir = (
        Path(args.checkpoint_path)
        .parent
        / "tokenizer"
    )

    tokenizer.save_pretrained(
        tokenizer_dir
    )


def safe_torch_load(
    path: str,
    map_location: torch.device,
):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[
    MerchantHybridEncoder,
    Any,
    dict[str, Any],
]:
    checkpoint = safe_torch_load(
        checkpoint_path,
        map_location=device,
    )

    config = MerchantHybridConfig(
        **checkpoint["model_config"]
    )

    tokenizer_dir = (
        Path(checkpoint_path)
        .parent
        / "tokenizer"
    )

    tokenizer_source = (
        str(tokenizer_dir)
        if tokenizer_dir.exists()
        else config.pretrained_name
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            tokenizer_source
        )
    )

    model = MerchantHybridEncoder(
        config
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, tokenizer, checkpoint


def train(
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    train_data = pd.read_csv(
        args.train_data_path
    )

    valid_data = pd.read_csv(
        args.valid_data_path
    )

    meta, _, _, _ = load_meta(
        args.meta_path
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.pretrained_name
        )
    )

    train_loader = build_loader(
        train_data,
        tokenizer,
        args,
        has_labels=True,
        shuffle=True,
    )

    valid_loader = build_loader(
        valid_data,
        tokenizer,
        args,
        has_labels=True,
        shuffle=False,
    )

    config = build_config(
        args,
        meta,
    )

    model = MerchantHybridEncoder(
        config
    ).to(device)

    optimizer = build_optimizer(
        model,
        args,
    )

    total_steps = (
        len(train_loader) * args.epochs
    )

    warmup_steps = int(
        total_steps * args.warmup_ratio
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    )

    use_amp = device.type == "cuda"

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    best_macro_f1 = -1.0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        epoch_loss = 0.0
        epoch_count = 0

        for batch in train_loader:
            batch = move_batch(
                batch,
                device,
                has_labels=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):
                output = forward_model(
                    model,
                    batch,
                )

                loss, _ = calculate_loss(
                    output,
                    batch,
                    args,
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = batch[
                "fine_labels"
            ].size(0)

            epoch_loss += (
                loss.item() * batch_size
            )
            epoch_count += batch_size

        train_loss = (
            epoch_loss / max(epoch_count, 1)
        )

        valid_metrics = evaluate(
            model,
            valid_loader,
            device,
            args,
        )

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"fine_acc={valid_metrics['fine_accuracy']:.4f} "
            f"macro_f1={valid_metrics['fine_macro_f1']:.4f} "
            f"gate_mean={valid_metrics['gate_mean']:.4f}"
        )

        if (
            valid_metrics["fine_macro_f1"]
            > best_macro_f1
        ):
            best_macro_f1 = (
                valid_metrics["fine_macro_f1"]
            )

            save_checkpoint(
                model,
                tokenizer,
                args,
                best_macro_f1,
            )

            print(
                "최적 모델 저장:",
                args.checkpoint_path,
            )

    print(
        "학습 완료. Best Macro-F1:",
        f"{best_macro_f1:.4f}",
    )


@torch.no_grad()
def predict(
    model: MerchantHybridEncoder,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()

    rows = []

    for batch in loader:
        batch = move_batch(
            batch,
            device,
            has_labels=False,
        )

        output = forward_model(
            model,
            batch,
        )

        top_probs = torch.softmax(
            output["top_logits"],
            dim=-1,
        )

        mid_probs = torch.softmax(
            output["mid_logits"],
            dim=-1,
        )

        fine_probs = torch.softmax(
            output["fine_logits"],
            dim=-1,
        )

        top_conf, top_pred = (
            top_probs.max(dim=-1)
        )

        mid_conf, mid_pred = (
            mid_probs.max(dim=-1)
        )

        fine_conf, fine_pred = (
            fine_probs.max(dim=-1)
        )

        for index in range(
            len(batch["ids"])
        ):
            rows.append(
                {
                    "id": batch["ids"][index],
                    "text": batch["texts"][index],
                    "top_pred": int(
                        top_pred[index].item()
                    ),
                    "top_confidence": float(
                        top_conf[index].item()
                    ),
                    "mid_pred": int(
                        mid_pred[index].item()
                    ),
                    "mid_confidence": float(
                        mid_conf[index].item()
                    ),
                    "fine_pred": int(
                        fine_pred[index].item()
                    ),
                    "fine_confidence": float(
                        fine_conf[index].item()
                    ),
                    "text_gate_mean": float(
                        output["gate_mean"][
                            index
                        ].item()
                    ),
                }
            )

    return pd.DataFrame(rows)


def test(
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    (
        model,
        tokenizer,
        checkpoint,
    ) = load_checkpoint(
        args.checkpoint_path,
        device,
    )

    config = model.config
    args.max_byte_length = (
        config.max_byte_length
    )
    args.max_token_length = checkpoint.get(
        "max_token_length",
        args.max_token_length,
    )

    test_data = pd.read_csv(
        args.test_data_path
    )

    test_loader = build_loader(
        test_data,
        tokenizer,
        args,
        has_labels=True,
        shuffle=False,
    )

    metrics = evaluate(
        model,
        test_loader,
        device,
        args,
    )

    print(
        classification_report(
            metrics["fine_true"],
            metrics["fine_pred"],
            zero_division=0,
        )
    )

    print(
        "Test fine accuracy:",
        f"{metrics['fine_accuracy']:.4f}",
    )
    print(
        "Test fine Macro-F1:",
        f"{metrics['fine_macro_f1']:.4f}",
    )

    prediction_loader = build_loader(
        test_data[
            [
                "id",
                "text",
            ]
        ],
        tokenizer,
        args,
        has_labels=False,
        shuffle=False,
    )

    result = predict(
        model,
        prediction_loader,
        device,
    )

    result = result.merge(
        test_data[
            [
                "id",
                "top_label",
                "mid_label",
                "fine_label",
            ]
        ],
        how="left",
        on="id",
    )

    result["correct"] = (
        result["fine_pred"]
        == result["fine_label"]
    ).astype(int)

    output_path = (
        Path(args.result_dir)
        / "test_result.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print("테스트 결과 저장:", output_path)


def inference(
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    (
        model,
        tokenizer,
        checkpoint,
    ) = load_checkpoint(
        args.checkpoint_path,
        device,
    )

    config = model.config
    args.max_byte_length = (
        config.max_byte_length
    )
    args.max_token_length = checkpoint.get(
        "max_token_length",
        args.max_token_length,
    )

    pred_data = pd.read_csv(
        args.pred_data_path
    )

    if "id" not in pred_data.columns:
        pred_data.insert(
            0,
            "id",
            range(len(pred_data)),
        )

    loader = build_loader(
        pred_data[
            [
                "id",
                "text",
            ]
        ],
        tokenizer,
        args,
        has_labels=False,
        shuffle=False,
    )

    result = predict(
        model,
        loader,
        device,
    )

    (
        meta,
        top_names,
        mid_names,
        fine_names,
    ) = load_meta(args.meta_path)

    result["top_name"] = result[
        "top_pred"
    ].map(top_names)

    result["mid_name"] = result[
        "mid_pred"
    ].map(mid_names)

    result["fine_name"] = result[
        "fine_pred"
    ].map(fine_names)

    fine_meta = meta[
        [
            "fine_label",
            "store_category_cd",
        ]
    ].drop_duplicates()

    result = result.merge(
        fine_meta,
        how="left",
        left_on="fine_pred",
        right_on="fine_label",
    ).drop(
        columns=["fine_label"]
    )

    output_path = (
        Path(args.result_dir)
        / "inference_result.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(result.to_string(index=False))
    print("추론 결과 저장:", output_path)


def run_demo(
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    create_sample_data(".")

    train(
        args,
        device,
    )

    test(
        args,
        device,
    )

    inference(
        args,
        device,
    )


def main() -> None:
    args = parse_args()

    seed_everything(
        args.seed
    )

    ensure_directories(
        args
    )

    device = get_device(
        args.device
    )

    print("실행 장치:", device)
    print("실행 모드:", args.mode)

    if args.mode == "demo":
        run_demo(
            args,
            device,
        )
    elif args.mode == "train":
        train(
            args,
            device,
        )
    elif args.mode == "test":
        test(
            args,
            device,
        )
    else:
        inference(
            args,
            device,
        )


if __name__ == "__main__":
    main()
