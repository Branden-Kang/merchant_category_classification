import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "신규 20개 카테고리 데이터를 stratified split한 뒤 "
            "기존 train/valid/test CSV에 각각 병합합니다."
        )
    )

    parser.add_argument(
        "--train",
        default="train_data.csv",
        help="기존 train CSV 파일 경로 (기본값: train_data.csv)",
    )
    parser.add_argument(
        "--valid",
        default="valid_data.csv",
        help="기존 validation CSV 파일 경로 (기본값: valid_data.csv)",
    )
    parser.add_argument(
        "--test",
        default="test_data.csv",
        help="기존 test CSV 파일 경로 (기본값: test_data.csv)",
    )
    parser.add_argument(
        "--new",
        default="20_category_data.csv",
        help="신규 20개 카테고리 CSV 파일 경로 (기본값: 20_category_data.csv)",
    )
    parser.add_argument(
        "--label-col",
        default="store_category_cd",
        help="stratify에 사용할 라벨 컬럼명 (기본값: store_category_cd)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="결과 저장 디렉터리 (기본값: output)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="난수 seed (기본값: 42)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="신규 데이터의 train 비율 (기본값: 0.70)",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.15,
        help="신규 데이터의 validation 비율 (기본값: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="신규 데이터의 test 비율 (기본값: 0.15)",
    )

    return parser.parse_args()


def validate_ratio(train_ratio, valid_ratio, test_ratio):
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"train/valid/test 비율의 합은 1.0이어야 합니다. 현재 합: {total}"
        )

    for name, value in [
        ("train_ratio", train_ratio),
        ("valid_ratio", valid_ratio),
        ("test_ratio", test_ratio),
    ]:
        if value <= 0:
            raise ValueError(f"{name}은 0보다 커야 합니다. 현재 값: {value}")


def check_columns_compatible(base_df, new_df, base_name, new_name):
    base_cols = list(base_df.columns)
    new_cols = list(new_df.columns)

    if base_cols != new_cols:
        missing_in_new = [c for c in base_cols if c not in new_cols]
        extra_in_new = [c for c in new_cols if c not in base_cols]

        raise ValueError(
            f"\n[{base_name}]와 [{new_name}]의 컬럼 구성이 일치하지 않습니다.\n"
            f"- {new_name}에 없는 컬럼: {missing_in_new}\n"
            f"- {new_name}에만 있는 컬럼: {extra_in_new}\n"
            "컬럼명 및 순서를 확인해주세요."
        )


def split_new_data(
    new_df,
    label_col,
    train_ratio,
    valid_ratio,
    test_ratio,
    random_state,
):
    if label_col not in new_df.columns:
        raise KeyError(
            f"라벨 컬럼 '{label_col}'이 신규 데이터에 없습니다.\n"
            f"사용 가능한 컬럼: {list(new_df.columns)}"
        )

    if new_df[label_col].isna().any():
        null_count = int(new_df[label_col].isna().sum())
        raise ValueError(
            f"라벨 컬럼 '{label_col}'에 결측치가 {null_count}건 있습니다."
        )

    class_counts = new_df[label_col].value_counts()

    # 2단계 stratified split을 수행하므로 지나치게 작은 클래스는 실패할 수 있음
    too_small = class_counts[class_counts < 3]
    if not too_small.empty:
        raise ValueError(
            "일부 클래스의 데이터가 너무 적어 stratified split이 어렵습니다.\n"
            f"{too_small.to_string()}"
        )

    temp_ratio = valid_ratio + test_ratio

    new_train, new_temp = train_test_split(
        new_df,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=new_df[label_col],
    )

    test_ratio_in_temp = test_ratio / temp_ratio

    new_valid, new_test = train_test_split(
        new_temp,
        test_size=test_ratio_in_temp,
        random_state=random_state,
        stratify=new_temp[label_col],
    )

    return (
        new_train.reset_index(drop=True),
        new_valid.reset_index(drop=True),
        new_test.reset_index(drop=True),
    )


def make_split_summary(new_train, new_valid, new_test, label_col):
    summary = pd.concat(
        [
            new_train[label_col].value_counts().rename("train"),
            new_valid[label_col].value_counts().rename("valid"),
            new_test[label_col].value_counts().rename("test"),
        ],
        axis=1,
    ).fillna(0).astype(int)

    summary["total"] = summary[["train", "valid", "test"]].sum(axis=1)
    summary = summary.sort_index()
    summary.index.name = label_col

    return summary


def main():
    args = parse_args()

    validate_ratio(
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("CSV 파일을 읽는 중입니다...")
    train_df = pd.read_csv(args.train)
    valid_df = pd.read_csv(args.valid)
    test_df = pd.read_csv(args.test)
    new_df = pd.read_csv(args.new)

    # 기존 세 데이터셋과 신규 데이터의 컬럼 구성이 같은지 확인
    check_columns_compatible(train_df, new_df, "train_data", "20_category_data")
    check_columns_compatible(valid_df, new_df, "valid_data", "20_category_data")
    check_columns_compatible(test_df, new_df, "test_data", "20_category_data")

    print("\n신규 20개 카테고리 데이터 분포")
    print(new_df[args.label_col].value_counts().sort_index().to_string())

    print("\n신규 데이터를 train / valid / test로 분리합니다...")
    new_train, new_valid, new_test = split_new_data(
        new_df=new_df,
        label_col=args.label_col,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )

    # 기존 데이터와 병합
    train_v2 = pd.concat([train_df, new_train], ignore_index=True)
    valid_v2 = pd.concat([valid_df, new_valid], ignore_index=True)
    test_v2 = pd.concat([test_df, new_test], ignore_index=True)

    # 최종 데이터 셔플
    train_v2 = train_v2.sample(
        frac=1,
        random_state=args.random_state,
    ).reset_index(drop=True)

    valid_v2 = valid_v2.sample(
        frac=1,
        random_state=args.random_state,
    ).reset_index(drop=True)

    test_v2 = test_v2.sample(
        frac=1,
        random_state=args.random_state,
    ).reset_index(drop=True)

    # 저장
    train_output = output_dir / "train_data_v2.csv"
    valid_output = output_dir / "valid_data_v2.csv"
    test_output = output_dir / "test_data_v2.csv"

    new_train_output = output_dir / "new_20_train.csv"
    new_valid_output = output_dir / "new_20_valid.csv"
    new_test_output = output_dir / "new_20_test.csv"

    summary_output = output_dir / "new_20_split_summary.csv"

    train_v2.to_csv(train_output, index=False, encoding="utf-8-sig")
    valid_v2.to_csv(valid_output, index=False, encoding="utf-8-sig")
    test_v2.to_csv(test_output, index=False, encoding="utf-8-sig")

    # 신규 데이터 split 자체도 별도 저장
    new_train.to_csv(new_train_output, index=False, encoding="utf-8-sig")
    new_valid.to_csv(new_valid_output, index=False, encoding="utf-8-sig")
    new_test.to_csv(new_test_output, index=False, encoding="utf-8-sig")

    summary = make_split_summary(
        new_train,
        new_valid,
        new_test,
        args.label_col,
    )
    summary.to_csv(summary_output, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("신규 데이터 split 결과")
    print("=" * 60)
    print(f"전체      : {len(new_df):,}")
    print(f"Train     : {len(new_train):,}")
    print(f"Validation: {len(new_valid):,}")
    print(f"Test      : {len(new_test):,}")

    print("\n카테고리별 split 결과")
    print(summary.to_string())

    print("\n" + "=" * 60)
    print("기존 → 최종 데이터 건수")
    print("=" * 60)
    print(f"Train : {len(train_df):,} -> {len(train_v2):,}")
    print(f"Valid : {len(valid_df):,} -> {len(valid_v2):,}")
    print(f"Test  : {len(test_df):,} -> {len(test_v2):,}")

    print("\n저장 완료:")
    print(f"- {train_output}")
    print(f"- {valid_output}")
    print(f"- {test_output}")
    print(f"- {new_train_output}")
    print(f"- {new_valid_output}")
    print(f"- {new_test_output}")
    print(f"- {summary_output}")


if __name__ == "__main__":
    main()
