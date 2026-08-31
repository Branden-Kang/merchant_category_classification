import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "misclassified_data.csv를 store_name 패턴 그룹 기준으로 "
            "Train / Hard Test로 분리합니다.\n\n"
            "예:\n"
            "  CSV/PHAMACY #01234\n"
            "  CSV/PHAMACY #05678\n"
            "  -> store_group = CSV/PHAMACY\n\n"
            "같은 store_group 안에 서로 다른 store_name이 2개 이상이면 "
            "store_name 단위로 Train / Hard Test에 분배하고, "
            "store_group에 store_name이 1개뿐이면 Train에만 넣습니다."
        )
    )

    parser.add_argument(
        "--input",
        default="misclassified_data.csv",
        help="입력 CSV 경로 (기본값: misclassified_data.csv)",
    )
    parser.add_argument(
        "--store-col",
        default="store_name",
        help="가맹점명 컬럼 (기본값: store_name)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
        help="그룹 내 서로 다른 store_name 중 Hard Test 목표 비율 (기본값: 0.20)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="난수 seed (기본값: 42)",
    )
    parser.add_argument(
        "--output-dir",
        default="misclassified_output",
        help="결과 저장 폴더 (기본값: misclassified_output)",
    )

    return parser.parse_args()


def normalize_store_group(store_name):
    """
    store_name에서 지점/식별 번호로 보이는 suffix를 보수적으로 제거하여
    유사 가맹점명을 하나의 store_group으로 묶습니다.

    기본 처리 예:
      CSV/PHAMACY #04181 -> CSV/PHAMACY
      CSV/PHAMACY #04745 -> CSV/PHAMACY

    중요:
    너무 공격적인 정규화는 서로 다른 가맹점을 잘못 합칠 수 있으므로,
    기본적으로 '#숫자' 형태의 끝 suffix를 중심으로 처리합니다.
    """

    if pd.isna(store_name):
        return store_name

    name = str(store_name).strip()

    # 연속 공백 정리
    name = re.sub(r"\s+", " ", name)

    # 예: "CSV/PHAMACY #04181" -> "CSV/PHAMACY"
    name = re.sub(
        r"\s*#\s*\d+\s*$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    # 혹시 "# 04181" 제거 후 뒤에 불필요한 구분자가 남는 경우 정리
    name = re.sub(r"[\s\-_/,]+$", "", name).strip()

    return name


def build_store_group(df, store_col):
    result = df.copy()

    if store_col not in result.columns:
        raise KeyError(
            f"'{store_col}' 컬럼이 없습니다.\n"
            f"사용 가능한 컬럼: {list(result.columns)}"
        )

    if result[store_col].isna().any():
        null_count = int(result[store_col].isna().sum())
        raise ValueError(
            f"'{store_col}' 컬럼에 결측치가 {null_count}건 있습니다. "
            "분리 전에 처리해주세요."
        )

    result["store_group"] = result[store_col].apply(
        normalize_store_group
    )

    empty_group = result["store_group"].astype(str).str.strip().eq("")
    if empty_group.any():
        raise ValueError(
            f"정규화 후 store_group이 빈 값인 행이 "
            f"{int(empty_group.sum())}건 있습니다."
        )

    return result


def split_train_hard_test(
    df,
    store_col="store_name",
    test_ratio=0.20,
    random_state=42,
):
    """
    분리 규칙
    ---------
    1. store_group 안에 고유 store_name이 1개
       -> 해당 store_name의 모든 행을 Train

    2. store_group 안에 고유 store_name이 2개 이상
       -> 서로 다른 store_name 단위로 Train / Hard Test 분리

    3. 동일한 정확한 store_name의 여러 행이 존재한다면
       -> 그 모든 행은 반드시 같은 split으로 이동

    4. 각 multi-store group은 Train에 최소 1개 store_name,
       Hard Test에 최소 1개 store_name을 갖도록 함
    """

    if not (0 < test_ratio < 1):
        raise ValueError("test_ratio는 0과 1 사이여야 합니다.")

    rng_seed = random_state

    train_parts = []
    test_parts = []
    summary_rows = []

    # store_group 별 처리
    for group_idx, (store_group, group_df) in enumerate(
        df.groupby("store_group", sort=False)
    ):
        unique_store_names = (
            group_df[store_col]
            .drop_duplicates()
            .tolist()
        )

        n_unique = len(unique_store_names)
        n_rows = len(group_df)

        # -------------------------------
        # Singleton family -> Train only
        # -------------------------------
        if n_unique == 1:
            train_parts.append(group_df.copy())

            summary_rows.append(
                {
                    "store_group": store_group,
                    "unique_store_name_count": n_unique,
                    "total_rows": n_rows,
                    "train_store_name_count": 1,
                    "hard_test_store_name_count": 0,
                    "train_rows": n_rows,
                    "hard_test_rows": 0,
                    "split_rule": "single store_name -> train only",
                }
            )
            continue

        # -------------------------------
        # 같은 family에 store_name 2개 이상
        # -------------------------------
        store_name_series = pd.Series(unique_store_names)

        shuffled_names = (
            store_name_series.sample(
                frac=1,
                random_state=rng_seed + group_idx,
            )
            .tolist()
        )

        # 목표 test 개수: 비율 기반, 단 최소 1개
        test_name_count = max(
            1,
            int(round(n_unique * test_ratio)),
        )

        # Train에 최소 1개는 남겨야 함
        test_name_count = min(
            test_name_count,
            n_unique - 1,
        )

        hard_test_names = set(
            shuffled_names[:test_name_count]
        )

        train_names = set(
            shuffled_names[test_name_count:]
        )

        train_group_df = group_df[
            group_df[store_col].isin(train_names)
        ].copy()

        test_group_df = group_df[
            group_df[store_col].isin(hard_test_names)
        ].copy()

        train_parts.append(train_group_df)
        test_parts.append(test_group_df)

        summary_rows.append(
            {
                "store_group": store_group,
                "unique_store_name_count": n_unique,
                "total_rows": n_rows,
                "train_store_name_count": len(train_names),
                "hard_test_store_name_count": len(hard_test_names),
                "train_rows": len(train_group_df),
                "hard_test_rows": len(test_group_df),
                "split_rule": (
                    f"multi store_name -> approx "
                    f"{test_ratio:.0%} hard test"
                ),
            }
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    if test_parts:
        hard_test_df = pd.concat(
            test_parts,
            ignore_index=True,
        )
    else:
        hard_test_df = df.iloc[0:0].copy()

    # 최종 셔플
    if len(train_df) > 0:
        train_df = train_df.sample(
            frac=1,
            random_state=random_state,
        ).reset_index(drop=True)

    if len(hard_test_df) > 0:
        hard_test_df = hard_test_df.sample(
            frac=1,
            random_state=random_state,
        ).reset_index(drop=True)

    summary_df = pd.DataFrame(summary_rows)

    return train_df, hard_test_df, summary_df


def validate_split(
    original_df,
    train_df,
    hard_test_df,
    store_col,
):
    """
    leakage 및 singleton 규칙 검증
    """

    # 동일한 exact store_name이 Train / Hard Test 양쪽에 존재하면 안 됨
    train_names = set(train_df[store_col].unique())
    test_names = set(hard_test_df[store_col].unique())

    overlap = train_names & test_names

    if overlap:
        raise RuntimeError(
            "동일 store_name이 Train과 Hard Test에 동시에 존재합니다.\n"
            f"예시: {sorted(overlap)[:20]}"
        )

    # store_group 내 고유 store_name이 1개뿐인 그룹은 test에 없어야 함
    group_unique_counts = (
        original_df.groupby("store_group")[store_col]
        .nunique()
    )

    singleton_groups = set(
        group_unique_counts[
            group_unique_counts == 1
        ].index
    )

    hard_test_groups = set(
        hard_test_df["store_group"].unique()
    )

    invalid_singletons = (
        singleton_groups & hard_test_groups
    )

    if invalid_singletons:
        raise RuntimeError(
            "store_name이 1개뿐인 store_group이 "
            "Hard Test에 포함되었습니다.\n"
            f"예시: {sorted(invalid_singletons)[:20]}"
        )


def make_store_name_assignment(
    train_df,
    hard_test_df,
    store_col,
):
    """
    개별 store_name이 어느 split에 들어갔는지 확인하기 위한 표.
    """
    train_assignment = (
        train_df[
            ["store_group", store_col]
        ]
        .drop_duplicates()
        .assign(split="train")
    )

    test_assignment = (
        hard_test_df[
            ["store_group", store_col]
        ]
        .drop_duplicates()
        .assign(split="hard_test")
    )

    assignment = pd.concat(
        [train_assignment, test_assignment],
        ignore_index=True,
    )

    return assignment.sort_values(
        ["store_group", "split", store_col]
    ).reset_index(drop=True)


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_df = pd.read_csv(input_path)

    df = build_store_group(
        raw_df,
        args.store_col,
    )

    print("=" * 75)
    print("misclassified 데이터 확인")
    print("=" * 75)
    print(f"전체 행 수              : {len(df):,}")
    print(
        f"고유 store_name 수       : "
        f"{df[args.store_col].nunique():,}"
    )
    print(
        f"고유 store_group 수      : "
        f"{df['store_group'].nunique():,}"
    )

    group_stats = (
        df.groupby("store_group")[args.store_col]
        .nunique()
    )

    print(
        f"store_name 1개인 그룹 수 : "
        f"{int((group_stats == 1).sum()):,}"
    )
    print(
        f"store_name 2개+ 그룹 수  : "
        f"{int((group_stats >= 2).sum()):,}"
    )

    # 정규화가 실제로 여러 store_name을 묶은 그룹 예시
    multi_groups = group_stats[
        group_stats >= 2
    ].index

    if len(multi_groups) > 0:
        print("\n[여러 store_name이 같은 그룹으로 묶인 예시]")

        example = (
            df[
                df["store_group"].isin(multi_groups)
            ][["store_group", args.store_col]]
            .drop_duplicates()
            .sort_values(
                ["store_group", args.store_col]
            )
            .head(30)
        )

        print(example.to_string(index=False))

    # split
    train_df, hard_test_df, summary_df = (
        split_train_hard_test(
            df=df,
            store_col=args.store_col,
            test_ratio=args.test_ratio,
            random_state=args.random_state,
        )
    )

    validate_split(
        original_df=df,
        train_df=train_df,
        hard_test_df=hard_test_df,
        store_col=args.store_col,
    )

    assignment_df = make_store_name_assignment(
        train_df=train_df,
        hard_test_df=hard_test_df,
        store_col=args.store_col,
    )

    # 결과 CSV에는 분석에 유용하도록 store_group을 유지
    train_path = (
        output_dir
        / "misclassified_train.csv"
    )

    hard_test_path = (
        output_dir
        / "misclassified_hard_test.csv"
    )

    summary_path = (
        output_dir
        / "misclassified_group_split_summary.csv"
    )

    assignment_path = (
        output_dir
        / "misclassified_store_name_assignment.csv"
    )

    train_df.to_csv(
        train_path,
        index=False,
        encoding="utf-8-sig",
    )

    hard_test_df.to_csv(
        hard_test_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    assignment_df.to_csv(
        assignment_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 결과 출력
    print("\n" + "=" * 75)
    print("Train / Hard Test 분리 결과")
    print("=" * 75)

    print(
        f"Train 행 수               : "
        f"{len(train_df):,}"
    )
    print(
        f"Hard Test 행 수           : "
        f"{len(hard_test_df):,}"
    )

    print(
        f"Train 고유 store_name     : "
        f"{train_df[args.store_col].nunique():,}"
    )
    print(
        f"Hard Test 고유 store_name : "
        f"{hard_test_df[args.store_col].nunique():,}"
    )

    if len(df) > 0:
        print(
            f"실제 Hard Test 행 비율     : "
            f"{len(hard_test_df) / len(df):.2%}"
        )

    print("\n검증 완료")
    print(
        "- store_name이 1개뿐인 그룹은 Train에만 포함"
    )
    print(
        "- 동일 exact store_name은 Train/Hard Test 양쪽에 중복되지 않음"
    )

    print("\n저장 파일")
    print(f"- {train_path}")
    print(f"- {hard_test_path}")
    print(f"- {summary_path}")
    print(f"- {assignment_path}")

    print("\n예시 동작")
    print(
        "CSV/PHAMACY #04181, #04745, #08833, #09527"
    )
    print(
        "  -> store_group = CSV/PHAMACY"
    )
    print(
        "  -> 서로 다른 store_name들을 Train / Hard Test로 분배"
    )
    print(
        "카카오택시-서울 하나만 존재"
    )
    print(
        "  -> 해당 그룹은 Train에만 포함"
    )


if __name__ == "__main__":
    main()
