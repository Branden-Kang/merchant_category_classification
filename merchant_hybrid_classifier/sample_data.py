from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


SEED = 42

CATEGORY_META = [
    {
        "fine_label": 0,
        "fine_name": "카페",
        "mid_label": 0,
        "mid_name": "카페",
        "top_label": 0,
        "top_name": "음식",
        "store_category_cd": "C000",
    },
    {
        "fine_label": 1,
        "fine_name": "한식",
        "mid_label": 1,
        "mid_name": "한식",
        "top_label": 0,
        "top_name": "음식",
        "store_category_cd": "C001",
    },
    {
        "fine_label": 2,
        "fine_name": "편의점",
        "mid_label": 2,
        "mid_name": "편의점",
        "top_label": 1,
        "top_name": "소매",
        "store_category_cd": "C002",
    },
    {
        "fine_label": 3,
        "fine_name": "약국",
        "mid_label": 3,
        "mid_name": "약국",
        "top_label": 2,
        "top_name": "의료",
        "store_category_cd": "C003",
    },
    {
        "fine_label": 4,
        "fine_name": "의류",
        "mid_label": 4,
        "mid_name": "의류",
        "top_label": 1,
        "top_name": "소매",
        "store_category_cd": "C004",
    },
]

TEXT_POOLS = {
    0: {
        "brands": [
            "봄날", "라온", "달빛", "모닝빈", "아띠", "블루문",
            "커피온", "빈스", "카페드림", "로스터리101",
        ],
        "keywords": [
            "카페", "커피전문점", "아메리카노", "라떼하우스",
            "디저트카페", "coffee", "CAFE",
        ],
    },
    1: {
        "brands": [
            "고향집", "정든밥상", "엄마손", "시골밥상", "한상차림",
            "맛고을", "우리식당", "서울뚝배기", "푸른마을", "행복식당",
        ],
        "keywords": [
            "한식", "김치찌개", "백반", "국밥", "불고기",
            "된장찌개", "한정식",
        ],
    },
    2: {
        "brands": [
            "스마일", "365", "우리동네", "굿데이", "새벽",
            "프레시", "나이스", "올데이", "미니", "행복",
        ],
        "keywords": [
            "편의점", "24시마트", "도시락편의점", "생활편의점",
            "convenience", "CU스타일", "GS25스타일",
        ],
    },
    3: {
        "brands": [
            "건강", "온누리", "새봄", "행복", "중앙",
            "우리", "푸른", "참좋은", "튼튼", "메디컬",
        ],
        "keywords": [
            "약국", "처방전조제", "의약품", "건강약국",
            "pharmacy", "24H약국", "메디약국",
        ],
    },
    4: {
        "brands": [
            "스타일온", "모던핏", "데일리룩", "블랑", "어반",
            "패션하우스", "옷장", "시크", "루나", "더핏",
        ],
        "keywords": [
            "의류", "여성복", "남성복", "패션", "캐주얼웨어",
            "FASHION", "옷가게",
        ],
    },
}

BRANCHES = [
    "강남점", "홍대점", "서울역점", "센트럴점", "2호점",
    "본점", "역삼점", "동탄점", "신도시점", "24H",
]


def _make_class_texts(label: int, count: int = 30) -> list[str]:
    pool = TEXT_POOLS[label]
    texts = []

    for index in range(count):
        brand = pool["brands"][index % len(pool["brands"])]
        keyword = pool["keywords"][(index * 3) % len(pool["keywords"])]
        branch = BRANCHES[(index * 7) % len(BRANCHES)]

        variants = [
            f"{brand} {keyword} {branch}",
            f"{brand}{keyword}{branch}",
            f"(주){brand}-{keyword}_{branch}",
            f"{keyword} {brand} {branch}",
        ]
        texts.append(variants[index % len(variants)])

    random.Random(SEED + label).shuffle(texts)
    return texts


def _build_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows = []
    valid_rows = []
    test_rows = []

    row_ids = {
        "train": 0,
        "valid": 0,
        "test": 0,
    }

    meta_by_label = {
        item["fine_label"]: item
        for item in CATEGORY_META
    }

    for fine_label in sorted(meta_by_label):
        meta = meta_by_label[fine_label]
        texts = _make_class_texts(fine_label, count=30)

        split_texts = {
            "train": texts[:20],
            "valid": texts[20:25],
            "test": texts[25:30],
        }

        for split_name, values in split_texts.items():
            target_rows = {
                "train": train_rows,
                "valid": valid_rows,
                "test": test_rows,
            }[split_name]

            for text in values:
                target_rows.append(
                    {
                        "id": row_ids[split_name],
                        "text": text,
                        "top_label": meta["top_label"],
                        "mid_label": meta["mid_label"],
                        "fine_label": meta["fine_label"],
                    }
                )
                row_ids[split_name] += 1

    return (
        pd.DataFrame(train_rows),
        pd.DataFrame(valid_rows),
        pd.DataFrame(test_rows),
    )


def create_sample_data(base_dir: str | Path = ".") -> None:
    base_path = Path(base_dir)
    data_path = base_path / "data"
    meta_path = base_path / "meta_data"

    data_path.mkdir(parents=True, exist_ok=True)
    meta_path.mkdir(parents=True, exist_ok=True)

    train_data, valid_data, test_data = _build_split()

    train_data.to_csv(
        data_path / "train_data.csv",
        index=False,
    )
    valid_data.to_csv(
        data_path / "valid_data.csv",
        index=False,
    )
    test_data.to_csv(
        data_path / "test_data.csv",
        index=False,
    )

    pred_data = pd.DataFrame(
        {
            "id": list(range(10)),
            "text": [
                "MEGA MGC 커피 강남2호점",
                "김치찌개랑 백반 파는 고향집",
                "GS25_동탄센트럴24H",
                "행복온누리약국 처방전",
                "URBAN FASHION 여성 캐주얼",
                "카페봄날아메리카노",
                "24시 도시락 convenience",
                "메디컬pharmacy",
                "정든밥상 한정식",
                "블랑남성복홍대점",
            ],
        }
    )
    pred_data.to_csv(
        data_path / "pred_data.csv",
        index=False,
    )

    pd.DataFrame(CATEGORY_META).to_csv(
        meta_path / "category_meta.csv",
        index=False,
    )

    print("샘플 데이터 생성 완료")
    print(f"train: {len(train_data)}")
    print(f"valid: {len(valid_data)}")
    print(f"test : {len(test_data)}")
    print(f"pred : {len(pred_data)}")


if __name__ == "__main__":
    create_sample_data(".")
