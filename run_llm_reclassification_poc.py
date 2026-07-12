import os
import time
from typing import List, Literal

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


CATEGORY_TAXONOMY = [
    "외식/음식점",
    "카페/디저트",
    "편의점/마트",
    "온라인쇼핑",
    "패션/잡화",
    "생활/잡화",
    "의료/건강",
    "교통/자동차",
    "여행/숙박",
    "교육",
    "문화/여가",
    "기타",
    "UNKNOWN",
]


class MerchantClassificationResult(BaseModel):
    merchant_id: str
    merchant_name: str
    predicted_category: str
    confidence: float = Field(ge=0.0, le=1.0)
    decision_type: Literal["RECLASSIFIED", "KEEP_OTHER", "NEEDS_REVIEW"]
    reason: str
    is_ambiguous: bool
    alternative_categories: List[str]


class BatchClassificationResponse(BaseModel):
    results: List[MerchantClassificationResult]


def build_system_prompt() -> str:
    categories = "\n".join([f"- {c}" for c in CATEGORY_TAXONOMY])

    return f"""
당신은 금융 거래 데이터의 가맹점 카테고리 재분류 전문가입니다.

현재 입력되는 가맹점들은 Rule 기반 분류와 ML 기반 분류 이후에도
"기타", "미분류", "UNKNOWN" 또는 낮은 신뢰도 상태로 남은 가맹점입니다.

가맹점명만 보고 재분류 가능한 경우 제공된 카테고리 체계 중 하나로 분류하고,
근거가 부족하면 "기타" 또는 "UNKNOWN"으로 유지하세요.

중요 규칙:
1. 반드시 아래 카테고리 체계 안에서만 선택하세요.
2. 새로운 카테고리를 만들지 마세요.
3. 입력 정보는 가맹점명만 사용하세요.
4. 한글, 영어, 로마자 표기, 일본어, 중국어, 기타 외국어, 혼합 표기를 고려하세요.
5. 잘 알려진 브랜드이거나 상호명에 업종 단서가 명확한 경우에만 높은 confidence를 부여하세요.
6. 상호명이 일반적이거나 업종 단서가 부족하면 "기타" 또는 "UNKNOWN"으로 유지하세요.
7. 모든 가맹점을 억지로 재분류하지 마세요.
8. 최종 출력은 지정된 JSON schema를 반드시 따르세요.

decision_type 기준:
- RECLASSIFIED: 특정 카테고리로 재분류 가능
- KEEP_OTHER: 근거 부족으로 기타 또는 UNKNOWN 유지
- NEEDS_REVIEW: 사람이 검수해야 할 정도로 애매함

confidence 기준:
- 0.75 이상: 자동 반영 가능성이 높은 수준
- 0.50 이상 0.75 미만: 검수 또는 보류 필요
- 0.50 미만: 기타 유지 권장

카테고리 체계:
{categories}
""".strip()


def build_user_prompt(batch_df: pd.DataFrame) -> str:
    items = [
        {
            "merchant_id": str(row["merchant_id"]),
            "merchant_name": str(row["merchant_name"]),
        }
        for _, row in batch_df.iterrows()
    ]

    return f"""
아래 가맹점들을 각각 재분류하세요.

입력:
{items}
""".strip()


def classify_batch_with_llm(batch_df: pd.DataFrame, max_retries: int = 3) -> List[dict]:
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(batch_df)

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.parse(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=BatchClassificationResponse,
            )

            parsed = completion.choices[0].message.parsed
            return [r.model_dump() for r in parsed.results]

        except Exception as e:
            wait_seconds = 2 ** attempt
            print(f"[WARN] LLM 호출 실패: attempt={attempt + 1}, error={e}")
            time.sleep(wait_seconds)

    return [
        {
            "merchant_id": str(row["merchant_id"]),
            "merchant_name": str(row["merchant_name"]),
            "predicted_category": "UNKNOWN",
            "confidence": 0.0,
            "decision_type": "NEEDS_REVIEW",
            "reason": "LLM 호출 실패로 인해 수기 검수가 필요합니다.",
            "is_ambiguous": True,
            "alternative_categories": [],
        }
        for _, row in batch_df.iterrows()
    ]


def decide_final_action(row: pd.Series) -> str:
    predicted_category = row["predicted_category"]
    confidence = float(row["confidence"])
    decision_type = row["decision_type"]
    is_ambiguous = bool(row["is_ambiguous"])

    if predicted_category not in CATEGORY_TAXONOMY:
        return "INVALID_CATEGORY_REVIEW"

    if (
        decision_type == "RECLASSIFIED"
        and confidence >= 0.75
        and not is_ambiguous
        and predicted_category not in ["기타", "UNKNOWN"]
    ):
        return "AUTO_APPLY"

    if confidence >= 0.50 or decision_type == "NEEDS_REVIEW":
        return "HUMAN_REVIEW"

    return "KEEP_OTHER"


def postprocess_results(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    df["final_action"] = df.apply(decide_final_action, axis=1)
    df["final_category"] = df.apply(
        lambda row: row["predicted_category"] if row["final_action"] == "AUTO_APPLY" else "기타",
        axis=1,
    )
    return df


def run_llm_reclassification_streaming(
    input_csv_path: str,
    output_csv_path: str,
    read_chunk_size: int = 10_000,
    llm_batch_size: int = 20,
    max_rows: int | None = None,
):
    """
    PoC용 streaming 처리.
    전체 데이터를 메모리에 올리지 않고 chunk 단위로 읽고 결과를 append 저장합니다.

    max_rows:
    - None이면 전체 처리
    - 10000처럼 지정하면 일부 샘플만 처리
    """

    required_cols = {"merchant_id", "merchant_name"}
    is_first_write = True
    processed = 0

    for chunk_df in pd.read_csv(input_csv_path, chunksize=read_chunk_size):
        missing_cols = required_cols - set(chunk_df.columns)
        if missing_cols:
            raise ValueError(f"입력 CSV에 필요한 컬럼이 없습니다: {missing_cols}")

        if max_rows is not None:
            remaining = max_rows - processed
            if remaining <= 0:
                break
            chunk_df = chunk_df.head(remaining)

        chunk_results = []

        for start in tqdm(range(0, len(chunk_df), llm_batch_size)):
            batch_df = chunk_df.iloc[start:start + llm_batch_size].copy()
            batch_results = classify_batch_with_llm(batch_df)
            chunk_results.extend(batch_results)

        result_df = pd.DataFrame(chunk_results)
        result_df = postprocess_results(result_df)

        result_df.to_csv(
            output_csv_path,
            mode="w" if is_first_write else "a",
            header=is_first_write,
            index=False,
            encoding="utf-8-sig",
        )

        is_first_write = False
        processed += len(chunk_df)

        print(f"processed rows: {processed}")

        if max_rows is not None and processed >= max_rows:
            break

    print("저장 완료:", output_csv_path)
