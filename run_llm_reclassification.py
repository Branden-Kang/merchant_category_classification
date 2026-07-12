import os
import json
import math
import pandas as pd


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


def build_system_prompt_for_batch() -> str:
    categories = "\n".join([f"- {c}" for c in CATEGORY_TAXONOMY])

    return f"""
당신은 금융 거래 데이터의 가맹점 카테고리 재분류 전문가입니다.

입력되는 가맹점들은 Rule 기반 분류와 ML 기반 분류 이후에도
"기타", "미분류", "UNKNOWN" 또는 낮은 신뢰도 상태로 남은 가맹점입니다.

가맹점명만 보고 재분류 가능한 경우 제공된 카테고리 체계 중 하나로 분류하고,
근거가 부족하면 "기타" 또는 "UNKNOWN"으로 유지하세요.

중요 규칙:
1. 반드시 카테고리 체계 안에서만 선택하세요.
2. 새로운 카테고리를 만들지 마세요.
3. 입력 정보는 가맹점명만 사용하세요.
4. 한글, 영어, 로마자 표기, 일본어, 중국어, 기타 외국어, 혼합 표기를 고려하세요.
5. 잘 알려진 브랜드이거나 상호명에 업종 단서가 명확한 경우에만 높은 confidence를 부여하세요.
6. 상호명이 일반적이거나 업종 단서가 부족하면 "기타" 또는 "UNKNOWN"으로 유지하세요.
7. 모든 가맹점을 억지로 재분류하지 마세요.
8. 반드시 JSON만 출력하세요.

출력 JSON 형식:
{{
  "results": [
    {{
      "merchant_id": "입력 merchant_id",
      "merchant_name": "입력 merchant_name",
      "predicted_category": "카테고리 체계 중 하나",
      "confidence": 0.0,
      "decision_type": "RECLASSIFIED | KEEP_OTHER | NEEDS_REVIEW",
      "reason": "한국어 한 문장",
      "is_ambiguous": true,
      "alternative_categories": ["대안 카테고리"]
    }}
  ]
}}

카테고리 체계:
{categories}
""".strip()


def build_batch_request(custom_id: str, merchant_items: list[dict]) -> dict:
    system_prompt = build_system_prompt_for_batch()

    user_content = json.dumps(
        {"merchants": merchant_items},
        ensure_ascii=False,
    )

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
        },
    }


def iter_merchant_groups(input_csv_path: str, read_chunk_size: int, merchants_per_request: int):
    for chunk_df in pd.read_csv(input_csv_path, chunksize=read_chunk_size):
        for start in range(0, len(chunk_df), merchants_per_request):
            batch_df = chunk_df.iloc[start:start + merchants_per_request]

            items = [
                {
                    "merchant_id": str(row["merchant_id"]),
                    "merchant_name": str(row["merchant_name"]),
                }
                for _, row in batch_df.iterrows()
            ]

            yield items


def create_batch_jsonl_files(
    input_csv_path: str,
    output_dir: str,
    read_chunk_size: int = 100_000,
    merchants_per_request: int = 20,
    max_requests_per_file: int = 40_000,
    max_file_size_mb: int = 180,
):
    """
    OpenAI Batch API 입력용 JSONL 파일을 여러 개 생성합니다.

    max_requests_per_file:
    - 공식 최대는 50,000 requests지만, 여유 있게 40,000 권장

    max_file_size_mb:
    - 공식 최대는 200MB지만, 여유 있게 180MB 권장
    """

    os.makedirs(output_dir, exist_ok=True)

    file_idx = 0
    request_idx = 0
    current_file_request_count = 0

    output_path = os.path.join(output_dir, f"batch_input_{file_idx:05d}.jsonl")
    f = open(output_path, "w", encoding="utf-8")

    for merchant_items in iter_merchant_groups(
        input_csv_path=input_csv_path,
        read_chunk_size=read_chunk_size,
        merchants_per_request=merchants_per_request,
    ):
        custom_id = f"merchant-group-{request_idx:012d}"

        request_obj = build_batch_request(
            custom_id=custom_id,
            merchant_items=merchant_items,
        )

        line = json.dumps(request_obj, ensure_ascii=False) + "\n"
        line_size_mb = len(line.encode("utf-8")) / (1024 * 1024)

        current_size_mb = f.tell() / (1024 * 1024)

        need_new_file = (
            current_file_request_count >= max_requests_per_file
            or current_size_mb + line_size_mb >= max_file_size_mb
        )

        if need_new_file:
            f.close()

            file_idx += 1
            current_file_request_count = 0

            output_path = os.path.join(output_dir, f"batch_input_{file_idx:05d}.jsonl")
            f = open(output_path, "w", encoding="utf-8")

        f.write(line)
        current_file_request_count += 1
        request_idx += 1

        if request_idx % 10_000 == 0:
            print(f"created requests: {request_idx:,}, current file: {output_path}")

    f.close()

    print(f"완료. 생성된 파일 수: {file_idx + 1}")
    print(f"총 request 수: {request_idx:,}")


from openai import OpenAI
import os
import glob
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def submit_batch_file(jsonl_path: str) -> dict:
    batch_input_file = client.files.create(
        file=open(jsonl_path, "rb"),
        purpose="batch",
    )

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    return {
        "jsonl_path": jsonl_path,
        "file_id": batch_input_file.id,
        "batch_id": batch.id,
        "status": batch.status,
    }


def submit_all_batch_files(input_dir: str, manifest_path: str):
    rows = []

    for jsonl_path in sorted(glob.glob(os.path.join(input_dir, "*.jsonl"))):
        print("submit:", jsonl_path)
        info = submit_batch_file(jsonl_path)
        rows.append(info)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("manifest saved:", manifest_path)


submit_all_batch_files(
    input_dir="batch_inputs",
    manifest_path="batch_manifest.json",
)


import json
import pandas as pd


def decide_final_action_record(record: dict) -> str:
    predicted_category = record.get("predicted_category", "UNKNOWN")
    confidence = float(record.get("confidence", 0.0))
    decision_type = record.get("decision_type", "NEEDS_REVIEW")
    is_ambiguous = bool(record.get("is_ambiguous", True))

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


def parse_batch_output_jsonl(output_jsonl_path: str, output_csv_path: str):
    is_first_write = True
    buffer = []
    buffer_size = 10_000

    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            response_line = json.loads(line)

            custom_id = response_line.get("custom_id")

            try:
                content = (
                    response_line["response"]["body"]["choices"][0]["message"]["content"]
                )
                parsed_content = json.loads(content)
                results = parsed_content.get("results", [])

                for r in results:
                    r["custom_id"] = custom_id
                    r["final_action"] = decide_final_action_record(r)
                    r["final_category"] = (
                        r["predicted_category"]
                        if r["final_action"] == "AUTO_APPLY"
                        else "기타"
                    )
                    buffer.append(r)

            except Exception as e:
                buffer.append({
                    "custom_id": custom_id,
                    "merchant_id": "",
                    "merchant_name": "",
                    "predicted_category": "UNKNOWN",
                    "confidence": 0.0,
                    "decision_type": "NEEDS_REVIEW",
                    "reason": f"Batch output parsing failed: {e}",
                    "is_ambiguous": True,
                    "alternative_categories": [],
                    "final_action": "HUMAN_REVIEW",
                    "final_category": "기타",
                })

            if len(buffer) >= buffer_size:
                pd.DataFrame(buffer).to_csv(
                    output_csv_path,
                    mode="w" if is_first_write else "a",
                    header=is_first_write,
                    index=False,
                    encoding="utf-8-sig",
                )
                is_first_write = False
                buffer = []

    if buffer:
        pd.DataFrame(buffer).to_csv(
            output_csv_path,
            mode="w" if is_first_write else "a",
            header=is_first_write,
            index=False,
            encoding="utf-8-sig",
        )
