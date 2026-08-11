# End-to-End 가맹점 카테고리 분류 파이프라인 구축 프롬프팅

1차 분류 모델(`merchant_hybrid_classifier`)과 2차 LLM 재분류(`run_llm_reclassification.py`)를
하나의 End-to-End 파이프라인으로 통합하는 작업을 코딩 에이전트에게 요청할 때 사용하는 프롬프트입니다.

- **전체 버전**: 실제 구현 요청 시 그대로 복사해서 사용
- **요약 버전**: 이미 컨텍스트를 공유한 상태에서 짧게 지시할 때 사용
- **부록**: 구현 변형과 후속 요청 프롬프트

전제 조건 (이 문서 전체에 적용):

- 데이터 전처리(정규화, 지점명 제거 등)는 하지 않습니다. RoBERTa와 LLM 모두 원문을 그대로 처리합니다.
- 수기 라벨 DB 우선 매칭 단계는 두지 않습니다.
- 2차 LLM은 계층 분류를 하지 않고 단일 카테고리만 예측합니다.
- Temperature Scaling은 적용하지 않습니다. 근거는 [5.3](#53-temperature-scaling을-적용하지-않는-이유)에 정리했습니다.

---

## 1. 전체 버전 프롬프트

```text
당신은 금융 데이터 ML 파이프라인 엔지니어입니다.
이 레포지토리에 흩어져 있는 1차 분류 모델과 2차 LLM 재분류 코드를
하나의 End-to-End 배치 파이프라인으로 통합해 주세요.

────────────────────────────────────────
[1] 배경

현재 레포지토리 상태는 다음과 같습니다.

- merchant_hybrid_classifier/
  KLUE-RoBERTa + UTF-8 Byte Transformer + Gated Fusion 구조의 1차 분류 모델입니다.
  main.py가 --mode {demo, train, test, inference} CLI를 제공합니다.
  --mode inference 실행 시 result/inference_result.csv를 생성하며 컬럼은 다음과 같습니다.

    id, text,
    top_pred, top_confidence,
    mid_pred, mid_confidence,
    fine_pred, fine_confidence,
    text_gate_mean,
    top_name, mid_name, fine_name,
    store_category_cd

  카테고리 체계는 meta_data/category_meta.csv의 대분류/중분류/세분류 3계층입니다.
  hybrid_dataset.py의 MerchantDataset은 추론 시 id, text 두 컬럼만 요구하고
  라벨 컬럼은 has_labels=True일 때만 검사합니다.
  따라서 추가 컬럼은 모델을 건드리지 않고 그대로 통과시킬 수 있습니다.

- run_llm_reclassification.py
  OpenAI Batch API 기반 2차 LLM 재분류 코드입니다.
  JSONL 생성 → 배치 제출 → 결과 파싱 → 후처리 판정의 흐름을 담고 있습니다.

- llm_merchant_reclassification_architecture.md
  2차 재분류의 아키텍처와 임계값 정책이 정리된 설계 문서입니다.
  구현 시 이 문서의 임계값을 기준으로 삼되, 코드와 문서가 충돌하면 문서를 우선하고
  충돌 지점을 보고해 주세요.

────────────────────────────────────────
[2] 목표

가맹점 데이터를 입력하면 아래 4가지 최종 상태 중 하나로 확정되어 나오는
단일 파이프라인을 구성합니다.

  1) 1차 모델 자동 확정
  2) 2차 LLM 자동 확정
  3) Human Review 대기
  4) 기타 / 미분류

핵심 라우팅 원칙:

  - 1차 모델에서 신뢰도가 높은 건은 LLM을 타지 않고 즉시 확정한다.
  - 1차 모델에서 신뢰도가 낮은 건만 2차 LLM으로 routing 한다.
  - 2차 LLM 결과 중 고신뢰 건만 자동 확정한다.
  - 중간 신뢰 건은 Human Review 큐로 보낸다.
  - 저신뢰 건은 기타(미분류)로 확정한다.

LLM 호출 비용이 전체 비용의 대부분을 차지하므로,
"LLM으로 몇 건이 흘러가는가"를 통제 가능한 설정값으로 만드는 것이 설계의 중심입니다.

────────────────────────────────────────
[3] 설계 전제 (임의로 바꾸지 말 것)

아래는 이미 결정된 사항입니다. 더 좋아 보이는 대안이 있어도 임의로 추가하지 말고,
필요하다고 판단되면 제안만 하고 확인을 받으세요.

1. 데이터 전처리를 하지 않습니다.
   정규화, 지점명 제거, 노이즈 제거, 전각/반각 변환 등을 수행하지 마세요.
   1차 모델(RoBERTa + Byte Encoder)과 2차 LLM 모두 원문을 그대로 처리할 수 있습니다.
   가맹점명은 파이프라인 전 구간에서 원문 그대로 유지되어야 합니다.

   예외는 하나뿐입니다. 비용 절감 목적의 중복 제거는 허용합니다.
   단, 반드시 원문 완전 일치(exact match) 기준이어야 합니다.
   원문을 변형해서 묶는 방식(대소문자 통일, 공백 제거 등)은 금지합니다.
   이 중복 제거는 결과를 바꾸지 않는 캐싱이며, config에서 끌 수 있어야 합니다.

2. 수기 라벨 DB 우선 매칭 단계를 만들지 마세요.

3. 2차 LLM은 계층 분류를 하지 않습니다.
   대분류/중분류/세분류를 각각 예측하게 하지 말고, 세분류에 해당하는
   단일 카테고리 하나만 예측하게 하세요.
   상위 계층이 필요하면 예측된 세분류로부터 category_meta.csv를 조회해 채우세요.

4. Temperature Scaling을 적용하지 마세요.
   대신 [6]의 방식으로 임계값을 검증 데이터에서 직접 산출하세요.

────────────────────────────────────────
[4] 반드시 먼저 해결해야 할 기존 코드 문제

구현 전에 아래 4가지를 확인하고 해결한 뒤 진행해 주세요.
임의로 넘어가지 말고, 해결 방식을 코드와 함께 설명해 주세요.

1. 학습 데이터 스키마와 실제 데이터 스키마의 불일치
   현재 학습 데이터는 (id, text, top_label, mid_label, fine_label) 구조입니다.
   실제 운영 데이터는 (id, text, label) 구조에 메타데이터 컬럼이 추가로 붙습니다.
     예: mcc_code, mcc_name, business_name, addr 등

   차이는 두 가지입니다.
     - 라벨이 3계층이 아니라 단일 label 컬럼 하나입니다.
     - 모델이 사용하지 않는 메타데이터 컬럼이 함께 들어옵니다.

   [5]의 Stage 0에 명시한 스키마 매핑 계층에서 이 차이를 흡수하세요.
   모델 코드나 Dataset 클래스를 수정해서 해결하지 마세요.

2. 파이프라인 내부 스키마 불일치
   1차 출력은 (id, text)이고 2차 LLM 입력은 (merchant_id, merchant_name)을 요구합니다.
   어댑터 계층에서 명시적으로 매핑하고, 원본 id를 파이프라인 끝까지 보존하세요.

3. 택소노미 불일치
   1차는 category_meta.csv 기반 3계층 체계이고,
   2차는 run_llm_reclassification.py에 하드코딩된 13개 평면 리스트입니다.
   category_meta.csv를 Single Source of Truth로 삼아 통합하세요.
   LLM 프롬프트에 주입할 카테고리 목록은 하드코딩하지 말고 메타 파일에서 로드하세요.
   두 체계를 즉시 통합할 수 없다면 taxonomy/taxonomy_map.csv 매핑 테이블을 만들고,
   매핑되지 않는 카테고리는 실패시키지 말고 리포트에 남기세요.

4. import 시 부작용
   run_llm_reclassification.py는 모듈 최상위에서 submit_all_batch_files(...)를 즉시 호출하고,
   OpenAI 클라이언트도 최상위에서 생성합니다.
   import만 해도 실제 배치가 제출되고 과금이 발생하는 구조입니다.
   모든 실행 코드를 함수 안으로 옮기고 if __name__ == "__main__" 가드를 추가하세요.
   클라이언트는 지연 생성(lazy)으로 바꾸세요.

────────────────────────────────────────
[5] 파이프라인 단계 정의

Stage 0. 스키마 매핑 및 적재
  이 단계는 전처리 단계가 아닙니다. 컬럼을 맞추는 단계이며 text 값은 변경하지 않습니다.

  (a) 컬럼명 매핑
      실제 데이터의 컬럼명이 파이프라인 표준명과 다를 수 있으므로,
      config의 schema 블록에서 매핑을 정의하게 하세요.

        schema:
          id_column: id
          text_column: text
          label_column: label          # 없으면 null (추론 전용)
          metadata_columns:
            - mcc_code
            - mcc_name
            - business_name
            - addr

      metadata_columns는 위 목록으로 고정하지 말고, config에 적힌 대로 받아들이세요.
      명시되지 않은 잔여 컬럼도 버리지 말고 그대로 보존할지 여부를
      config의 keep_unmapped_columns 옵션으로 제어하세요.

  (b) 라벨 매핑 (학습 및 평가 입력에만 해당)
      실데이터의 단일 label을 3계층 라벨로 확장합니다.
      category_meta.csv와 조인해 fine_label을 얻고, 같은 행에서 mid_label, top_label을 가져옵니다.
      조인 키는 config에서 지정하게 하세요.

        label_mapping:
          join_key: store_category_cd   # 또는 fine_label, fine_name
      
      매핑 실패 행은 조용히 버리지 말고 별도 파일(00_unmapped_labels.csv)로 떨어뜨리고
      건수를 리포트에 기록하세요.
      추론 입력에는 label이 없어도 되며, 없을 때 이 단계는 건너뜁니다.

  (c) 메타데이터 분리 보존
      모델에는 (id, text)만 전달합니다.
      메타데이터는 id를 키로 하는 별도 테이블로 분리해 보관하고,
      최종 결과 병합 시 다시 붙이세요.
      메타데이터가 모델 입력에 섞여 들어가지 않도록 하세요.

  (d) 중복 제거 (선택, 기본 활성)
      text 원문 완전 일치 기준으로만 중복을 제거합니다.
      원본 id ↔ 대표 행 매핑 테이블을 보존해 마지막에 원본 행 수를 그대로 복원하세요.
      메타데이터가 서로 다른데 text만 같은 경우가 있으므로,
      메타데이터를 LLM에 제공하는 설정(아래 Stage 3)이 켜져 있으면
      중복 제거 키에 메타데이터를 포함할지 여부를 config로 선택하게 하세요.
      기본값은 "text만으로 묶기"이며, 이 경우 대표 행의 메타데이터를 사용합니다.

Stage 1. 1차 모델 추론
  - merchant_hybrid_classifier의 추론 로직을 재사용합니다.
  - main.py를 복사하지 말고 함수로 import 하거나, 필요하면 최소한으로 리팩터링하세요.
  - 입력은 (id, text)만 전달합니다.
  - 결과에 fine_confidence를 포함한 전체 예측 컬럼을 유지하세요.

Stage 2. 라우팅 판정
  - 아래 [6]의 규칙에 따라 각 건을 STAGE1_AUTO 또는 LLM_ROUTE로 분기합니다.
  - 분기 결과 건수와 비율을 로그와 리포트에 남기세요.

Stage 3. 2차 LLM 재분류
  - LLM_ROUTE로 분기된 건만 대상으로 합니다.
  - OpenAI Batch API를 사용합니다. 기존 run_llm_reclassification.py의 방식을 유지하세요.
  - 배치 입력 JSONL 생성 → 제출 → 상태 폴링 → 결과 수집 → 파싱

  (a) 출력 형식
      단일 카테고리 하나만 예측하게 하세요. 계층별로 나누어 예측하게 하지 마세요.
      응답 스키마는 기존 코드를 따릅니다.
        merchant_id, merchant_name, predicted_category, confidence,
        decision_type(RECLASSIFIED | KEEP_OTHER | NEEDS_REVIEW),
        reason, is_ambiguous, alternative_categories

  (b) 카테고리 후보 제시 방식
      기본은 category_meta.csv의 세분류 전체 목록을 프롬프트에 주입하는 것입니다.
      1차 모델의 top-k 후보만 제시하는 방식은 기본값으로 쓰지 마세요.
      LLM으로 넘어온 건은 애초에 1차 모델이 확신하지 못한 건이므로,
      그 모델의 후보로 선택지를 좁히면 1차의 오류를 그대로 상속합니다.
      토큰 비용이 문제가 되면 config의 llm.candidate_mode: all | topk 로
      전환 가능하게만 만들고, 기본값은 all로 두세요.

  (c) 메타데이터 활용
      config의 llm.use_metadata가 켜져 있으면
      mcc_name, business_name, addr 등을 가맹점명과 함께 프롬프트에 제공하세요.
      이때 프롬프트의 "입력 정보는 가맹점명만 사용하세요" 규칙을 그대로 두면 모순되므로,
      제공되는 필드를 명시하는 문구로 교체해야 합니다.
      메타데이터는 결측이 잦으므로 값이 없으면 해당 필드를 아예 넣지 말고,
      "없음" 같은 문자열을 채워 넣지 마세요.
      기본값은 use_metadata: true 로 두되,
      false로 실행한 결과와 비교할 수 있도록 두 설정 모두 동작해야 합니다.

  (d) 1차 예측 힌트
      1차 모델의 예측 카테고리와 confidence를 참고 정보로 함께 제공하되,
      "1차 예측은 참고용이며 그대로 따를 필요가 없다"고 프롬프트에 명시하세요.

  (e) 공통 설정
      temperature=0, JSON 강제 출력을 유지하세요.

Stage 4. 최종 판정 및 병합
  - 아래 [6]의 규칙에 따라 final_status를 확정합니다.
  - 중복 제거된 결과를 원본 전체 행에 다시 join 합니다.
  - Stage 0에서 분리해 둔 메타데이터를 다시 붙입니다.
  - 최종 CSV와 요약 리포트를 생성합니다.

────────────────────────────────────────
[6] 라우팅 및 판정 규칙

■ 1차 라우팅 (Stage 2)

  fine_confidence >= T1_HIGH           → STAGE1_AUTO   (1차 결과로 자동 확정)
  fine_confidence <  T1_HIGH           → LLM_ROUTE     (2차 LLM으로 routing)

  - T1_HIGH는 설정 파일 값으로 두고, 기본값은 0.90으로 시작합니다.
  - 이 값은 반드시 검증 데이터에서 산출해야 합니다. tune_threshold.py를 함께 제공하세요.
      · 검증셋에서 fine_confidence 기준으로 임계값을 0.50~0.99까지 스윕합니다.
      · 각 지점의 자동 확정 정밀도, 커버리지, LLM 전송 건수, 예상 비용을 표로 출력합니다.
      · 목표 정밀도(config의 target_precision, 기본 0.95)를 만족하는
        최소 임계값을 추천하고, 그 값을 config에 기록합니다.
  - Temperature Scaling이나 별도의 확률 보정을 적용하지 마세요.
    라우팅에 필요한 것은 "확률값의 절대적 의미"가 아니라 "임계값 통과 여부"이며,
    임계값 자체를 검증 데이터에서 목표 정밀도로 직접 잡으면 보정 단계는 불필요합니다.
    자세한 근거는 이 요청의 참고 문서 [5.3]에 있습니다.

■ 2차 LLM 판정 (Stage 4)

  판정 우선순위는 위에서 아래 순서로 적용합니다.

  1. predicted_category가 택소노미에 없음
       → HUMAN_REVIEW  (사유: INVALID_CATEGORY)

  2. 응답 누락 또는 JSON 파싱 실패
       → HUMAN_REVIEW  (사유: PARSE_FAILED)
       ※ 절대 조용히 버리거나 자동 확정하지 말 것

  3. decision_type == "RECLASSIFIED"
     and confidence >= T2_AUTO
     and is_ambiguous == false
     and predicted_category not in ("기타", "UNKNOWN")
       → AUTO_CONFIRMED_STAGE2

  4. confidence >= T2_REVIEW  또는 decision_type == "NEEDS_REVIEW"
       → HUMAN_REVIEW

  5. 그 외 (confidence < T2_REVIEW 또는 decision_type == "KEEP_OTHER")
       → UNCLASSIFIED  (기타 / 미분류로 확정)

  기본 임계값 (llm_merchant_reclassification_architecture.md 기준):
    T2_AUTO   = 0.75
    T2_REVIEW = 0.50

  ※ LLM이 스스로 보고하는 confidence는 통계적으로 보정된 값이 아닙니다.
    이 임계값도 초기값일 뿐이므로, 검수 결과가 쌓이면 재조정할 수 있게
    설정값으로만 두고 판정 함수를 순수 함수로 분리하세요.

■ 최종 상태 enum

  AUTO_CONFIRMED_STAGE1   1차 모델 고신뢰 자동 확정
  AUTO_CONFIRMED_STAGE2   2차 LLM 고신뢰 자동 확정
  HUMAN_REVIEW            사람 검수 대기
  UNCLASSIFIED            기타 / 미분류

  - final_category는 AUTO_CONFIRMED_* 인 경우에만 실제 카테고리를 채웁니다.
  - HUMAN_REVIEW와 UNCLASSIFIED는 final_category를 "기타"로 두되,
    HUMAN_REVIEW 건은 LLM이 제안한 카테고리를 suggested_category 컬럼에 별도 보존하세요.
    검수자가 판단 근거 없이 처음부터 다시 시작하게 만들지 않기 위함입니다.
  - 모든 임계값은 코드 상수가 아니라 설정 파일 값이어야 합니다.

────────────────────────────────────────
[7] 산출물

■ 디렉터리 구조 (권장, 조정 가능하나 이유를 설명할 것)

  merchant_pipeline/
    config/pipeline_config.yaml     모든 임계값·경로·스키마·모델 설정
    taxonomy/taxonomy_map.csv       1차 ↔ 2차 카테고리 매핑
    schema_map.py                   Stage 0 (컬럼 매핑, 라벨 확장, 메타 분리, 중복 제거)
    stage1_infer.py                 Stage 1
    routing.py                      Stage 2
    stage2_llm/
      prompt.py                     프롬프트 빌더 (택소노미·메타데이터 동적 주입)
      schema.py                     응답 스키마 및 검증
      build_batch.py                JSONL 생성
      submit_batch.py               배치 제출
      poll_batch.py                 상태 폴링 및 결과 다운로드
      parse_batch.py                결과 파싱
    finalize.py                     Stage 4
    run_pipeline.py                 오케스트레이터 CLI
    tune_threshold.py               T1_HIGH 산출 스크립트
    README.md

■ 오케스트레이터 CLI

  python -m merchant_pipeline.run_pipeline \
    --input data/merchants.csv \
    --config config/pipeline_config.yaml \
    --output-dir outputs/run_20260812 \
    --stage all

  요구사항:
  - --stage 로 특정 단계만 실행 가능 (schema / stage1 / route / llm / finalize / all)
  - --resume 으로 중단 지점부터 재개 가능
  - --dry-run 으로 LLM을 실제 호출하지 않고 라우팅 건수와 예상 비용만 출력
    (이 옵션은 필수입니다. 비용 사고를 막는 안전장치입니다.)
  - --max-llm-items 로 LLM 전송 건수 상한 지정 가능

■ 출력 파일

  outputs/<run_id>/
    00_unmapped_labels.csv      라벨 매핑 실패 행 (있을 때만)
    01_mapped.csv               스키마 매핑 및 중복 제거 결과
    01_metadata.csv             id 기준 메타데이터 분리 보관본
    02_stage1_pred.csv          1차 모델 예측 전체
    03_routing.csv              라우팅 분기 결과
    04_llm_input/*.jsonl        배치 입력
    05_llm_raw/*.jsonl          배치 원본 응답 (반드시 원본 그대로 보존)
    06_llm_parsed.csv           파싱 결과
    07_final.csv                최종 결과 (원본 전체 행)
    run_report.md               요약 리포트
    run_state.json              단계별 진행 상태 (resume용)

■ 최종 CSV 스키마 (07_final.csv)

  id
  text                    원문 그대로. 변형본을 만들지 말 것
  final_category
  final_status
  decided_by              STAGE1 | STAGE2_LLM
  stage1_category
  stage1_confidence
  stage2_category
  stage2_confidence
  suggested_category
  review_reason
  reason
  (+ config의 metadata_columns 전체를 원본 값 그대로 뒤에 붙임)

────────────────────────────────────────
[8] 비기능 요구사항

- 멱등성: 같은 입력으로 다시 실행하면 같은 결과가 나와야 합니다.
- 재개: 각 단계 완료 시 run_state.json을 갱신하고, --resume 시 완료 단계를 건너뜁니다.
- 배치 실패 처리: 배치가 failed/expired 상태이면 해당 파일만 재제출 가능해야 합니다.
- 메모리: 수백만 건 입력을 가정하고 chunk 단위로 처리하세요. 전체 로드 금지.
- 로깅: 각 단계의 입력 건수, 출력 건수, 소요 시간을 남기세요.
  단계 간 건수가 맞지 않으면 경고를 출력하세요.
- 비용: --dry-run에서 LLM 전송 건수, 예상 토큰, 예상 비용을 출력하세요.
  메타데이터를 프롬프트에 넣으면 토큰이 늘어나므로 이 추정에 반영하세요.
- 인코딩: 한글 CSV는 utf-8-sig로 저장하세요.
- 비밀정보: API 키는 환경변수로만 읽고, 로그와 결과 파일에 절대 남기지 마세요.
  addr 등 메타데이터는 개인정보가 될 수 있으므로 로그에 원문을 남기지 마세요.

────────────────────────────────────────
[9] 테스트

pytest 기준으로 아래를 반드시 포함하세요.

- 스키마 매핑 테스트
  · 컬럼명이 다른 실데이터가 표준명으로 매핑되는지
  · 단일 label이 3계층 라벨로 확장되는지
  · 매핑 실패 행이 00_unmapped_labels.csv로 분리되고 본류에서 제외되는지
  · label 컬럼이 없는 추론 입력이 정상 처리되는지
  · 메타데이터 컬럼이 모델 입력에 섞여 들어가지 않는지
- text 원문 보존 테스트
  공백·대소문자·특수문자가 포함된 가맹점명이 파이프라인 전 구간에서
  단 한 글자도 변경되지 않는지 확인 (전처리 금지 요구사항의 회귀 방지)
- 중복 제거 테스트
  원문 완전 일치만 묶이는지, 원본 행 수가 그대로 복원되는지
- 라우팅 판정 테스트: T1_HIGH 경계값(초과/같음/미만) 동작
- 2차 판정 테스트: [6]의 5개 분기 각각에 대한 케이스
- 파싱 실패가 HUMAN_REVIEW로 떨어지는지 확인하는 테스트
- 택소노미 밖 카테고리가 HUMAN_REVIEW로 떨어지는지 확인하는 테스트
- 프롬프트 빌더 테스트: use_metadata on/off 각각에서
  결측 메타데이터 필드가 프롬프트에 포함되지 않는지
- LLM 호출은 실제로 하지 말고 mock 응답으로 대체하세요.
- 샘플 데이터로 --dry-run 전체 실행이 통과하는 스모크 테스트

────────────────────────────────────────
[10] 하지 말아야 할 것

- 기존 merchant_hybrid_classifier의 모델 구조나 학습 로직을 변경하지 마세요.
  이번 작업은 파이프라인 통합이며 모델 변경이 아닙니다.
- 1차 모델을 재학습하지 마세요.
- 가맹점명 전처리·정규화를 하지 마세요.
- Temperature Scaling을 넣지 마세요.
- 수기 라벨 DB 매칭 단계를 만들지 마세요.
- 2차 LLM에 계층 분류를 시키지 마세요.
- 임계값을 코드에 하드코딩하지 마세요.
- LLM에 전 건을 보내지 마세요. 라우팅된 건만 보냅니다.
- 애매한 건을 억지로 자동 확정하지 마세요. 미분류로 남기는 것이 정상 동작입니다.
- 실제 배치 API를 테스트 목적으로 호출하지 마세요.
- 기존 파일을 지우지 말고, 필요하면 그대로 두고 새 모듈에서 감싸세요.

────────────────────────────────────────
[11] 작업 순서

1. 먼저 기존 코드를 읽고, [4]의 4가지 문제에 대한 해결 방안을 정리해서 보고하세요.
2. config 스키마(schema 블록, 임계값, llm 옵션)와 통합 택소노미를 먼저 확정하세요.
3. 파이프라인 골격과 판정 로직(순수 함수)을 구현하고 테스트를 먼저 통과시키세요.
4. 1차 모델 연동, 2차 LLM 연동 순으로 붙이세요.
5. 샘플 데이터로 --dry-run 전체 실행을 검증하세요.
6. README에 실행 방법, config 각 항목의 의미, 임계값 조정 가이드를 작성하세요.

구현 중 판단이 갈리는 지점이 있으면 임의로 결정하지 말고
선택지와 근거를 제시한 뒤 확인을 받으세요.
```

---

## 2. 요약 버전 프롬프트

이미 레포지토리 컨텍스트를 공유한 상태에서 짧게 지시할 때 사용합니다.

```text
merchant_hybrid_classifier(1차 분류)와 run_llm_reclassification.py(2차 LLM 재분류)를
하나의 End-to-End 배치 파이프라인 merchant_pipeline/ 으로 통합해 주세요.

흐름:
  입력 CSV → 스키마 매핑 → 1차 모델 추론 → 라우팅 → 2차 LLM(Batch API) → 최종 판정

스키마 매핑 (전처리 아님, 컬럼만 맞추는 단계):
- 실데이터는 (id, text, label) + 메타데이터(mcc_code, mcc_name, business_name, addr 등) 구조이고
  학습 데이터는 (id, text, top_label, mid_label, fine_label) 구조입니다.
- 단일 label은 category_meta.csv와 조인해 3계층 라벨로 확장하세요. 조인 키는 config로 지정.
- 매핑 실패 행은 버리지 말고 별도 파일로 분리하고 건수를 리포트에 남기세요.
- 컬럼명 매핑은 config의 schema 블록에서 정의하게 하세요.
- 모델에는 (id, text)만 전달하고, 메타데이터는 id 기준으로 분리 보관 후 최종 결과에 재결합하세요.

라우팅 규칙:
  1차 fine_confidence >= T1_HIGH(기본 0.90)  → 1차 결과로 자동 확정, LLM 미호출
  1차 fine_confidence <  T1_HIGH             → 2차 LLM으로 routing

2차 판정 규칙 (위에서부터 순서대로 적용):
  택소노미 밖 카테고리 또는 파싱 실패          → HUMAN_REVIEW
  RECLASSIFIED & conf >= 0.75 & !is_ambiguous → AUTO_CONFIRMED_STAGE2
  conf >= 0.50 또는 NEEDS_REVIEW              → HUMAN_REVIEW
  그 외                                        → UNCLASSIFIED (기타)

하지 말 것:
- 가맹점명 전처리·정규화 금지. RoBERTa와 LLM 모두 원문을 그대로 처리합니다.
  비용 절감용 중복 제거만 예외로 허용하되, 원문 완전 일치 기준으로만 묶으세요.
- Temperature Scaling 금지. T1_HIGH를 검증셋에서 목표 정밀도(기본 0.95) 기준으로
  직접 산출하는 tune_threshold.py를 대신 제공하세요.
- 수기 라벨 DB 매칭 단계 만들지 말 것.
- 2차 LLM에 계층 분류 시키지 말 것. 세분류 단일 카테고리 하나만 예측하게 하세요.
- 모델 구조 변경 및 재학습 금지.

반드시 지킬 것:
- 모든 임계값과 옵션은 config/pipeline_config.yaml 값으로. 하드코딩 금지.
- run_llm_reclassification.py의 모듈 최상위 배치 제출 호출을 제거하고
  함수 + __main__ 가드로 리팩터링할 것. (지금은 import만 해도 과금됩니다.)
- 1차 출력 (id, text) ↔ 2차 입력 (merchant_id, merchant_name) 어댑터를 명시적으로 둘 것.
- LLM 프롬프트의 카테고리 목록은 category_meta.csv에서 동적으로 주입할 것.
- config의 llm.use_metadata로 mcc_name, business_name, addr을 LLM에 제공할지 선택하게 할 것.
  켜질 경우 기존 프롬프트의 "가맹점명만 사용하세요" 문구를 반드시 수정할 것.
- --dry-run(LLM 미호출, 라우팅 건수·예상 비용만 출력)과 --resume을 지원할 것.
- 판정 로직은 순수 함수로 분리하고 경계값 테스트를 작성할 것.
- text 원문이 전 구간에서 변경되지 않음을 보장하는 회귀 테스트를 작성할 것.
```

---

## 3. 부록

### 3.1 동기 호출 방식으로 구현할 때 추가할 문단

Batch API(최대 24시간 대기) 대신 즉시 응답이 필요한 경우 위 프롬프트에 덧붙입니다.

```text
Stage 3은 Batch API 대신 동기 호출 방식으로도 실행할 수 있어야 합니다.
config의 llm.mode 값으로 batch / sync를 선택하게 하고,
sync 모드에서는 다음을 구현하세요.

- 동시 요청 수 제한 (기본 8)
- 429 및 5xx에 대한 지수 백오프 재시도 (최대 5회)
- 요청 단위 타임아웃
- 부분 실패 시 성공분은 보존하고 실패분만 재시도 대상으로 기록

sync 모드는 소량 검증용이며, 대량 처리 기본값은 batch 모드로 두세요.
```

### 3.2 메타데이터 활용 효과를 검증할 때 추가할 문단

`use_metadata` 기본값을 확정하기 전에 실측이 필요할 때 사용합니다.

```text
LLM 입력에 메타데이터를 포함하는 것이 실제로 도움이 되는지 검증하는
비교 실행 스크립트를 추가해 주세요.

- 정답이 있는 샘플 500~1000건을 대상으로 합니다.
- 동일 샘플에 대해 use_metadata: false / true 두 설정으로 각각 실행합니다.
- 다음을 비교표로 출력하세요.
    · 자동 확정 정밀도
    · 자동 확정 커버리지
    · HUMAN_REVIEW 비율
    · UNCLASSIFIED 비율
    · 요청당 평균 토큰 및 총비용
- 메타데이터 필드별 결측률도 함께 출력하세요.
  결측이 많은 필드는 프롬프트 토큰만 늘리고 효과가 없을 수 있습니다.
- addr처럼 값이 긴 필드는 그대로 넣는 방식과 앞부분만 넣는 방식을 비교해도 좋습니다.
  단, 이는 가맹점명이 아닌 메타데이터에 대한 처리이므로 전처리 금지 원칙과 무관합니다.
```

### 3.3 후속 요청 프롬프트

파이프라인 구축 이후 단계적으로 요청할 항목입니다.

```text
[임계값 튜닝]
검증 데이터로 T1_HIGH를 0.50~0.99까지 0.01 간격으로 스윕하면서
각 지점의 자동 확정 정밀도, 커버리지, LLM 전송 건수, 예상 비용을 표로 만들어 주세요.
목표 정밀도 0.95를 만족하는 최소 임계값을 추천하고 근거를 설명해 주세요.
카테고리별로 정밀도 편차가 큰지도 함께 보여 주세요.
편차가 크면 단일 임계값 대신 카테고리별 임계값을 검토합니다.

[Human Review 큐]
HUMAN_REVIEW로 분류된 건을 검수하는 최소 인터페이스를 만들어 주세요.
검수 화면에는 가맹점명과 함께 mcc_name, business_name, addr 메타데이터,
1차 예측과 LLM 제안(suggested_category), 그리고 판단 근거(reason)를 함께 보여 주세요.

[피드백 루프]
확정된 라벨을 1차 모델 재학습 데이터로 환류하는 스크립트를 만들어 주세요.
LLM 자동 확정 건과 사람 검수 확정 건의 신뢰도 가중치를 구분해서 다루세요.
환류 데이터는 학습 스키마(id, text, top_label, mid_label, fine_label)로 변환되어야 합니다.

[모니터링]
실행마다 각 최종 상태의 건수 분포, 1차/2차 통과율, 건당 비용을 기록하고
직전 실행 대비 변화가 임계치를 넘으면 경고하는 리포트를 추가해 주세요.
```

---

## 4. 프롬프트 작성 시 반영한 판단 근거

| 항목 | 근거 |
|---|---|
| 스키마 매핑을 Stage 0으로 분리 | 학습 데이터는 3계층 라벨, 실데이터는 단일 label + 메타데이터 구조. 모델 코드를 고치지 않고 어댑터에서 흡수하는 것이 변경 범위를 최소화 |
| 메타데이터를 모델에 넣지 않고 통과만 시킴 | `hybrid_dataset.py`의 `MerchantDataset`은 추론 시 `id`, `text`만 요구하므로 추가 컬럼은 모델을 건드리지 않고 보존 가능 |
| 라벨 조인 키를 config로 노출 | `category_meta.csv`에 `store_category_cd`와 `fine_label`, `fine_name`이 모두 있어 실데이터 `label`이 어느 값인지에 따라 조인 키가 달라짐 |
| 전처리 금지를 명시적 요구사항 + 회귀 테스트로 고정 | 기존 프롬프트 문서들이 "지점명·지역명 제거" 규칙을 담고 있어, 에이전트가 관성적으로 정규화 코드를 넣을 가능성이 높음 |
| 중복 제거만 예외로 허용 | 원문 완전 일치 기준 dedup은 모델 입력을 변형하지 않으므로 결과가 동일. 전처리가 아니라 캐싱 |
| LLM 카테고리 후보를 top-k가 아닌 전체로 | LLM으로 넘어온 건은 1차 모델이 확신하지 못한 건이므로, 그 모델의 top-k로 선택지를 좁히면 1차 오류를 상속 |
| 2차 임계값 0.75 / 0.50 | `llm_merchant_reclassification_architecture.md`의 정책값이며 `run_llm_reclassification.py`의 `decide_final_action_record`와 일치 |
| import 부작용 제거를 최우선 항목으로 배치 | `run_llm_reclassification.py` 하단에서 `submit_all_batch_files(...)`가 모듈 최상위에서 실행되어, 재사용을 위해 import하는 순간 실제 배치가 제출되고 과금됨 |
| `--dry-run` 필수화 | 위 문제와 결합될 경우 실수 한 번의 비용이 크므로 안전장치를 요구사항으로 고정 |
| 파싱 실패를 HUMAN_REVIEW로 | 기존 코드도 동일하게 처리하고 있으며, 조용한 유실 방지 목적 |
| `suggested_category` 컬럼 추가 | 기존 코드는 자동 확정이 아닌 건의 `final_category`를 일괄 "기타"로 덮어써 LLM 제안이 유실됨. 검수자가 근거 없이 처음부터 판단하게 되는 것을 방지 |

---

## 5. Temperature Scaling 재검토

적용 후 성능이 떨어졌다는 관측에 대한 분석입니다. 결론부터 적으면 **이 파이프라인에는 필요 없습니다.**

### 5.1 Temperature Scaling은 정확도를 바꿀 수 없다

Temperature Scaling은 로짓 벡터를 양의 스칼라 `T`로 나누는 변환입니다.

```text
p = softmax(z / T),   T > 0
```

`T`는 모든 클래스에 동일하게 적용되는 스칼라이므로 로짓의 **대소 관계가 보존**되고,
따라서 `argmax`가 바뀌지 않습니다. 예측 라벨이 한 건도 바뀌지 않으므로
Top-1 Accuracy, Macro F1 등 예측 라벨만으로 계산되는 지표는 **수학적으로 변할 수 없습니다.**

즉 "Temperature Scaling을 적용했더니 성능이 떨어졌다"는 관측은
정확도 계열 지표가 떨어진 것이 아니라 다른 것이 떨어진 것입니다.

### 5.2 무엇이 떨어졌을 가능성이 있는가

| 관측된 현상 | 원인 | 판단 |
|---|---|---|
| 고정 임계값(예: 0.90) 통과 건수가 급감 | `T > 1`로 확률이 전반적으로 낮아져 같은 임계값을 넘는 건이 줄어듦 | **가장 가능성 높음.** 모델이 나빠진 것이 아니라 임계값이 더 이상 같은 운영점이 아닌 것 |
| 자동 확정 커버리지 하락, LLM 전송량 증가 | 위와 동일한 원인 | 임계값을 다시 잡으면 회복됨 |
| ECE / NLL이 오히려 나빠짐 | `T`를 학습 데이터나 분포가 다른 셋에서 적합했거나, 클래스 불균형이 심한 상태에서 단일 `T`로 적합 | 검증셋 구성 문제. 보정 방법 자체의 문제가 아님 |
| 세분류 성능만 나빠짐 | 대/중/세 3개 헤드에 단일 `T`를 공유해 적용 | 헤드별로 `T`를 따로 적합해야 함 |

특히 첫 두 줄은 임계값을 보정 이전 척도에 맞춰둔 채 확률만 바꿨을 때
필연적으로 나타나는 현상입니다. 보정과 임계값은 반드시 함께 조정해야 합니다.

### 5.3 그럼에도 적용하지 않는 이유

임계값을 다시 잡으면 회복된다는 것은, 뒤집으면 **보정 단계가 없어도 된다**는 뜻입니다.

이 파이프라인이 confidence로부터 필요로 하는 것은 단 하나, "임계값을 넘는가"입니다.
확률값이 실제 정답률과 일치하는지(보정)는 사용하지 않습니다.

- 이진 분류에서 Temperature Scaling은 로짓 마진의 단조 변환이므로 **순위를 정확히 보존**합니다.
- 다중 분류에서는 최대 확률이 전체 로짓 벡터에 의존하므로 순위가 엄밀히 보존되지는 않지만,
  실무상 거의 바뀌지 않습니다.

따라서 보정 전후의 **정밀도-커버리지 곡선이 사실상 동일**하고,
어느 쪽이든 임계값만 다시 잡으면 같은 운영점에 도달합니다.
보정은 도달 가능한 운영점의 집합을 넓혀주지 않고, 임계값의 숫자만 바꿉니다.

결론적으로 검증셋에서 목표 정밀도 기준으로 `T1_HIGH`를 직접 산출하는 방식이
보정 단계를 대체하며, 단계가 하나 줄어 유지보수 대상도 줄어듭니다.
프롬프트의 `tune_threshold.py`가 이 역할을 합니다.

### 5.4 보정이 다시 필요해지는 조건

아래에 해당하게 되면 재검토하세요. 현재는 어느 것도 해당하지 않습니다.

- confidence를 확률로 해석해 기대 비용을 계산해야 할 때
  (예: "이 임계값이면 오분류 건당 손실이 얼마"를 금액으로 산출)
- 1차 모델과 2차 LLM의 confidence를 같은 척도에서 비교해야 할 때
- 외부 보고나 감사 대응에서 confidence 수치 자체를 근거로 제시해야 할 때
- 모델을 교체했을 때 기존 임계값을 그대로 재사용하고 싶을 때

이 경우에도 Temperature Scaling보다 **Isotonic Regression**을 먼저 검토하세요.
`fine_confidence → 정답 여부`에 단조 회귀를 적합하는 방식으로,
단조 변환이므로 라우팅 순위를 보존하면서 확률 해석만 얻을 수 있고,
단일 스칼라 `T`보다 유연해 클래스 불균형 상황에 잘 맞습니다.

### 5.5 재현이 필요할 때 확인할 것

Temperature Scaling 실험을 다시 돌려 원인을 확정하려면 다음을 점검하세요.

1. `T`를 학습셋이 아니라 **별도 검증셋**에서 적합했는가
2. 대/중/세 헤드별로 `T`를 **따로** 적합했는가
3. 비교 대상 지표가 정확도인가, 임계값 의존 지표인가
   (정확도가 변했다면 구현 오류입니다. 순수 Temperature Scaling은 정확도를 바꿀 수 없습니다.)
4. 보정 전후 각각에서 임계값을 **다시 최적화한 뒤** 비교했는가
   같은 임계값으로 비교했다면 그 비교는 성립하지 않습니다.
