# 가맹점 카테고리 분류 모델링 논문 / 산업 사례 비교 및 고도화 제안

> 작성일: 2026-07-14  
> 목적: 2023년 2월 학습 이후 갱신되지 않았고 재학습 파이프라인도 유실된 기존 SubwordCNN/ML 모델을 전제로, **재현 가능한 신규 1차 분류 모델을 다시 구축하고 LLM을 기타·저신뢰 가맹점 재분류기로 결합하는 1개월 MVP**를 설계한다.

---

## 0. 문서 범위와 핵심 결론

### 0.1 현재 상황

현재 가맹점 카테고리 분류 환경은 다음과 같이 정리된다.

```text
- 기존 SubwordCNN / ML 분류 모델은 2023년 2월 학습됨
- 이후 신규 가맹점·브랜드·거래 표현 변화가 반영되지 않음
- 최근 데이터에서 오분류가 다수 발생함
- 담당자 변경·퇴사로 학습 코드, 데이터 생성 절차, 모델 배포 절차가 남아 있지 않음
- 모델 재학습과 성능 재현이 어려움
- LLM을 기타 가맹점 재분류에 적용했을 때 효과가 확인됨
- 수기 정답 가맹점 데이터가 존재함
```

이 조건에서는 기존 모델을 중심으로 LLM만 추가하는 방식이 적절하지 않다. 기존 모델의 성능과 데이터 적합성을 신뢰하기 어렵고, 재학습이 불가능해 향후 유지보수도 할 수 없기 때문이다.

### 0.2 최종 판단

권장 방향은 다음 두 작업을 **병렬로 추진**하는 것이다.

```text
Track A. 신규 1차 분류 모델과 재학습 파이프라인 재구축
    - 데이터셋 생성
    - 학습/평가 코드
    - 모델 버전 관리
    - 배치 추론
    - 성능 모니터링

Track B. LLM 재분류 모듈 구축
    - 신규 1차 모델의 기타·저신뢰 결과 추출
    - 임베딩 기반 후보 카테고리 검색
    - 후보 제한형 LLM 재분류
    - 자동 반영 / 검수 / 기타 유지 라우팅
```

1개월 안에 현실적으로 목표로 할 수 있는 결과는 **완전한 운영 전환이 아니라 재현 가능한 신규 모델과 shadow/backfill 가능한 통합 MVP**다.

### 0.3 권장 최종 역할 분리

| 구성요소 | 권장 역할 |
|---|---|
| Rule / Dictionary | 정확도가 검증된 항목의 early exit, 모델 feature, 충돌 탐지 |
| 신규 1차 분류 모델 | 전체 가맹점의 기본 카테고리 예측과 Top-K 후보 생성 |
| 임베딩 검색 | 유사 수기 정답 가맹점과 Category Label Card 검색 |
| LLM | 기타·저신뢰·모델 충돌·신규 가맹점의 재분류 |
| Human Review | 근거 부족, 고위험 카테고리, 모델 간 충돌 사례 확인 |
| Feedback Store | 검수 정답과 고신뢰 결과를 다음 재학습 데이터로 축적 |
| 기존 2023년 모델 | 신규 모델과 비교하기 위한 legacy baseline 또는 shadow 신호로만 사용 |

### 0.4 이번 문서에서 제외하는 독립 과제

다음 항목은 중요하지만 별도 정비 과제로 가정한다.

- Rule / Dictionary / Merchant Normalization 자체 정비
- MCC 체계와 내부 소비·마케팅 taxonomy 정비
- 사용자·가맹점 관계 그래프 구축
- 실시간 LLM serving

단, 이 결과들은 신규 모델의 입력 feature와 운영 신호로 사용한다.

---

## 1. 논문 및 산업 사례 비교

### 1.1 연구 논문

| 논문 | 문제 정의 | 데이터 | 방법 | 장점 | 한계 | 적용 포인트 | 링크 |
|---|---|---|---|---|---|---|---|
| **Merchant Category Identification Using Credit Card Transactions** (2020) | 가맹점이 신고한 business type 또는 merchant category가 실제 거래 행동과 일치하는지 식별 | 실제 대규모 신용카드 거래 데이터. 71,668개 가맹점과 고객·가맹점 거래 관계 및 시계열 사용 | Temporal transaction encoder와 merchant–merchant affinity encoder를 결합한 multi-modal learning | 가맹점명만으로 판단하기 어려운 업종을 거래 시간 패턴과 유사 가맹점 관계로 보완 | 충분한 거래 이력이 필요하고 신규 가맹점 cold-start에는 약함 | 1개월 이후 이름만으로 해결되지 않는 오분류에 거래 패턴·merchant affinity feature를 추가할 근거 | [arXiv](https://arxiv.org/abs/2011.02602) |
| **Building Payment Classification Models from Rules and Crowdsourced Labels: A Case Study** (2018) | 룰 기반 payment classification의 coverage 한계와 사용자 수정 라벨을 이용한 bootstrapping 문제 해결 | 익명화된 금융기관 wire transfer·card payment 데이터, 초기 룰, 사용자 수정 라벨, 66개 카테고리 | 초기 룰로 분류기를 bootstrap하고 사용자 수정 라벨과 함께 ML 모델을 학습 | 기존 룰과 사용자 수정 결과를 버리지 않고 학습 자산으로 전환 | 사용자 라벨의 일관성·편향·노이즈 관리 필요 | 기존 룰 결과와 수기 수정 데이터를 신규 학습 데이터 및 confidence feature로 전환하고, LLM·검수 결과를 재학습에 누적 | [Springer](https://link.springer.com/chapter/10.1007/978-3-319-92898-2_7), [PDF](https://lepo.it.da.ut.ee/~dumas/pubs/caise2018PaymentClassification.pdf) |
| **Scalable and Weakly Supervised Bank Transaction Classification** (2023) | 수작업 라벨이 부족한 은행 거래 데이터를 확장 가능하게 분류 | 거래 설명, 시간·금액 feature, 휴리스틱과 도메인 지식으로 만든 weak label | FastText 기반 anchoring, labeling functions, label model, multimodal neural classifier | 룰·휴리스틱을 대규모 학습 데이터 생성 파이프라인으로 연결 가능 | labeling function 품질과 상관관계에 민감하고 weak-label 확률이 실제 정확도와 다를 수 있음 | 1개월에는 label source와 confidence를 저장하는 스키마를 먼저 만들고, 이후 gold+weak label 혼합 학습으로 확장 | [arXiv](https://arxiv.org/abs/2305.18430) |
| **Hierarchical Classification of Financial Transactions Through Context-Fusion of Transformer-based Embeddings and Taxonomy-aware Attention Layer** (2023) | 금융 거래를 macro·micro category로 동시에 분류하면서 계층 불일치를 줄이는 문제 | Card 및 current-account 데이터의 merchant name과 business activity 텍스트 | Transformer encoder, Context Fusion, macro/micro two-head classifier, Taxonomy-aware Attention Layer | 짧은 merchant text와 업종 설명을 결합하고 부모·자식 카테고리 불일치를 줄임 | taxonomy 품질이 중요하며 입력 문맥이 부족하면 효과가 제한될 수 있음 | 신규 1차 모델의 향후 hierarchical multi-head 확장 근거. 1개월에는 유효 category path 검증과 Top-K masking으로 단순 적용 | [arXiv](https://arxiv.org/abs/2312.07730) |
| **Identifying Banking Transaction Descriptions via Support Vector Machine Short-Text Classification Based on a Specialized Labelled Corpus** (2024) | 약어와 정보 부족이 많은 은행 거래 설명을 낮은 비용으로 분류 | 실제 고객 거래 설명으로 구성된 전문 라벨 corpus | Short-text similarity detector와 SVM을 결합한 2단계 분류 | 짧은 문자열에서는 char/word n-gram과 SVM이 강한 저비용 baseline이 될 수 있음 | 의미 일반화와 복잡한 카테고리 경계 이해에는 제한 | 재구축 첫 모델로 char n-gram + Logistic Regression/Linear SVM을 반드시 구현해 빠른 성능 기준과 fallback을 확보 | [arXiv](https://arxiv.org/abs/2404.08664) |
| **E-commerce Product Categorization with LLM-based Dual-Expert Classification Paradigm** (2024) | 수천 개의 세밀한 e-commerce taxonomy에서 정확한 상품 카테고리를 선택 | 대규모 상품 텍스트, 계층 taxonomy path, 카테고리 정의 | Fine-tuned domain expert가 Top-K 후보를 생성하고 범용 LLM expert가 후보를 재순위화 | LLM이 전체 taxonomy를 자유 탐색하지 않고 유사 후보 간 미세한 차이를 비교 | 상품 설명보다 가맹점명이 훨씬 짧고 2단계 추론 비용이 발생 | 신규 1차 모델·임베딩이 Top-K 후보를 만들고 LLM이 기타·저신뢰 건을 후보 안에서 재분류하는 핵심 근거 | [Amazon Science](https://www.amazon.science/publications/e-commerce-product-categorization-with-llm-based-dual-expert-classification-paradigm) |
| **Transaction Categorization with Relational Deep Learning in QuickBooks** (2025) | QuickBooks 거래 카테고리화를 관계형 데이터베이스 위의 link prediction 문제로 재정의 | 거래, 기업, 계정·카테고리 코드, 과거 분류 이력이 연결된 관계형 데이터 | Txn-BERT, Top-K nearest-neighbor early exit, heterogeneous GNN link prediction | 반복 거래는 최근접 사례로 빠르게 처리하고 관계 정보로 unseen category와 cold-start를 보완 | graph schema와 학습·서빙 파이프라인 복잡도가 높음 | 1개월에는 nearest-neighbor early exit와 유사 정답 검색만 적용하고 GNN은 잔여 오류 분석 후 판단 | [arXiv](https://arxiv.org/abs/2506.09234) |
| **Categorising SME Bank Transactions with Machine Learning and Synthetic Data Generation** (2025) | SME 거래 설명의 비표준성, 라벨 부족, 클래스 불균형 해결 | SME bank transaction 데이터와 synthetic transaction 데이터 | Synthetic data generation, fine-tuned classifier, confidence calibration | 라벨 부족·불균형을 보완하고 고신뢰 자동 처리 구간을 분리 | 합성 데이터가 실제 분포를 왜곡할 수 있고 calibration 성능에 민감 | 신규 모델 평가 후 소수 카테고리 데이터 부족이 명확할 때만 제한적으로 적용 | [arXiv](https://arxiv.org/abs/2508.05425) |
| **Better with Less: Small Proprietary Models Surpass Large Language Models in Financial Transaction Understanding** (2025) | 금융 거래 문자열 이해에서 범용 대형 모델과 도메인 특화 소형 모델의 성능·속도·비용 비교 | Raw transaction descriptions와 대규모 merchant 후보를 이용한 merchant understanding·matching 과제 | 소형 도메인 Transformer와 encoder/decoder 계열 대형 모델 비교 | 제한된 거래 도메인에서는 작은 전용 모델이 대형 범용 모델보다 비용·속도·성능 면에서 유리할 수 있음을 제시 | 가맹점 카테고리 분류를 직접 평가한 연구는 아님 | 전체 가맹점을 LLM으로 처리하지 않고 신규 경량 1차 모델을 개발하며 LLM은 재분류에만 사용하는 근거 | [arXiv](https://arxiv.org/abs/2509.25803) |
| **Enhancing Foundation Models in Transaction Understanding with LLM-based Sentence Embeddings** (2025) | Transaction foundation model에서 merchant·MCC·location 같은 categorical index가 의미 정보를 잃는 문제 | Payment-network transaction sequence와 merchant, MCC, location 등 categorical field | 외부 문맥으로 categorical entity 설명을 보강하고 LLM sentence embedding을 오프라인 생성해 초기 표현에 주입 | 온라인 LLM 호출 없이 merchant·MCC 의미 정보를 경량 모델에 전달 | 정적 임베딩 갱신 문제가 있고 가맹점 분류 직접 실험은 아님 | 1개월에는 Label Card·merchant retrieval에 적용하고, 이후 category/MCC embedding 초기화에 활용 | [ACL Anthology](https://aclanthology.org/2025.emnlp-industry.61/) |

### 1.2 산업 사례

| 사례 | 문제 정의 | 데이터 | 방법 | 장점 | 한계 | 적용 포인트 | 링크 |
|---|---|---|---|---|---|---|---|
| **Plaid Enrich** | 비정형 transaction description을 정제하고 merchant·category·location·counterparty 정보로 보강 | Raw description, amount, account type, location 등 카드·은행 거래 입력 | Proprietary transaction enrichment와 merchant identification·categorization API | 카테고리뿐 아니라 표준화 merchant와 부가정보를 함께 제공 | 외부 taxonomy와 내부 taxonomy 차이, API 비용·보안·벤더 종속성 | 신규 시스템 출력도 `normalized merchant + category + confidence + evidence + model version`을 함께 제공 | [Product](https://plaid.com/products/enrich/), [API](https://plaid.com/docs/api/products/enrich/) |
| **Yodlee Transaction Data Enrichment** | 이해하기 어려운 금융 거래 설명을 merchant name·category·geolocation으로 보강 | Bank/card transaction 데이터. Retail 및 business account transaction 지원 | Proprietary ML engine 기반 simple description, merchant, category, geolocation enrichment | 거래 설명 정제와 카테고리화를 하나의 enrichment 흐름으로 제공 | 모델과 taxonomy가 비공개이고 내부 taxonomy로 재학습하기 어려움 | 분류 결과뿐 아니라 정규화명과 category path를 함께 저장하고 소비·사업자 taxonomy를 분리 관리 | [Overview](https://developer.yodlee.com/resources/yodlee/transaction-data-enrichment/docs), [Retail](https://developer.yodlee.com/resources/yodlee/transaction-data-enrichment/docs/retail-category), [Business](https://developer.yodlee.com/resources/yodlee/transaction-data-enrichment/docs/business-category) |
| **Salt Edge Data Enrichment** | Raw personal·business transaction을 카테고리와 merchant 정보로 변환 | Open-banking 등 다양한 transaction source의 raw description과 거래 정보 | Self-learning categorization, merchant identification, user-defined category 학습 API | 개인·사업자 카테고리와 사용자 정의 category를 지원 | 외부 category mapping과 결과 검증 체계를 사용자가 설계해야 함 | 검수·사용자 수정 결과를 feedback table에 저장하고 다음 재학습에 반영하는 구조의 산업적 근거 | [Product](https://www.saltedge.com/products/data_enrichment), [Docs](https://docs.saltedge.com/data_enrichment/v5/) |
| **QuickBooks AI-powered Banking and Transaction Categorization** | 은행 거래를 기존 장부와 match하거나 적합한 회계 category를 추천하고 고신뢰 항목을 빠르게 처리 | Full bank description, vendor 정보, 과거 transaction history와 수정 이력 | 과거 분류 이력·거래 상세를 이용한 AI suggestion, high-confidence ready-to-post, 사용자 feedback | 고신뢰 자동 처리와 검토 대상을 분리하고 반복 거래·사용자 이력을 활용 | 개인화된 회계 category 문제로 일반 소비 업종 분류와 차이가 있음 | `자동 확정 + LLM 재분류 + 검수 + 수정 이력 재학습`으로 운영 구간을 분리할 근거 | [AI Banking](https://quickbooks.intuit.com/learn-support/en-us/help-article/matching-rules/learn-updates-new-ai-powered-banking-page/L0hR7A9Zf_US_en_US), [Category Suggestions](https://quickbooks.intuit.com/learn-support/en-global/help-article/bank-transactions/ai-suggestions-help-match-categorise-bank/L8FHOh4AD_ROW_en) |

---

## 2. 기존 모델 상태가 설계에 미치는 영향

### 2.1 기존 모델을 primary classifier로 유지하면 안 되는 이유

```text
1. 학습 시점이 2023년 2월로 오래되어 현재 상호·브랜드 분포와 차이가 큼
2. 오분류가 많이 발생하고 있으나 원인을 재현할 학습·평가 코드가 없음
3. 동일 데이터에서 재학습하거나 threshold를 조정할 수 없음
4. feature·label·taxonomy 버전을 추적하기 어려움
5. 담당자 의존적인 운영이 반복될 가능성이 큼
```

따라서 기존 모델은 다음 용도로만 사용한다.

- 동일 평가셋에서 신규 모델과 비교하는 legacy baseline
- 기존 예측 결과와 신규 모델의 disagreement 분석
- 과거 결과 호환성이 필요한 경우에만 shadow output 저장
- 학습 코드나 weight가 남아 있더라도 신규 파이프라인의 필수 구성요소로 사용하지 않음

### 2.2 신규 개발의 첫 번째 산출물은 모델이 아니라 파이프라인

모델 정확도가 높아도 다음 항목이 없으면 다시 같은 문제가 발생한다.

```text
- 데이터 추출 SQL과 snapshot 기준
- merchant 단위 중복 제거 기준
- label source와 label priority
- train / validation / test split 코드
- feature 생성 코드
- 학습 configuration
- 평가 및 오류 분석 코드
- model artifact와 metadata 저장
- batch inference 코드
- model / taxonomy / prompt version 관리
- monitoring과 retraining 절차
```

신규 모델 개발의 완료 기준은 단순히 weight 파일을 만드는 것이 아니라 **다른 담당자가 동일 버전의 데이터를 이용해 동일 결과를 재현할 수 있는가**다.

---

## 3. 권장 기술 구성

### 3.1 기술 영역별 적용 판단

| 기술 영역 | 1개월 적용 여부 | 판단 |
|---|---:|---|
| 데이터셋·학습 파이프라인 재구축 | 필수 | 신규 모델 개발보다 먼저 완료해야 함 |
| Char n-gram + Logistic Regression / Linear SVM | 필수 | 짧은 가맹점명에 강하고 빠르게 재현 가능한 기준 모델 |
| Text + MCC / Rule / Tabular Feature Fusion | 필수 또는 우선 | 이름만으로 모호한 사례를 보완하는 현실적 주 모델 후보 |
| 경량 Transformer fine-tuning | 병렬 challenger | 성능 향상 가능성이 높으나 1개월 내 안정화 여부를 평가해야 함 |
| Hierarchical constraint | 부분 적용 | category path masking과 출력 검증부터 적용 |
| Embedding retrieval | 필수 | LLM에 전달할 Top-K 후보와 유사 정답 사례 생성 |
| LLM reclassification | 필수 | 기타·저신뢰·충돌·신규 가맹점에 한정 |
| Confidence calibration / routing | 필수 | 자동 확정·재분류·검수 구간 분리 |
| Weak supervision | 후속 | 룰·LLM·검수 라벨이 축적된 후 적용 |
| Transaction time-series / graph | 후속 | 1개월 결과의 잔여 오류가 이름 정보 부족인지 확인 후 판단 |
| Synthetic data | 선택적 후속 | 소수 category 부족이 확인될 때 제한 적용 |

### 3.2 신규 1차 분류 모델 후보

#### 모델 A: Char n-gram TF-IDF + Logistic Regression 또는 Linear SVM

```text
입력:
- raw_merchant_name
- normalized_merchant_name
- raw_transaction_description(optional)

특징:
- character n-gram
- word n-gram
- 영문·숫자·한글 혼합 토큰
```

장점:

- 학습과 추론이 빠름
- 오타, 띄어쓰기, 지점명, 약어에 강함
- 학습 코드와 feature를 재현하기 쉬움
- 대량 batch 처리 비용이 낮음
- 1개월 안에 반드시 확보 가능한 안전한 baseline

한계:

- 의미적으로 새로운 브랜드에 대한 일반화가 약함
- 이름만으로 업종이 드러나지 않는 가맹점에 한계
- 복잡한 hierarchy와 거래 문맥을 직접 학습하기 어려움

#### 모델 B: Text Probability + Tabular Feature Fusion

권장 초기 구조는 다음과 같다.

```text
Text Model:
    char n-gram classifier 또는 경량 Transformer
        → category probability / text embedding

Tabular Features:
    MCC
    rule_category / rule_score
    online_offline_flag
    PG / marketplace flag
    거래 금액·시간대·반복성 feature(optional)

Fusion Model:
    LightGBM / CatBoost / MLP
        → final category probability
```

1개월 내 transaction aggregation feature 생성이 어렵다면 다음 최소 feature로 시작한다.

```text
- raw / normalized merchant name
- MCC
- rule category
- rule confidence
- WordNet 또는 lexical similarity score
- brand / chain flag
- online / offline / PG flag
```

#### 모델 C: 경량 Transformer Challenger

후보 예시는 multilingual DistilBERT 계열 또는 내부 인프라에서 안정적으로 fine-tuning 가능한 encoder model이다.

```text
Input text:
[RAW] 원본 가맹점명
[NORM] 정규화 가맹점명
[MCC] MCC 설명
[RULE] 룰 후보 category
```

권장 운영 방식:

- 모델 A/B와 동일한 split에서 비교
- 전체 정확도만 보지 않고 Macro-F1, long-tail F1, new merchant 성능 비교
- 학습·추론 비용과 배치 처리량을 함께 비교
- 4주차에 안정화되지 않으면 challenger로 남기고 모델 A/B를 MVP champion으로 사용

### 3.3 Hierarchical Classification의 1개월 적용 방식

1개월 안에 복잡한 taxonomy-aware multi-head 모델을 필수로 만들지는 않는다. 대신 다음을 적용한다.

```text
- leaf category별 parent path 저장
- 모델의 Top-K 결과를 valid taxonomy path로 변환
- 불가능한 parent-child 조합 제거
- LLM에는 category_id가 아니라 category path와 정의를 제공
- 평가 시 major / middle / minor 수준을 각각 계산
```

taxonomy가 안정적이고 label 수가 충분하면 이후 shared encoder + multi-head 구조로 확장한다.

### 3.4 임베딩 검색

임베딩은 신규 1차 모델을 대체하지 않고, **LLM 재분류 후보 생성기**로 사용한다.

#### Merchant Reference Index

```yaml
reference_id: string
raw_merchant_name: string
normalized_merchant_name: string
brand_id: string | null
category_id: string
category_path: string
label_source: human | verified_rule | reviewed_llm
label_version: string
review_status: string
```

#### Category Label Card Index

```yaml
category_id: C102
category_name: 단체급식
category_path: 식음료 > 급식 > 단체급식
definition: 기업·학교·병원 등의 구내식당을 운영하거나 다수 인원에게 식사를 제공하는 업종
include:
  - 위탁급식
  - 사내식당
  - 학교급식
exclude:
  - 일반 음식점
  - 식자재 도매
aliases:
  - 푸드서비스
  - 케이터링
confusable_categories:
  - 일반 음식점
  - 식자재 유통
```

#### Hybrid Candidate Score

```text
candidate_score(category) =
    dense merchant similarity
  + category Label Card similarity
  + char n-gram / lexical similarity
  + new primary model probability
  + MCC prior
  + soft rule score
```

초기에는 정규화된 가중합으로 시작하고, 검증 라벨이 충분해지면 Logistic Regression 또는 LightGBM으로 결합 점수를 학습한다.

### 3.5 LLM 재분류

LLM은 다음 대상에만 사용한다.

```text
필수 대상:
- 신규 1차 모델 결과가 기타인 가맹점

선택 대상:
- calibrated confidence가 낮은 가맹점
- Top-1과 Top-2 margin이 작은 가맹점
- Rule과 신규 모델이 충돌한 가맹점
- 신규·미등록 가맹점
- 특정 long-tail 또는 오류 비용이 큰 category
```

권장 흐름:

```text
신규 1차 모델 예측
    ↓
기타 / 저신뢰 / 충돌 대상 추출
    ↓
수기 정답 exact match
    ↓
Embedding + lexical + model probability 기반 Top-K 검색
    ↓
LLM candidate-only reclassification
    ↓
자동 반영 / 검수 / 기타 유지
```

LLM은 다음 중 하나만 반환하도록 제한한다.

```text
selected
none_of_candidates
insufficient_information
review_required
```

LLM이 제공한 confidence 숫자를 그대로 사용하지 않고, 검색·모델·룰 신호를 함께 이용해 자동 반영 여부를 판단한다.

### 3.6 Confidence Calibration과 Routing

신규 모델의 softmax probability 또는 margin은 validation set에서 보정해야 한다.

권장 신호:

| 신호 | 역할 |
|---|---|
| `primary_max_probability` | 신규 모델의 최대 확률 |
| `primary_entropy` | 예측 분포 불확실성 |
| `primary_top1_top2_margin` | 상위 후보 간 구분 정도 |
| `rule_model_agreement` | Rule과 신규 모델의 일치 여부 |
| `embedding_model_agreement` | 검색 후보와 신규 모델의 일치 여부 |
| `neighbor_purity` | 유사 정답 사례의 category 일관성 |
| `candidate_recall_risk` | 후보 생성 실패 가능성 |
| `new_merchant_flag` | 신규 상호 여부 |
| `category_risk` | category별 오류 비용 |

권장 라우팅:

| 구간 | 액션 |
|---|---|
| 고신뢰 + Rule/검색 일치 | 신규 1차 모델 결과 자동 반영 |
| 중간 신뢰 + 후보 명확 | LLM 재분류 후 조건부 반영 |
| 저신뢰·충돌 | LLM 재분류 또는 검수 |
| 정보 부족·후보 누락 | 기타 유지 또는 human review |

---

## 4. 1개월 내 구현 가능한 권장 시스템 아키텍처

### 4.1 설계 원칙

- 기존 2023년 모델을 production dependency로 두지 않는다.
- 신규 학습 파이프라인과 신규 1차 모델을 먼저 재구축한다.
- 1개월 안에는 batch/shadow 실행을 목표로 한다.
- LLM은 전체 분류기가 아니라 재분류기로 사용한다.
- 수기 정답과 Label Card를 후보 검색에 활용한다.
- 결과·feature·모델·taxonomy·prompt 버전을 모두 저장한다.

### 4.2 학습 파이프라인

```text
[Raw Merchant / Transaction Data]
        |
        v
[Dataset Builder]
    - 기간 snapshot
    - merchant entity dedup
    - label source priority
    - taxonomy version
    - train/valid/test split
        |
        v
[Feature Builder]
    - raw / normalized name
    - char / word n-gram
    - MCC / rule / brand feature
    - optional transaction aggregate
        |
        +--------------------------+
        |                          |
        v                          v
[Fast Baseline]             [Transformer Challenger]
- Logistic Regression       - lightweight encoder
- Linear SVM                - fine-tuning
        |                          |
        +-------------+------------+
                      v
             [Evaluation & Selection]
             - Macro/Weighted F1
             - long-tail / new merchant
             - Top-K recall
             - calibration
             - latency / cost
                      |
                      v
                [Model Registry]
             - model artifact
             - data/config version
             - metrics/model card
```

### 4.3 배치 분류 및 LLM 재분류 파이프라인

```text
[New Merchant / Backfill Data]
        |
        v
[Preparation]
    - normalization result
    - merchant entity dedup
    - exact gold / dictionary match
        |
        +---- high-precision match ----> [Final Category]
        |
        v
[New Primary Classifier]
    - category probabilities
    - Top-K candidates
    - calibrated confidence
        |
        +---- high confidence ----------> [Auto Approval]
        |
        v
[Reclassification Router]
    - other
    - low confidence
    - model-rule conflict
    - new merchant
        |
        v
[Hybrid Retrieval]
    - labeled merchant index
    - Category Label Card index
    - lexical similarity
    - primary model probability
    - MCC / rule prior
        |
        v
[Top-3~Top-5 Candidate Package]
        |
        v
[LLM Reclassification]
    - candidate-only selection
    - none / insufficient 허용
    - structured JSON
        |
        v
[Acceptance Policy]
    - hierarchy validation
    - evidence / agreement check
    - category risk
       /        |         \
      /         |          \
[Accept]   [Human Review]  [Other 유지]
      \         |          /
       \        |         /
        [Versioned Result & Feedback Store]
                     |
                     v
             [Next Training Dataset]
```

### 4.4 최소 운영 구성요소

| 구성요소 | 최소 기능 |
|---|---|
| Dataset Builder | 날짜·taxonomy·label source가 고정된 학습 데이터 생성 |
| Training CLI / Job | config 기반 재학습과 seed 고정 |
| Evaluation Report | 전체·category·subset별 성능 및 confusion matrix |
| Model Registry | artifact, metrics, data version, code commit 저장 |
| Batch Inference | 재실행 가능한 idempotent job |
| Retrieval Index | 수기 정답·Label Card 버전 관리 |
| LLM Batch Module | retry, JSON validation, cache, token/cost log |
| Result Store | model·taxonomy·prompt 버전과 근거 저장 |
| Monitoring | category 분포, 기타율, confidence, disagreement, drift |

### 4.5 대규모 처리 원칙

1천만 거래 레코드를 직접 분류하거나 LLM에 보내지 않는다.

```text
원천 거래 1천만 건
    ↓ merchant_id / normalized merchant 기준 집계
고유 merchant entity
    ↓ exact match / cache 제거
신규 또는 재분류 대상 entity
    ↓ 신규 primary model
기타·저신뢰·충돌 entity만 LLM
```

권장 캐시 키:

```text
normalized_merchant_name
+ MCC(optional)
+ candidate_set_hash
+ taxonomy_version
+ primary_model_version
+ prompt_version
+ llm_model_version
```

---

## 5. 신규 모델 학습 데이터 설계

### 5.1 Label Source Priority

```text
Priority 1. 최근 수기 검수 완료 정답
Priority 2. 명확한 브랜드·사업자 기준 gold mapping
Priority 3. 사용자/운영팀 수정 결과
Priority 4. 검수된 고신뢰 LLM 결과
Priority 5. 고정밀 rule weak label
Priority 6. 기존 2023년 모델 예측은 label로 사용하지 않음
```

기존 모델 예측을 새 모델의 정답으로 사용하면 과거 오류를 그대로 학습할 수 있으므로 feature나 비교 신호로만 제한한다.

### 5.2 Split 원칙

단순 random row split은 동일 가맹점·브랜드가 train과 test에 동시에 포함되는 leakage를 만들 수 있다.

권장 split:

```text
- merchant_id 또는 normalized merchant group 단위 분리
- 가능하면 시간 기반 out-of-time test 추가
- 동일 brand의 지점이 양쪽에 섞이는지 별도 확인
- 신규 merchant subset 별도 유지
- long-tail category와 기타 재분류 subset 별도 유지
```

### 5.3 필수 평가 subset

| Subset | 목적 |
|---|---|
| Random Merchant | 전체 평균 성능 |
| Out-of-time Merchant | 최신 분포 일반화 |
| New Merchant | 신규 상호 대응 |
| Long-tail Category | 희소 category 성능 |
| Existing Other with Gold | 기타 감소 가능성 |
| Similar Category Pair | 미세 경계 구분 |
| Generic Name | 이름 정보 부족 대응 |
| PG / Marketplace | 실제 seller 정보 부재 오류 |
| Rule–Model Conflict | 하이브리드 라우팅 평가 |
| High-volume Merchant | 사업 영향이 큰 오류 평가 |

---

## 6. 실험 설계

### 6.1 신규 1차 모델 실험

```text
P0. 기존 2023년 SubwordCNN/ML 결과 재평가 가능한 범위의 legacy baseline
P1. char n-gram TF-IDF + Logistic Regression
P2. char n-gram TF-IDF + Linear SVM
P3. P1/P2 probability + MCC/rule feature LightGBM
P4. lightweight Transformer text-only
P5. lightweight Transformer + MCC/rule context
P6. P3와 P5 stacking 또는 champion-challenger 비교
```

1개월 MVP의 champion은 가장 복잡한 모델이 아니라 다음 조건을 만족하는 모델로 선정한다.

```text
- 신규·out-of-time 성능이 안정적임
- category별 편차가 수용 가능함
- confidence calibration이 가능함
- 대량 batch 추론 비용이 적정함
- 재학습과 배포를 다른 담당자가 재현할 수 있음
```

### 6.2 LLM 재분류 실험

```text
L0. 가맹점명 + 전체 category 목록을 이용한 direct LLM
L1. 신규 primary Top-K + LLM
L2. Embedding Top-K + LLM
L3. Primary + embedding + lexical + MCC/rule Top-K + LLM
L4. L3 + Category Label Card
L5. L4 + 유사 수기 정답 사례
L6. L5 + hard-negative / 혼동 사례
L7. L6 + acceptance routing
```

### 6.3 임베딩 검색 실험

```text
R0. char TF-IDF / lexical only
R1. multilingual dense embedding
R2. raw name + normalized name embedding
R3. merchant reference + Label Card dual index
R4. dense + lexical hybrid
R5. R4 + primary model probability + MCC/rule prior
```

평가:

```text
Recall@3 / Recall@5
MRR
Candidate Miss Rate
Neighbor Purity
New Merchant Recall@5
Long-tail Recall@5
검색 latency / index size
```

### 6.4 End-to-end 평가 지표

| 영역 | 지표 |
|---|---|
| 1차 모델 | Accuracy, Macro-F1, Weighted-F1, category F1 |
| 일반화 | Out-of-time F1, New Merchant F1, long-tail F1 |
| 후보 생성 | Recall@K, MRR, candidate miss rate |
| Confidence | ECE, Brier score, risk–coverage, top1-top2 margin |
| LLM | Conditional accuracy, repeat consistency, none rate |
| 운영 | Auto-approval precision, coverage, review rate, other reduction |
| 비용 | 고유 가맹점당 추론 비용, LLM 호출률, token 비용 |
| 성능 | Batch throughput, P50/P95 latency, cache hit rate |

오류는 반드시 다음처럼 분해한다.

```text
1. Label / taxonomy 오류
2. Primary classifier 오류
3. Candidate retrieval miss
4. LLM candidate selection 오류
5. Acceptance / routing 오류
6. 가맹점명만으로 판단 불가능한 사례
```

---

## 7. 1개월 단계별 구현 로드맵

### 7.1 전제와 현실적인 완료 범위

1개월은 타이트하지만 다음 수준의 MVP는 가능하다.

```text
가능:
- 재현 가능한 데이터셋·학습·평가 파이프라인
- 최소 2개 이상의 신규 1차 모델 baseline
- 신규 모델 champion 또는 shadow candidate 선정
- 기타·저신뢰 대상 LLM 재분류 모듈
- 과거 데이터 일부 backfill / shadow run

1개월 내 보장하기 어려움:
- 모든 category의 완전한 라벨 정비
- GNN·거래 시계열·대규모 weak supervision
- 실시간 production serving과 완전 자동 전환
- 신규 Transformer의 장기 안정성 검증
```

### 7.2 1주차 — 데이터·평가·재현 환경 복구

#### 목표

기존 담당자나 과거 코드에 의존하지 않는 데이터 및 학습 기반을 만든다.

#### 구현

```text
1. 최근 기간의 merchant·transaction·label source를 추출
2. 수기 정답, rule 결과, 기존 모델 결과의 source와 날짜를 구분
3. merchant entity와 normalized merchant 기준을 확정
4. taxonomy version과 label priority를 정의
5. group/time-aware train/validation/test split 구현
6. 공통 평가 subset과 지표 구현
7. 학습 configuration, random seed, artifact 저장 구조 생성
8. 기존 2023년 모델 결과를 동일 test set에서 가능한 범위로 평가
```

#### 산출물

```text
- Dataset Builder SQL / code
- versioned train/valid/test dataset
- label quality report
- legacy model baseline report
- training/evaluation repository skeleton
- environment / dependency file
- model artifact naming convention
```

#### 완료 기준

```text
동일 snapshot과 config로 데이터와 평가 결과를 재생성할 수 있음
기존 모델의 실제 현재 성능과 주요 오분류 유형을 파악함
```

### 7.3 2주차 — 신규 1차 분류 모델 개발

#### 목표

낮은 위험으로 재학습 가능한 baseline과 성능 challenger를 확보한다.

#### 구현

```text
1. char n-gram Logistic Regression 학습
2. char n-gram Linear SVM 또는 fastText 계열 비교
3. MCC/rule/brand feature를 결합한 LightGBM fusion 실험
4. 가능하면 경량 Transformer fine-tuning을 병렬 수행
5. category imbalance 처리와 class weight 실험
6. probability calibration 적용
7. 모델별 오류·비용·처리량 비교
```

#### 산출물

```text
- 최소 2개 신규 모델 artifact
- 재현 가능한 training command
- category/subset별 평가 리포트
- calibrated confidence output
- primary champion 및 challenger 후보
```

#### 완료 기준

```text
기존 2023년 모델보다 최근 test set에서 명확히 개선됨
또는 최소한 재현성과 유지보수성을 확보하면서 유사한 성능을 달성함
Top-K category probability를 안정적으로 생성할 수 있음
```

### 7.4 3주차 — 임베딩 후보 검색과 LLM 재분류 통합

#### 목표

신규 1차 모델이 놓치는 기타·저신뢰 사례를 LLM이 안전하게 재분류하도록 한다.

#### 구현

```text
1. 수기 정답 Merchant Reference Index 구축
2. Category Label Card 작성 및 embedding 생성
3. Dense + lexical + primary probability hybrid Top-K 구현
4. Candidate-only LLM prompt 및 JSON schema 구현
5. none_of_candidates / insufficient_information 처리
6. direct LLM 대비 성능·비용 비교
7. 자동 반영 / 검수 / 기타 유지 정책 구현
8. prompt/model/taxonomy version과 cache 저장
```

#### 산출물

```text
- Hybrid Retrieval 모듈
- LLM batch reclassification module
- Candidate Recall@K 리포트
- LLM end-to-end 재분류 리포트
- acceptance routing policy
```

#### 완료 기준

```text
기타·저신뢰 gold subset에서 신규 1차 모델 단독 대비 성능이 개선됨
자동 반영 구간의 precision이 사전 합의한 기준을 충족함
후보 밖 category 생성과 형식 오류가 통제됨
```

### 7.5 4주차 — 통합 배치·Shadow 검증 및 인수 가능 상태 정리

#### 목표

다른 담당자가 운영·재학습할 수 있는 통합 MVP를 만든다.

#### 구현

```text
1. 전처리 → 신규 primary → routing → retrieval → LLM → 결과 저장 연결
2. 과거 데이터 일부 backfill 및 최근 데이터 shadow run
3. batch idempotency, retry, cache, failure recovery 확인
4. category 분포·기타율·confidence drift 모니터링 구현
5. 운영 threshold와 rollback 기준 확정
6. 코드·데이터·모델·prompt 인수 문서 작성
7. 다음 재학습 주기와 owner 역할 정의
```

#### 산출물

```text
- End-to-end batch MVP
- 신규 primary model과 model card
- LLM 재분류 결과 table
- shadow/backfill 성능·비용 리포트
- retraining runbook
- 운영·장애·rollback 가이드
- 미해결 오류와 2개월차 backlog
```

#### 완료 기준

```text
기존 모델 파이프라인 없이 신규 데이터부터 결과까지 재현 가능함
신규 모델과 LLM 결과를 버전별로 추적할 수 있음
production 반영 전 shadow 판단에 필요한 정확도·비용·안정성 근거가 있음
```

---

## 8. 적용 우선순위

| 우선순위 | 항목 | 1개월 역할 | 판단 |
|---:|---|---|---|
| 1 | Dataset Builder와 label/version 정의 | 모든 모델 개발의 기반 | 필수 |
| 2 | Train/validation/test 및 평가 파이프라인 | 재현성과 비교 가능성 확보 | 필수 |
| 3 | Char n-gram 신규 baseline | 빠르고 안정적인 primary 후보 | 필수 |
| 4 | MCC/rule feature fusion | 이름 기반 한계 보완 | 필수 또는 우선 |
| 5 | 경량 Transformer challenger | 의미 일반화 성능 검증 | 병렬 권장 |
| 6 | Confidence calibration과 routing | 자동 처리와 재분류 구간 분리 | 필수 |
| 7 | Merchant·Label Card embedding index | LLM 후보 검색 | 필수 |
| 8 | Candidate-only LLM 재분류 | 기타·저신뢰 보정 | 필수 |
| 9 | Batch·registry·monitoring·runbook | 담당자 변경에도 운영 가능 | 필수 |
| 10 | Transaction pattern / weak supervision / distillation | MVP 결과 기반 후속 고도화 | 후속 |
| 11 | Relational GNN | 잔여 오류가 관계 정보 부족일 때 검토 | 장기 선택 |

---

## 9. 운영 정책과 모니터링

### 9.1 Model / Data Version

최소한 다음 값을 모든 결과에 저장한다.

```yaml
dataset_snapshot_date: date
label_version: string
taxonomy_version: string
feature_version: string
primary_model_name: string
primary_model_version: string
calibration_version: string
retrieval_index_version: string
prompt_version: string
llm_model_version: string
inference_run_id: string
```

### 9.2 결과 저장 스키마

```yaml
merchant_id: string
raw_merchant_name: string
normalized_merchant_name: string
mcc: string | null
rule_category_id: string | null
rule_score: float | null
legacy_category_id: string | null
primary_category_id: string
primary_topk: array
primary_confidence: float
primary_entropy: float
routing_reason: string
retrieval_candidates: array
neighbor_purity: float | null
candidate_margin: float | null
llm_selected_category_id: string | null
llm_decision: string | null
final_category_id: string
final_decision_source: string
review_status: string
model_version: string
taxonomy_version: string
prompt_version: string | null
created_at: timestamp
```

### 9.3 필수 모니터링

```text
- category 분포와 기타 비율
- 신규 merchant 비율
- 평균 confidence / entropy / margin
- Rule–primary disagreement 비율
- LLM routing 비율과 accepted 비율
- human review 정답률
- category별 precision 표본
- input text 길이·문자 유형·MCC 분포 drift
- 모델 version별 처리량·실패율·비용
```

### 9.4 재학습 주기

초기에는 고정 주기보다 조건 기반 재학습이 적합하다.

```text
- 최근 검수 gold label이 일정량 이상 축적됨
- 기타율 또는 신규 merchant 비율이 기준 이상 증가함
- 주요 category precision이 하락함
- 신규 브랜드·결제 표현이 대규모로 유입됨
- taxonomy가 변경됨
```

안정화 이후 월 1회 또는 분기 1회 정기 재학습과 drift-triggered 재학습을 결합할 수 있다.

---

## 10. 리스크와 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| 수기 정답 라벨이 오래되었거나 충돌 | 신규 모델이 잘못된 정답을 학습 | label source/date 저장, 충돌 제거, 표본 검수 |
| 기존 모델 예측을 정답으로 재사용 | 과거 오류가 신규 모델에 전파 | legacy output은 label로 사용하지 않음 |
| 동일 브랜드 leakage | test 성능 과대평가 | merchant/brand group split과 out-of-time test |
| 1개월 내 Transformer 안정화 실패 | 전체 일정 지연 | char n-gram + fusion을 guaranteed baseline으로 유지 |
| LLM이 후보 밖 지식을 생성 | 잘못된 자동 반영 | candidate-only schema, none/insufficient 허용 |
| 임베딩 후보에 정답이 없음 | LLM 선택 한계 | Recall@K를 별도 최적화하고 candidate miss 추적 |
| 기타 감소율만 강조 | 오분류 증가 | auto-approved precision과 risk–coverage를 핵심 기준으로 사용 |
| 담당자 변경 후 다시 파이프라인 유실 | 유지보수 불가 | repository, config, registry, runbook, owner 정의를 완료 기준에 포함 |
| 1천만 건 LLM 비용 | 비용·처리시간 증가 | merchant entity dedup, exact match, primary routing, cache 적용 |

---

## 11. 1개월 이후 권장 확장

1개월 MVP 결과를 기반으로 다음 순서로 확장한다.

```text
1. 신규 primary 모델 production shadow 기간 연장 및 threshold 안정화
2. 기타 외 low-confidence·conflict 대상의 LLM routing 확대
3. 거래 금액·시간대·반복성 feature 추가
4. 검수·LLM 결과를 이용한 weak supervision 또는 student distillation
5. Hierarchical multi-head classifier 개발
6. category별 threshold와 correctness model 학습
7. PG·marketplace·generic 상호 잔여 오류가 크면 relational graph 검토
```

### 11.1 2개월차에 우선할 항목

```text
- 신규 모델과 legacy 결과의 장기간 shadow 비교
- 검수 feedback UI 또는 batch review workflow
- transaction aggregate feature fusion
- 자동 재학습 job과 model promotion gate
- LLM 비용 최적화와 repeated pattern의 rule/student 흡수
```

---

## 12. 최종 제안

### 12.1 권장 의사결정

1. **기존 2023년 SubwordCNN/ML 모델은 유지 대상이 아니라 교체 대상이다.** 학습 파이프라인이 없고 현재 오분류가 많으므로 legacy baseline으로만 사용한다.
2. **1개월의 최우선 과제는 신규 모델 자체보다 재현 가능한 학습·평가·배포 파이프라인을 복구하는 것이다.**
3. **신규 1차 모델은 char n-gram 기반 저비용 baseline과 경량 Transformer challenger를 병렬 개발한다.** 일정 리스크를 줄이면서 성능 개선 가능성을 확인할 수 있다.
4. **MCC·Rule·브랜드 신호를 fusion feature로 활용한다.** 기존 룰과 모델을 경쟁 관계가 아니라 보완 신호로 사용한다.
5. **LLM을 재분류 용도로 사용하는 제안은 타당하다.** 신규 1차 모델의 기타·저신뢰·충돌 사례에 한정하는 것이 비용과 안정성 면에서 최적이다.
6. **임베딩은 LLM 앞단의 후보 검색기로 사용한다.** 수기 정답 사례와 Category Label Card를 검색해 후보를 좁힌다.
7. **1개월 완료 목표는 production 전면 전환이 아니라 end-to-end batch MVP와 shadow 판단 근거다.** 완전 자동화는 정확도·비용·검수 결과를 확인한 뒤 진행한다.
8. **모든 검수·LLM 재분류 결과는 다음 재학습 데이터로 축적한다.** 시간이 지날수록 LLM 호출을 경량 모델로 흡수하는 구조가 바람직하다.

### 12.2 한 줄 요약

```text
기존 2023년 모델을 유지한 채 LLM만 붙이는 것이 아니라,
1개월 동안 재현 가능한 신규 1차 모델 파이프라인을 다시 만들고
기타·저신뢰 결과만 임베딩 후보 검색과 LLM으로 재분류하는 구조가 최적이다.
```

---

## 13. 참고 링크

### Research Papers

- [Merchant Category Identification Using Credit Card Transactions](https://arxiv.org/abs/2011.02602)
- [Building Payment Classification Models from Rules and Crowdsourced Labels: A Case Study](https://link.springer.com/chapter/10.1007/978-3-319-92898-2_7)
- [Scalable and Weakly Supervised Bank Transaction Classification](https://arxiv.org/abs/2305.18430)
- [Hierarchical Classification of Financial Transactions Through Context-Fusion of Transformer-based Embeddings and Taxonomy-aware Attention Layer](https://arxiv.org/abs/2312.07730)
- [Identifying Banking Transaction Descriptions via Support Vector Machine Short-Text Classification Based on a Specialized Labelled Corpus](https://arxiv.org/abs/2404.08664)
- [E-commerce Product Categorization with LLM-based Dual-Expert Classification Paradigm](https://www.amazon.science/publications/e-commerce-product-categorization-with-llm-based-dual-expert-classification-paradigm)
- [Transaction Categorization with Relational Deep Learning in QuickBooks](https://arxiv.org/abs/2506.09234)
- [Categorising SME Bank Transactions with Machine Learning and Synthetic Data Generation](https://arxiv.org/abs/2508.05425)
- [Better with Less: Small Proprietary Models Surpass Large Language Models in Financial Transaction Understanding](https://arxiv.org/abs/2509.25803)
- [Enhancing Foundation Models in Transaction Understanding with LLM-based Sentence Embeddings](https://aclanthology.org/2025.emnlp-industry.61/)

### Industry / Product References

- [Plaid Enrich](https://plaid.com/products/enrich/)
- [Plaid Enrich API](https://plaid.com/docs/api/products/enrich/)
- [Yodlee Transaction Data Enrichment](https://developer.yodlee.com/resources/yodlee/transaction-data-enrichment/docs)
- [Yodlee Retail Transaction Enrichment](https://developer.yodlee.com/resources/yodlee/transaction-data-enrichment/docs/retail-category)
- [Yodlee Business Transaction Enrichment](https://developer.yodlee.com/resources/yodlee/transaction-data-enrichment/docs/business-category)
- [Salt Edge Data Enrichment](https://www.saltedge.com/products/data_enrichment)
- [Salt Edge Data Enrichment Docs](https://docs.saltedge.com/data_enrichment/v5/)
- [QuickBooks AI-powered Banking](https://quickbooks.intuit.com/learn-support/en-us/help-article/matching-rules/learn-updates-new-ai-powered-banking-page/L0hR7A9Zf_US_en_US)

---

## 부록 A. 최소 권장 Repository 구조

```text
merchant-category-classification/
├── configs/
│   ├── data.yaml
│   ├── train_char_ngram.yaml
│   ├── train_transformer.yaml
│   └── inference.yaml
├── sql/
│   ├── build_merchant_snapshot.sql
│   └── build_labels.sql
├── src/
│   ├── data/
│   │   ├── build_dataset.py
│   │   ├── split.py
│   │   └── validation.py
│   ├── features/
│   │   ├── text_features.py
│   │   └── tabular_features.py
│   ├── models/
│   │   ├── train_linear.py
│   │   ├── train_transformer.py
│   │   ├── calibrate.py
│   │   └── evaluate.py
│   ├── retrieval/
│   │   ├── build_index.py
│   │   └── search.py
│   ├── llm/
│   │   ├── prompt.py
│   │   ├── batch_reclassify.py
│   │   └── schema.py
│   └── pipeline/
│       ├── batch_inference.py
│       └── routing.py
├── tests/
├── notebooks/
├── model_cards/
├── requirements.txt 또는 pyproject.toml
└── README.md
```

## 부록 B. 최소 권장 LLM 재분류 Prompt

```text
System:
너는 제공된 내부 가맹점 카테고리 taxonomy를 따르는 재분류 전문가다.
입력은 신규 1차 분류 모델에서 기타·저신뢰·충돌로 판단된 가맹점이다.
반드시 제공된 후보 중 하나를 선택하거나
none_of_candidates / insufficient_information / review_required를 반환한다.
상호명과 제공 근거에 없는 사업 정보를 임의로 생성하지 않는다.

Input:
- 원본 가맹점명
- 정규화 가맹점명
- MCC 및 설명(optional)
- Rule 결과와 근거(optional)
- 신규 1차 모델 Top-K와 calibrated confidence
- 임베딩·lexical 결합 후보 Top-K
- 후보별 taxonomy path
- 후보별 definition / include / exclude
- 유사 수기 정답 사례
- 혼동 사례(optional)

Output JSON:
- selected_category_id
- alternative_category_id
- decision
- evidence
- ambiguity_type
- review_required
```

## 부록 C. 1개월 MVP 완료 체크리스트

```text
[ ] 최근 snapshot 기반 versioned dataset 생성 가능
[ ] merchant/group/time-aware split 재현 가능
[ ] 신규 char n-gram baseline 학습 가능
[ ] 경량 Transformer challenger 결과 확인
[ ] 신규 모델 calibration 및 Top-K 출력 가능
[ ] 수기 정답·Label Card retrieval index 생성 가능
[ ] Candidate-only LLM batch 실행 가능
[ ] 자동 반영 / 검수 / 기타 유지 routing 가능
[ ] batch 재실행과 cache/idempotency 확인
[ ] model/data/taxonomy/prompt version 저장
[ ] model card와 retraining runbook 작성
[ ] shadow/backfill 평가 리포트 완료
```
