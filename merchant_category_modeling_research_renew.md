# 가맹점 카테고리 분류 모델링 연구 및 권장 아키텍처

> 작성일: 2026-07-14  
> 목적: 2023년 2월 이후 갱신되지 않았고 재학습 파이프라인이 유실된 기존 SubwordCNN/ML 모델을 전제로, **재현 가능한 신규 1차 분류 모델을 구축하고 임베딩 검색과 LLM 재분류를 결합하는 방향**을 논문 및 산업 사례를 바탕으로 검토한다.

---

## 0. 개요

### 0.1 검토 배경

현재 가맹점 카테고리 분류 환경은 다음과 같다.

```text
- 기존 SubwordCNN / ML 분류 모델은 2023년 2월에 학습됨
- 이후 신규 가맹점, 브랜드, 결제 표현과 거래 분포 변화가 반영되지 않음
- 최근 데이터에서 오분류가 많이 발생하고 있음
- 담당자 변경 및 퇴사로 학습 코드, 데이터 생성 절차, 평가·배포 파이프라인이 남아 있지 않음
- 기존 모델의 재학습과 성능 재현이 어려움
- LLM을 기타 가맹점 재분류에 적용했을 때 개선 효과가 확인됨
- 수기 라벨링된 정답 가맹점 데이터가 존재함
```

따라서 기존 모델을 그대로 유지하면서 LLM만 추가하는 방식보다는 다음 두 축을 함께 추진하는 것이 적절하다.

```text
1. 재현 가능한 신규 1차 분류 모델과 학습·평가 파이프라인 구축
2. 신규 1차 모델의 기타·저신뢰·충돌 사례를 임베딩 검색과 LLM으로 재분류
```

### 0.2 문서의 초점

본 문서는 모델 리서치 관점에서 다음 내용을 정리한다.

```text
- 가맹점·금융 거래 카테고리 분류 관련 연구 논문과 산업 사례
- 적용 가능한 모델링 기술과 각각의 역할
- 기존 모델 파이프라인이 없는 상황에서의 신규 모델 후보
- 임베딩과 LLM을 결합한 재분류 구조
- 1개월 내 검증 가능한 권장 시스템 아키텍처와 실험 범위
```

Rule, Dictionary, Merchant Normalization, MCC 및 내부 taxonomy 정비는 별도 과제로 가정한다. 다만 이 결과는 신규 모델의 입력 feature, 후보 생성 신호, confidence routing 신호로 활용한다.

### 0.3 핵심 판단

| 구성요소 | 권장 역할 |
|---|---|
| 기존 2023년 SubwordCNN/ML | 신규 모델과 비교하기 위한 legacy baseline 또는 disagreement 분석용 신호 |
| 신규 1차 분류 모델 | 전체 가맹점의 기본 카테고리 및 Top-K 후보 생성 |
| Rule / Dictionary | 고정밀 early exit, 모델 feature, 모델과의 충돌 탐지 |
| 임베딩 검색 | 유사 수기 정답 가맹점과 Category Label Card를 검색해 후보 카테고리 생성 |
| LLM | 기타·저신뢰·후보 간 경계가 모호한 가맹점의 재분류 |
| Human Review | 정보 부족, 고위험 카테고리, 모델 간 충돌 사례의 최종 확인 |
| Feedback Data | 검수 결과와 고신뢰 재분류 결과를 다음 학습 데이터로 축적 |

결론적으로 LLM을 전체 가맹점의 기본 분류기로 사용하는 것보다, **재현 가능한 경량 1차 분류 모델을 새로 만들고 LLM은 재분류기로 한정하는 구조**가 성능·비용·유지보수 측면에서 가장 현실적이다.

---

## 1. 논문 / 산업 사례 비교표

### 1.1 연구 논문

| 논문 | 문제 정의 | 데이터 | 방법 | 장점 | 한계 | 적용 포인트 | 링크 |
|---|---|---|---|---|---|---|---|
| **Merchant Category Identification Using Credit Card Transactions** (2020) | 가맹점이 신고한 business type 또는 merchant category가 실제 거래 행동과 일치하는지 식별 | 실제 대규모 신용카드 거래 데이터. 71,668개 가맹점과 고객·가맹점 거래 관계 및 시계열 사용 | Temporal transaction encoder와 merchant–merchant affinity encoder를 결합한 multi-modal learning | 가맹점명만으로 판단하기 어려운 업종을 거래 시간 패턴과 유사 가맹점 관계로 보완 | 충분한 거래 이력이 필요하고 신규 가맹점 cold-start에는 약함 | 1차 텍스트 모델 이후에도 남는 모호한 가맹점에 거래 패턴과 merchant affinity feature를 추가할 근거 | [arXiv](https://arxiv.org/abs/2011.02602) |
| **Building Payment Classification Models from Rules and Crowdsourced Labels: A Case Study** (2018) | 룰 기반 payment classification의 coverage 한계와 사용자 수정 라벨을 이용한 bootstrapping 문제 해결 | 익명화된 금융기관 wire transfer·card payment 데이터, 초기 룰, 사용자 수정 라벨, 66개 카테고리 | 초기 룰로 분류기를 bootstrap하고 사용자 수정 라벨과 함께 ML 모델을 학습 | 기존 룰과 사용자 수정 결과를 버리지 않고 학습 자산으로 전환 | 사용자 라벨의 일관성·편향·노이즈 관리 필요 | 기존 룰 결과, 수기 수정, LLM 검수 결과를 학습 데이터와 confidence feature로 전환할 근거 | [Springer](https://link.springer.com/chapter/10.1007/978-3-319-92898-2_7), [PDF](https://lepo.it.da.ut.ee/~dumas/pubs/caise2018PaymentClassification.pdf) |
| **Scalable and Weakly Supervised Bank Transaction Classification** (2023) | 수작업 라벨이 부족한 은행 거래 데이터를 확장 가능하게 분류 | 거래 설명, 시간·금액 feature, 휴리스틱과 도메인 지식으로 만든 weak label | FastText 기반 anchoring, labeling functions, label model, multimodal neural classifier | 룰·휴리스틱을 대규모 학습 데이터 생성 파이프라인으로 연결 가능 | labeling function 품질과 상관관계에 민감하고 weak-label 확률이 실제 정확도와 다를 수 있음 | 초기에는 label source와 confidence를 저장하고, 이후 gold+weak label 혼합 학습으로 확장할 근거 | [arXiv](https://arxiv.org/abs/2305.18430) |
| **Hierarchical Classification of Financial Transactions Through Context-Fusion of Transformer-based Embeddings and Taxonomy-aware Attention Layer** (2023) | 금융 거래를 macro·micro category로 동시에 분류하면서 계층 불일치를 줄이는 문제 | Card 및 current-account 데이터의 merchant name과 business activity 텍스트 | Transformer encoder, Context Fusion, macro/micro two-head classifier, Taxonomy-aware Attention Layer | 짧은 merchant text와 업종 설명을 결합하고 부모·자식 카테고리 불일치를 줄임 | taxonomy 품질이 중요하며 입력 문맥이 부족하면 효과가 제한될 수 있음 | 신규 1차 모델의 계층형 multi-head 확장과 category path 검증에 활용 가능 | [arXiv](https://arxiv.org/abs/2312.07730) |
| **Identifying Banking Transaction Descriptions via Support Vector Machine Short-Text Classification Based on a Specialized Labelled Corpus** (2024) | 약어와 정보 부족이 많은 은행 거래 설명을 낮은 비용으로 분류 | 실제 고객 거래 설명으로 구성된 전문 라벨 corpus | Short-text similarity detector와 SVM을 결합한 2단계 분류 | 짧은 문자열에서는 char/word n-gram과 SVM이 강한 저비용 baseline이 될 수 있음 | 의미 일반화와 복잡한 카테고리 경계 이해에는 제한 | 재구축 첫 모델로 char n-gram + Logistic Regression/Linear SVM을 반드시 비교해야 하는 근거 | [arXiv](https://arxiv.org/abs/2404.08664) |
| **E-commerce Product Categorization with LLM-based Dual-Expert Classification Paradigm** (2024) | 수천 개의 세밀한 e-commerce taxonomy에서 정확한 상품 카테고리를 선택 | 대규모 상품 텍스트, 계층 taxonomy path, 카테고리 정의 | Fine-tuned domain expert가 Top-K 후보를 생성하고 범용 LLM expert가 후보를 재순위화 | LLM이 전체 taxonomy를 자유 탐색하지 않고 유사 후보 간 미세한 차이를 비교 | 상품 설명보다 가맹점명이 훨씬 짧고 2단계 추론 비용이 발생 | 신규 1차 모델·임베딩이 Top-K 후보를 만들고 LLM이 기타·저신뢰 건을 후보 안에서 재분류하는 핵심 근거 | [Amazon Science](https://www.amazon.science/publications/e-commerce-product-categorization-with-llm-based-dual-expert-classification-paradigm) |
| **Transaction Categorization with Relational Deep Learning in QuickBooks** (2025) | QuickBooks 거래 카테고리화를 관계형 데이터베이스 위의 link prediction 문제로 재정의 | 거래, 기업, 계정·카테고리 코드, 과거 분류 이력이 연결된 관계형 데이터 | Txn-BERT, Top-K nearest-neighbor early exit, heterogeneous GNN link prediction | 반복 거래는 최근접 사례로 빠르게 처리하고 관계 정보로 unseen category와 cold-start를 보완 | graph schema와 학습·서빙 파이프라인 복잡도가 높음 | 1개월에는 nearest-neighbor early exit와 유사 정답 검색을 적용하고 GNN은 후속 검토 | [arXiv](https://arxiv.org/abs/2506.09234) |
| **Categorising SME Bank Transactions with Machine Learning and Synthetic Data Generation** (2025) | SME 거래 설명의 비표준성, 라벨 부족, 클래스 불균형 해결 | SME bank transaction 데이터와 synthetic transaction 데이터 | Synthetic data generation, fine-tuned classifier, confidence calibration | 라벨 부족·불균형을 보완하고 고신뢰 자동 처리 구간을 분리 | 합성 데이터가 실제 분포를 왜곡할 수 있고 calibration 성능에 민감 | 신규 모델 평가 후 소수 카테고리의 라벨 부족이 명확할 때 제한적으로 적용 | [arXiv](https://arxiv.org/abs/2508.05425) |
| **Better with Less: Small Proprietary Models Surpass Large Language Models in Financial Transaction Understanding** (2025) | 금융 거래 문자열 이해에서 범용 대형 모델과 도메인 특화 소형 모델의 성능·속도·비용 비교 | Raw transaction descriptions와 대규모 merchant 후보를 이용한 merchant understanding·matching 과제 | 소형 도메인 Transformer와 encoder/decoder 계열 대형 모델 비교 | 제한된 거래 도메인에서는 작은 전용 모델이 대형 범용 모델보다 비용·속도·성능 면에서 유리할 수 있음을 제시 | 가맹점 카테고리 분류를 직접 평가한 연구는 아님 | 전체 건을 LLM으로 처리하지 않고 신규 경량 1차 모델을 구축하며 LLM은 재분류에만 사용하는 근거 | [arXiv](https://arxiv.org/abs/2509.25803) |
| **Enhancing Foundation Models in Transaction Understanding with LLM-based Sentence Embeddings** (2025) | Transaction foundation model에서 merchant·MCC·location 같은 categorical index가 의미 정보를 잃는 문제 | Payment-network transaction sequence와 merchant, MCC, location 등 categorical field | 외부 문맥으로 categorical entity 설명을 보강하고 LLM sentence embedding을 오프라인 생성해 초기 표현에 주입 | 온라인 LLM 호출 없이 merchant·MCC 의미 정보를 경량 모델에 전달 | 정적 임베딩 갱신 문제가 있고 가맹점 분류 직접 실험은 아님 | 초기에는 Label Card와 merchant retrieval에 사용하고, 이후 category/MCC embedding 초기화에 활용 가능 | [ACL Anthology](https://aclanthology.org/2025.emnlp-industry.61/) |

### 1.2 산업 사례

| 사례 | 문제 정의 | 데이터 | 방법 | 장점 | 한계 | 적용 포인트 | 링크 |
|---|---|---|---|---|---|---|---|
| **Plaid Enrich** | 비정형 transaction description을 정제하고 merchant·category·location·counterparty 정보로 보강 | Raw description, amount, account type, location 등 카드·은행 거래 입력 | Proprietary transaction enrichment와 merchant identification·categorization API | 카테고리뿐 아니라 표준화 merchant와 부가정보를 함께 제공 | 외부 taxonomy와 내부 taxonomy 차이, API 비용·보안·벤더 종속성 | 신규 시스템 출력도 `normalized merchant + category + confidence + evidence + model version`을 함께 제공하는 구조가 적합 | [Product](https://plaid.com/products/enrich/), [API](https://plaid.com/docs/api/products/enrich/) |
| **Yodlee Transaction Data Enrichment** | 이해하기 어려운 금융 거래 설명을 merchant name·category·geolocation으로 보강 | Bank/card transaction 데이터. Retail 및 business account transaction 지원 | Proprietary ML engine 기반 simple description, merchant, category, geolocation enrichment | 거래 설명 정제와 카테고리화를 하나의 enrichment 흐름으로 제공 | 모델과 taxonomy가 비공개이고 내부 taxonomy로 재학습하기 어려움 | 분류 결과뿐 아니라 정규화명, category path, 부가정보를 함께 저장하는 출력 구조에 참고 | [Overview](https://developer.yodlee.com/resources/yodlee/transaction-data-enrichment/docs), [Retail](https://developer.yodlee.com/resources/yodlee/transaction-data-enrichment/docs/retail-category), [Business](https://developer.yodlee.com/resources/yodlee/transaction-data-enrichment/docs/business-category) |
| **Salt Edge Data Enrichment** | Raw personal·business transaction을 카테고리와 merchant 정보로 변환 | Open-banking 등 다양한 transaction source의 raw description과 거래 정보 | Self-learning categorization, merchant identification, user-defined category 학습 API | 개인·사업자 카테고리와 사용자 정의 category를 지원 | 외부 category mapping과 결과 검증 체계를 사용자가 설계해야 함 | 검수·사용자 수정 결과를 feedback data로 저장하고 다음 학습에 반영하는 구조에 참고 | [Product](https://www.saltedge.com/products/data_enrichment), [Docs](https://docs.saltedge.com/data_enrichment/v5/) |
| **QuickBooks AI-powered Banking and Transaction Categorization** | 은행 거래를 기존 장부와 match하거나 적합한 회계 category를 추천하고 고신뢰 항목을 빠르게 처리 | Full bank description, vendor 정보, 과거 transaction history와 수정 이력 | 과거 분류 이력·거래 상세를 이용한 AI suggestion, high-confidence ready-to-post, 사용자 feedback | 고신뢰 자동 처리와 검토 대상을 분리하고 반복 거래·사용자 이력을 활용 | 개인화된 회계 category 문제로 일반 소비 업종 분류와 차이가 있음 | `자동 확정 + LLM 재분류 + 검수 + 수정 이력 재학습`으로 운영 구간을 분리하는 구조에 참고 | [AI Banking](https://quickbooks.intuit.com/learn-support/en-us/help-article/matching-rules/learn-updates-new-ai-powered-banking-page/L0hR7A9Zf_US_en_US), [Category Suggestions](https://quickbooks.intuit.com/learn-support/en-global/help-article/bank-transactions/ai-suggestions-help-match-categorise-bank/L8FHOh4AD_ROW_en) |

---

## 2. 가맹점 카테고리 분류 모델링 기술 카테고리 정리

가맹점 카테고리 분류는 하나의 모델로 해결하기보다, **가맹점명 기반 1차 분류·부가 feature 결합·후보 검색·LLM 재분류·confidence routing**을 역할별로 결합하는 것이 적합하다.

```text
1. Short-text / Subword 기반 신규 1차 분류
2. Text + Tabular Feature Fusion
3. Hierarchical Classification
4. Weak Supervision / Feedback Learning
5. Transaction Pattern / Time-series Modeling
6. Embedding Retrieval / Similar Example Search
7. LLM Candidate Reclassification
8. Confidence Calibration / Routing
9. Merchant Affinity / Relational Graph Modeling
```

### 2.1 Short-text / Subword 기반 신규 1차 분류

가맹점명과 거래 설명은 짧고 표기 변형이 많다.

```text
STARBUCKS 1234 SEOUL
스타벅스강남2호
SBUX GANGNAM
네이버페이_스마트스토어
쿠팡페이먼츠
OO상사
ABC1234
```

이러한 데이터에서는 word-level 의미 모델뿐 아니라 character n-gram, subword, 음절 단위 표현이 중요하다.

#### 권장 비교 모델

```text
Baseline A: char n-gram TF-IDF + Logistic Regression
Baseline B: char n-gram TF-IDF + Linear SVM
Baseline C: fastText 또는 Subword 기반 경량 모델
Challenger D: multilingual 경량 Transformer
Challenger E: Transformer + MCC/Rule context
```

#### 신규 모델 개발 방향

기존 SubwordCNN은 학습 파이프라인이 없으므로 재사용 가능한 primary classifier로 보기 어렵다. 신규 모델은 다음 조건을 우선 만족해야 한다.

```text
- 데이터 생성부터 평가까지 재현 가능함
- 최신 데이터로 재학습 가능함
- category probability와 Top-K 후보를 출력함
- batch 처리량과 비용이 관리 가능함
- confidence calibration이 가능함
```

#### 장점과 한계

| 구분 | 내용 |
|---|---|
| 장점 | 짧은 상호명, 오타, 띄어쓰기, 지점명, 약어에 강하고 대량 처리 비용이 낮음 |
| 한계 | 이름만으로 업종을 알 수 없는 generic merchant, PG/marketplace, 신규 브랜드에는 한계 |
| 적용 판단 | 1개월 내 반드시 새로 구축해야 하는 핵심 1차 모델 영역 |

### 2.2 Text + Tabular Feature Fusion

가맹점명만으로 분류하기 어려운 경우 MCC, Rule 결과, 거래 채널, 브랜드, 금액·시간대 패턴을 결합할 수 있다.

```text
Text Encoder:
    raw_merchant_name + normalized_merchant_name
        → text probability / text embedding

Categorical Features:
    MCC, rule_category, brand_id, PG flag, online/offline
        → categorical feature

Numerical Features:
    rule_score, WordNet score, amount/time/repeat pattern
        → numerical feature

Fusion:
    LightGBM / CatBoost / MLP
        → final category probability
```

#### 1개월 최소 입력 feature

```text
- raw_merchant_name
- normalized_merchant_name
- MCC 및 MCC 설명
- rule_category / rule_score
- lexical 또는 WordNet similarity score
- brand / chain flag
- online / offline / PG flag
```

거래 패턴 집계가 이미 준비되어 있지 않다면 1개월 MVP에서는 제외하고, 이후 성능 한계가 확인된 카테고리에 추가하는 것이 현실적이다.

### 2.3 Hierarchical Classification

카테고리가 대분류·중분류·소분류 구조라면 flat classification만으로는 계층 불일치가 발생할 수 있다.

```text
대분류: 쇼핑
소분류: 카페
```

#### 단기 적용

```text
- leaf category별 parent path 저장
- Top-K 결과를 valid taxonomy path로 변환
- 불가능한 parent-child 조합 제거
- LLM에 category name이 아니라 category path와 정의를 제공
- 평가를 major / middle / minor 수준으로 분리
```

#### 후속 확장

```text
Shared Encoder
    ├─ Major Category Head
    ├─ Middle Category Head
    └─ Minor Category Head
         + hierarchy consistency loss
```

1개월 안에는 taxonomy-aware multi-head를 필수 구현하기보다, path masking과 출력 검증을 우선 적용하는 것이 적절하다.

### 2.4 Weak Supervision / Feedback Learning

수기 정답이 충분하지 않은 경우 기존 Rule, MCC, 외부 업종 정보, 사용자 수정, LLM 검수 결과를 label source로 활용할 수 있다.

#### Label Source 예시

| Label Source | 권장 활용 |
|---|---|
| 최근 수기 검수 정답 | Gold label |
| 명확한 브랜드·사업자 mapping | Gold 또는 고신뢰 label |
| 운영팀·사용자 수정 | Feedback label |
| 검수된 LLM 결과 | Gold에 준하는 reviewed label |
| 고정밀 Rule 결과 | Confidence가 포함된 weak label |
| 기존 2023년 모델 예측 | 정답으로 사용하지 않고 비교·충돌 신호로만 사용 |

초기에는 weak label을 대규모 생성하기보다 `label_source`, `label_date`, `confidence`, `review_status`를 데이터 스키마에 저장해 후속 학습에 사용할 기반을 만드는 것이 중요하다.

### 2.5 Transaction Pattern / Time-series Modeling

가맹점 업종은 거래 패턴에도 나타난다.

| 업종 | 패턴 예시 |
|---|---|
| 카페 | 소액, 오전·오후 피크, 반복 방문 |
| 주점 | 야간·주말 비중 증가 |
| 병원 | 평일 주간, 금액 분산 큼 |
| 구독 서비스 | 월 단위 반복, 고정 금액 |
| 편의점 | 소액, 전 시간대 분포 |

#### 후보 feature

```text
amount: avg, median, std, percentile
hour: daypart ratio, night ratio
weekday: weekday/weekend/holiday ratio
recurrence: repeat ratio, fixed amount ratio, monthly cycle
customer: repeat user ratio, unique user count
channel: online/offline, PG, app payment
```

이 영역은 이름만으로 분류할 수 없는 잔여 오류에 효과적이지만, aggregation pipeline이 필요하므로 1개월 이후 우선 확장 후보로 두는 것이 적절하다.

### 2.6 Embedding Retrieval / Similar Example Search

임베딩은 최종 카테고리를 직접 확정하기보다, **LLM에 전달할 후보 카테고리와 유사 수기 정답 사례를 검색하는 역할**로 사용하는 것이 안정적이다.

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

#### 후보 점수 구성

```text
candidate_score(category) =
    merchant dense similarity
  + Category Label Card similarity
  + char n-gram / lexical similarity
  + primary model probability
  + MCC prior
  + soft rule score
```

초기에는 정규화된 가중합으로 시작하고, 검증 데이터가 충분해지면 Logistic Regression 또는 LightGBM으로 후보 점수를 학습한다.

#### 핵심 평가 지표

```text
Recall@3 / Recall@5
MRR
Candidate Miss Rate
Neighbor Purity
New Merchant Recall@5
Long-tail Recall@5
```

LLM은 후보에 정답이 없으면 정답을 선택하기 어려우므로, 임베딩 단계에서는 Top-1 Accuracy보다 Recall@K가 더 중요하다.

### 2.7 LLM Candidate Reclassification

LLM은 전체 가맹점 분류기가 아니라 다음 사례의 재분류기로 사용하는 것이 적절하다.

```text
- 신규 1차 모델의 결과가 기타인 가맹점
- calibrated confidence가 낮은 가맹점
- Top-1과 Top-2 margin이 작은 가맹점
- Rule과 신규 모델이 충돌한 가맹점
- 신규·미등록 가맹점
- 유사 category 간 경계가 모호한 사례
```

#### 권장 방식

```text
신규 1차 모델 Top-K
        +
Embedding / lexical 검색 Top-K
        +
Category Label Card
        +
유사 수기 정답 사례
        ↓
Candidate-only LLM Reclassification
```

LLM은 제공된 후보 중 하나를 선택하거나 다음 상태를 반환하도록 제한한다.

```text
selected
none_of_candidates
insufficient_information
review_required
```

이 방식은 LLM이 전체 taxonomy를 자유 생성하는 오류를 줄이고, Amazon Dual-Expert 구조처럼 1차 도메인 모델과 LLM의 역할을 분리한다.

### 2.8 Confidence Calibration / Routing

신규 모델의 확률을 그대로 자동 반영 기준으로 사용하지 않고 validation set에서 calibration과 routing을 설계해야 한다.

#### 주요 신호

| 신호 | 의미 |
|---|---|
| `primary_max_probability` | 신규 모델의 최대 예측 확률 |
| `primary_entropy` | 예측 분포 불확실성 |
| `primary_top1_top2_margin` | 상위 후보 간 구분 정도 |
| `rule_model_agreement` | Rule과 신규 모델의 일치 여부 |
| `embedding_model_agreement` | 검색 후보와 신규 모델의 일치 여부 |
| `neighbor_purity` | 유사 정답 사례의 category 일관성 |
| `new_merchant_flag` | 신규 상호 여부 |
| `category_risk` | 카테고리별 오류 비용 |

#### 권장 처리 구간

```text
고신뢰 + Rule/검색 일치
    → 신규 1차 모델 결과 자동 반영

중간 신뢰 + 후보 명확
    → LLM 재분류 후 조건부 반영

저신뢰·충돌
    → LLM 재분류 또는 Human Review

정보 부족·후보 누락
    → 기타 유지 또는 검수
```

### 2.9 Merchant Affinity / Relational Graph Modeling

사용자–가맹점, 가맹점–브랜드, 가맹점–MCC, 가맹점–카테고리 관계를 그래프로 모델링하면 텍스트가 모호한 사례를 보완할 수 있다.

```text
Node:
    merchant, user, brand, MCC, category, region, transaction

Edge:
    user-paid-at-merchant
    merchant-belongs-to-brand
    merchant-has-MCC
    merchant-labeled-as-category
    merchant-similar-to-merchant
```

후보 방법은 Node2Vec, GraphSAGE, GAT, heterogeneous GNN, link prediction이다. 다만 그래프 구축과 serving 비용이 높으므로, 1개월 결과에서 PG·marketplace·generic merchant의 잔여 오류가 큰 경우 중장기 후보로 검토한다.

### 2.10 기술 카테고리별 적용 우선도

| 기술 카테고리 | 대표 연구·사례 | 1개월 적용 판단 | 역할 |
|---|---|---:|---|
| Short-text / Subword | SVM Short-Text Classification, Better with Less | 필수 | 재현 가능한 신규 1차 모델 |
| Text + Tabular Fusion | Merchant Category Identification, 산업 enrichment 사례 | 필수 또는 우선 | MCC·Rule·브랜드 신호 결합 |
| Hierarchical Classification | Context-Fusion and Taxonomy-aware Attention | 부분 적용 | category path 검증과 Top-K masking |
| Weak Supervision | Rules and Crowdsourced Labels, Scalable Weak Supervision | 기반만 구축 | label source·confidence 저장 |
| Transaction Pattern | Merchant Category Identification | 후속 | 이름 정보 부족 사례 보완 |
| Embedding Retrieval | QuickBooks Rel-Cat, LLM-based Sentence Embeddings | 필수 | Top-K 후보와 유사 정답 검색 |
| LLM Reclassification | LLM-based Dual-Expert Classification | 필수 | 기타·저신뢰 재분류 |
| Confidence Routing | SME Categorisation, QuickBooks 운영 사례 | 필수 | 자동 반영·재분류·검수 분리 |
| Relational Graph | QuickBooks Rel-Cat | 중장기 | 관계 정보 기반 잔여 오류 보완 |

### 2.11 한 줄 정리

```text
가맹점 분류의 권장 방향은
재현 가능한 경량 1차 모델로 전체 건을 분류하고,
임베딩으로 후보를 검색한 뒤,
LLM은 기타·저신뢰 사례만 재분류하는 하이브리드 구조다.
```

---

## 3. 권장 시스템 아키텍처

### 3.1 설계 전제

기존 2023년 모델은 신규 시스템의 필수 구성요소로 사용하지 않는다.

```text
- 학습·평가 파이프라인이 없어 현재 성능을 재현하기 어려움
- 최신 가맹점 분포가 반영되지 않아 오분류가 많음
- threshold와 feature를 조정할 수 없음
- 담당자 의존적인 운영이 반복될 가능성이 큼
```

따라서 기존 모델은 다음 목적으로만 남긴다.

```text
- 신규 모델과 비교하는 legacy baseline
- 기존·신규 모델의 disagreement 분석
- 과거 결과 호환성을 위한 shadow output
```

신규 시스템의 첫 번째 산출물은 단순한 모델 weight가 아니라 다음을 포함하는 재현 가능한 파이프라인이어야 한다.

```text
- 데이터 추출과 snapshot 기준
- merchant entity와 중복 제거 기준
- label source와 label priority
- train / validation / test split
- feature 생성 코드
- 학습 configuration과 random seed
- 평가·오류 분석 코드
- model artifact와 metadata
- batch inference
- taxonomy / model / prompt version 관리
```

### 3.2 권장 전체 구조

```text
[Raw Merchant / Transaction Data]
        |
        v
[Dataset & Feature Builder]
    - raw / normalized merchant name
    - MCC / Rule / lexical feature
    - label source / taxonomy version
        |
        v
[New Primary Classifier]
    - char n-gram baseline
    - feature fusion model
    - lightweight Transformer challenger
    - Top-K probability
    - calibrated confidence
        |
        +---------------- high confidence ----------------+
        |                                                  |
        v                                                  v
[Reclassification Router]                           [Auto Approval]
    - other
    - low confidence
    - small margin
    - rule-model conflict
    - new merchant
        |
        v
[Hybrid Candidate Retrieval]
    - labeled merchant index
    - Category Label Card index
    - lexical similarity
    - primary model probability
    - MCC / Rule prior
        |
        v
[Top-3~Top-5 Candidate Package]
        |
        v
[LLM Reclassification]
    - candidate-only selection
    - none / insufficient / review 허용
    - structured JSON output
        |
        v
[Acceptance Policy]
    - hierarchy validation
    - model / rule / retrieval agreement
    - category risk
       /        |         \
      /         |          \
[Accept]   [Human Review]  [Other 유지]
      \         |          /
       \        |         /
        [Versioned Result & Feedback Data]
                     |
                     v
             [Next Training Dataset]
```

### 3.3 신규 모델 학습 데이터 설계

#### 3.3.1 Label Source Priority

```text
Priority 1. 최근 수기 검수 완료 정답
Priority 2. 명확한 브랜드·사업자 기준 gold mapping
Priority 3. 사용자 또는 운영팀 수정 결과
Priority 4. 검수된 고신뢰 LLM 결과
Priority 5. 고정밀 Rule weak label
Priority 6. 기존 2023년 모델 예측은 정답으로 사용하지 않음
```

기존 모델 예측을 신규 모델의 정답으로 사용하면 과거 오류가 그대로 전파될 수 있으므로, 비교 또는 disagreement feature로만 사용한다.

#### 3.3.2 권장 학습 데이터 스키마

```yaml
merchant_id: string
raw_merchant_name: string
normalized_merchant_name: string
brand_id: string | null
mcc: string | null
rule_category_id: string | null
rule_score: float | null
category_id: string
category_path: string
label_source: human | business_mapping | user_feedback | reviewed_llm | verified_rule
label_confidence: float | null
label_version: string
taxonomy_version: string
review_status: string
snapshot_date: date
```

#### 3.3.3 Split 원칙

단순 random row split은 동일 가맹점이나 동일 브랜드가 train과 test에 동시에 포함되는 leakage를 만들 수 있다.

```text
- merchant_id 또는 normalized merchant group 단위 분리
- 가능하면 out-of-time test 추가
- 동일 brand 지점이 양쪽에 섞이는지 확인
- 신규 merchant subset 별도 유지
- long-tail category subset 별도 유지
- 기타 재분류 gold subset 별도 유지
```

#### 3.3.4 필수 평가 subset

| Subset | 목적 |
|---|---|
| Random Merchant | 전체 평균 성능 |
| Out-of-time Merchant | 최신 분포 일반화 |
| New Merchant | 신규 상호 대응 |
| Long-tail Category | 희소 카테고리 성능 |
| Existing Other with Gold | 기타 감소 가능성 |
| Similar Category Pair | 유사 카테고리 경계 구분 |
| Generic Name | 이름 정보 부족 대응 |
| PG / Marketplace | 실제 seller 정보 부재 오류 |
| Rule–Model Conflict | 하이브리드 routing 평가 |
| High-volume Merchant | 사업 영향이 큰 오류 평가 |

### 3.4 추천 실험 설계

#### 3.4.1 신규 1차 모델 실험

```text
P0. 기존 2023년 SubwordCNN/ML 결과의 legacy baseline
P1. char n-gram TF-IDF + Logistic Regression
P2. char n-gram TF-IDF + Linear SVM
P3. P1/P2 probability + MCC/Rule/brand feature LightGBM
P4. lightweight Transformer text-only
P5. lightweight Transformer + MCC/Rule context
P6. P3와 P5의 champion-challenger 또는 stacking 비교
```

Champion은 가장 복잡한 모델이 아니라 다음 조건을 만족하는 모델로 선정한다.

```text
- 기존 모델보다 최신 test set에서 개선됨
- 신규·out-of-time 성능이 안정적임
- Macro-F1과 long-tail 성능이 수용 가능함
- confidence calibration이 가능함
- 대량 batch 처리 비용이 적정함
- 다른 담당자가 재학습과 평가를 재현할 수 있음
```

#### 3.4.2 임베딩 검색 실험

```text
R0. char TF-IDF / lexical retrieval
R1. multilingual dense embedding
R2. raw name + normalized name embedding
R3. Merchant Reference + Category Label Card dual index
R4. dense + lexical hybrid
R5. R4 + primary probability + MCC/Rule prior
```

평가 지표:

```text
Recall@3 / Recall@5
MRR
Candidate Miss Rate
Neighbor Purity
New Merchant Recall@5
Long-tail Recall@5
검색 latency / index size
```

#### 3.4.3 LLM 재분류 실험

```text
L0. 가맹점명 + 전체 category 목록을 이용한 direct LLM
L1. 신규 primary Top-K + LLM
L2. Embedding Top-K + LLM
L3. Primary + embedding + lexical + MCC/Rule Top-K + LLM
L4. L3 + Category Label Card
L5. L4 + 유사 수기 정답 사례
L6. L5 + hard-negative / 혼동 사례
L7. L6 + acceptance routing
```

권장 기준 모델은 `L0 direct LLM`이 아니라 `L3~L7 candidate-only LLM`이다. Direct LLM은 전체 taxonomy 자유 선택으로 인한 일관성·비용·환각 문제를 비교하기 위한 baseline으로 사용한다.

#### 3.4.4 End-to-end 평가 지표

| 영역 | 지표 |
|---|---|
| 1차 모델 | Accuracy, Macro-F1, Weighted-F1, category F1 |
| 일반화 | Out-of-time F1, New Merchant F1, Long-tail F1 |
| 후보 생성 | Recall@K, MRR, Candidate Miss Rate |
| Confidence | ECE, Brier Score, Risk–Coverage, Top1–Top2 Margin |
| LLM | Conditional Accuracy, Repeat Consistency, None Rate |
| 전체 구조 | Auto-approval Precision, Coverage, Review Rate, Other Reduction |
| 비용 | 고유 가맹점당 추론 비용, LLM 호출률, Token 비용 |
| 처리 성능 | Batch Throughput, P50/P95 Latency, Cache Hit Rate |

오류는 다음 단계로 분해해야 한다.

```text
1. Label / taxonomy 오류
2. Primary classifier 오류
3. Candidate retrieval miss
4. LLM candidate selection 오류
5. Acceptance / routing 오류
6. 가맹점명만으로 판단 불가능한 사례
```

### 3.5 1개월 단계별 구현 로드맵

1개월은 타이트하지만, **production 전면 전환이 아니라 end-to-end batch MVP와 shadow 평가**를 목표로 하면 가능하다.

#### 1주차: 데이터·평가 파이프라인 재구축

```text
- 최근 merchant·transaction·label snapshot 추출
- 수기 정답, Rule, 기존 모델 결과의 source와 날짜 구분
- merchant entity와 normalized merchant 기준 확정
- taxonomy version과 label priority 정의
- group/time-aware train/validation/test split 구현
- 공통 평가 subset과 지표 구현
- 기존 모델을 가능한 범위에서 동일 test set으로 재평가
```

핵심 산출물:

```text
Versioned Dataset
Dataset Builder
Evaluation Code
Legacy Baseline Report
Training Repository Skeleton
```

#### 2주차: 신규 1차 분류 모델 개발

```text
- char n-gram Logistic Regression 학습
- char n-gram Linear SVM 비교
- MCC/Rule/brand feature LightGBM fusion 실험
- 가능하면 lightweight Transformer challenger 병렬 수행
- class imbalance 처리
- probability calibration
- 성능·비용·처리량 비교
```

핵심 산출물:

```text
최소 2개 이상의 신규 모델
Calibrated Top-K Output
Category/Subset별 평가 리포트
Primary Champion과 Challenger 후보
```

#### 3주차: 임베딩 후보 검색과 LLM 재분류

```text
- 수기 정답 Merchant Reference Index 구축
- Category Label Card 작성과 embedding 생성
- Dense + lexical + primary probability hybrid Top-K 구현
- Candidate-only LLM prompt와 JSON schema 구현
- none_of_candidates / insufficient_information 처리
- direct LLM 대비 성능·비용 비교
- 자동 반영 / 검수 / 기타 유지 정책 구현
```

핵심 산출물:

```text
Hybrid Retrieval Module
Candidate Recall@K Report
LLM Reclassification Module
End-to-end Reclassification Report
```

#### 4주차: 통합 Batch MVP와 Shadow 검증

```text
- 전처리 → 신규 primary → routing → retrieval → LLM → 결과 저장 연결
- 과거 데이터 일부 backfill과 최근 데이터 shadow run
- cache, retry, idempotency, failure recovery 확인
- category 분포, 기타율, confidence, disagreement 확인
- model/data/taxonomy/prompt version 저장
- 재학습과 평가 절차 문서화
```

핵심 산출물:

```text
End-to-end Batch MVP
Shadow / Backfill 평가 리포트
신규 Primary Model Card
LLM 재분류 결과 데이터
Retraining Runbook
```

#### 1개월 완료 기준

```text
- 기존 파이프라인 없이 신규 데이터부터 결과까지 재현 가능함
- 기존 2023년 모델보다 최근 test set에서 개선되거나 유사 성능과 높은 재현성을 확보함
- 기타·저신뢰 gold subset에서 LLM 재분류가 1차 모델 단독보다 개선됨
- Candidate Recall@5와 자동 반영 Precision이 사전 기준을 충족함
- 생산 반영 전 shadow 판단에 필요한 정확도·비용·안정성 근거가 확보됨
```

### 3.6 1천만 건 규모 처리 원칙

1천만 거래 레코드를 모두 개별 분류하거나 LLM에 전달하지 않는다.

```text
원천 거래 1천만 건
    ↓ merchant_id / normalized merchant 기준 집계
고유 merchant entity
    ↓ exact gold / dictionary match와 cache 적용
신규 또는 재분류 대상 entity
    ↓ 신규 primary classifier
기타·저신뢰·충돌 entity만 LLM 재분류
```

권장 cache key:

```text
normalized_merchant_name
+ MCC(optional)
+ candidate_set_hash
+ taxonomy_version
+ primary_model_version
+ prompt_version
+ llm_model_version
```

이 구조는 LLM 호출량을 전체 거래 건수가 아니라 **고유 가맹점 중 재분류가 필요한 비율**로 제한한다.

### 3.7 기술 선택 우선순위

| 우선순위 | 기술·과제 | 판단 |
|---:|---|---|
| 1 | Versioned Dataset과 학습·평가 파이프라인 | 모든 모델 실험의 필수 기반 |
| 2 | Char n-gram 신규 baseline | 빠르고 재현 가능한 primary 후보 |
| 3 | MCC/Rule/brand Feature Fusion | 이름 기반 한계 보완 |
| 4 | 경량 Transformer Challenger | 의미 일반화 성능 비교 |
| 5 | Confidence Calibration과 Routing | 자동 처리와 재분류 구간 분리 |
| 6 | Merchant·Label Card Embedding Retrieval | LLM 후보 검색 |
| 7 | Candidate-only LLM Reclassification | 기타·저신뢰 보정 |
| 8 | Feedback Data 축적 | 지속적인 재학습 기반 |
| 9 | Transaction Pattern / Weak Supervision | MVP 이후 성능 고도화 |
| 10 | Relational GNN | 관계 정보가 필요한 잔여 오류가 큰 경우 검토 |

### 3.8 1개월 이후 권장 확장

MVP 결과를 기반으로 다음 순서로 확장한다.

```text
1. 신규 primary 모델의 shadow 기간 연장과 threshold 안정화
2. 기타 외 low-confidence·conflict 대상의 LLM routing 확대
3. 거래 금액·시간대·반복성 feature 추가
4. 검수·LLM 결과를 이용한 weak supervision 또는 student distillation
5. Hierarchical multi-head classifier 개발
6. category별 threshold 또는 correctness model 학습
7. PG·marketplace·generic 상호 오류가 크면 relational graph 검토
8. 반복적으로 LLM이 처리하는 패턴을 Rule 또는 경량 모델에 흡수
```

장기적으로는 LLM 호출 비율을 계속 늘리는 구조보다, 검수된 LLM 결과를 신규 1차 모델의 학습 데이터로 축적해 **LLM의 판단을 경량 모델로 이전하는 구조**가 바람직하다.

### 3.9 최종 권장안

```text
1. 기존 2023년 SubwordCNN/ML은 신규 시스템의 primary classifier로 유지하지 않는다.
2. 최신 데이터로 재학습 가능한 char n-gram·feature fusion 모델을 우선 구축한다.
3. 경량 Transformer는 동일 평가셋에서 challenger로 비교한다.
4. 신규 1차 모델은 전체 가맹점의 Top-K와 calibrated confidence를 생성한다.
5. 임베딩은 유사 정답과 Category Label Card를 검색해 후보를 보강한다.
6. LLM은 기타·저신뢰·충돌 가맹점만 candidate-only 방식으로 재분류한다.
7. 1개월의 목표는 production 전면 교체가 아니라 재현 가능한 Batch MVP와 shadow 검증이다.
8. 검수·LLM 결과를 다음 학습 데이터에 축적해 장기적으로 LLM 호출 비율을 낮춘다.
```

한 줄로 요약하면 다음과 같다.

```text
기존 모델에 LLM만 추가하는 것이 아니라,
재현 가능한 신규 1차 모델을 다시 구축하고
임베딩 기반 Top-K 후보 안에서 LLM이 기타·저신뢰 가맹점만 재분류하는 구조가 최적이다.
```

---

## 4. 참고 링크

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
- [QuickBooks AI Category Suggestions](https://quickbooks.intuit.com/learn-support/en-global/help-article/bank-transactions/ai-suggestions-help-match-categorise-bank/L8FHOh4AD_ROW_en)
