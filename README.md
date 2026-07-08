# 가맹점 카테고리 분류 모델링 논문 / 산업 사례 비교 정리

> 작성일: 2026-07-09  
> 목적: 기존 ML 모델 + 룰 기반 분류 로직`을 보유한 상황에서, 향후 가맹점 카테고리 분류 모델을 어떤 방향으로 고도화할지 검토하기 위한 모델링 중심 리서치 문서

---

## 0. 문서 범위

본 문서는 **가맹점 카테고리 분류 모델링**에 직접적으로 관련된 연구 논문과 산업 사례를 정리한다.

다음 항목은 이미 별도 정비 계획이 있다고 가정하고, 독립 섹션이나 우선순위에서는 제외한다.

- Rule / Dictionary / Merchant Normalization 정비
- MCC / 내부 소비 카테고리 / 마케팅 카테고리 분리 정비

다만 위 항목들은 모델링에서 완전히 배제되는 것은 아니며, 다음과 같은 방식으로 **모델 입력 또는 운영 피처**로 활용될 수 있다.

| 항목 | 본 문서에서의 취급 |
|---|---|
| Rule / Dictionary | 별도 정비 과제가 아니라, `rule_score`, `rule_category`, `rule_model_agreement` 같은 모델 feature 또는 confidence routing feature로 포함 |
| Merchant Normalization | 독립 주제가 아니라, 모델 입력 전 단계에서 생성된 `clean_merchant_name`, `standard_merchant_name`, `brand_id` feature로 포함 |
| MCC | taxonomy 설계 주제가 아니라, `mcc`, `mcc_embedding`, `mcc_to_model_category_prior` 같은 모델 feature로 포함 |
| 내부 소비 카테고리 / 마케팅 카테고리 | taxonomy 정비 주제가 아니라, multi-task 또는 multi-head 모델의 output label로 포함 |

즉, 본 문서의 초점은 다음과 같다.

```text
가맹점명/거래명/거래패턴/관계정보/기존 룰 결과/MCC 등을 입력으로 사용하여
가맹점 카테고리를 예측하는 모델 구조, 학습 방식, 추론 방식, 운영 전략을 정리한다.
```

---

## 1. 논문 / 산업 사례 비교표

| 논문 / 사례 | 문제정의 | 데이터 | 방법 | 장점 | 한계 | 우리 서비스 적용 포인트 | 링크 |
|---|---|---|---|---|---|---|---|
| **Merchant Category Identification Using Credit Card Transactions** (2020) | merchant가 신고한 business type 또는 merchant category가 실제 거래 패턴과 맞는지 식별하는 문제 | 실제 카드 거래 데이터. 논문은 71,668개 merchant와 433,772,755명 customer 간 거래 데이터를 사용 | **Temporal transaction encoder + merchant-merchant affinity encoder**를 결합한 multi-modal learning 구조 | 가맹점명 텍스트만 보지 않고 거래 시계열과 유사 가맹점 관계를 함께 사용 | 거래 패턴과 merchant affinity를 만들 수 있는 충분한 로그가 필요 | 가맹점명이 모호한 경우, **거래 패턴 + 유사 가맹점 관계 feature**를 추가하는 방향에 참고 가치가 큼 | [arXiv](https://arxiv.org/abs/2011.02602) |
| **Building Payment Classification Models from Rules and Crowdsourced Labels** (2018) | rule 기반 payment classification의 coverage 한계와 crowdsourced label의 bootstrapping 문제 해결 | 익명화된 금융기관 wire transfer/card payment 데이터, initial rules, customer-corrected labels | 초기 룰로 분류기를 bootstrap하고, 사용자 수정 라벨과 함께 ML 모델을 학습 | 기존 룰을 버리지 않고 학습 데이터 생성과 모델 개선에 재활용 가능 | crowdsourced label consistency 관리 필요 | 현재 보유한 룰 결과를 **weak label / model feature / feedback training data**로 전환하는 데 참고 | [PDF](https://lepo.it.da.ut.ee/~dumas/pubs/caise2018PaymentClassification.pdf), [ResearchGate](https://www.researchgate.net/publication/325564628_Building_Payment_Classification_Models_from_Rules_and_Crowdsourced_Labels_A_Case_Study) |
| **Scalable and Weakly Supervised Bank Transaction Classification** (2023) | 수작업 라벨이 부족한 은행 거래 데이터를 확장 가능하게 분류하는 문제 | bank transaction text, heuristic, domain knowledge 기반 weak label | preprocessing, text embedding, anchoring, label generation, neural network training으로 구성된 weak supervision pipeline | 라벨 부족 문제를 완화하고, 도메인 지식과 휴리스틱을 모델 학습에 연결 가능 | label noise 관리와 heuristic 품질이 중요 | 내부 룰/WordNet/MCC/외부 업종 정보를 **약지도 학습 파이프라인**으로 연결할 때 유용 | [arXiv](https://arxiv.org/abs/2305.18430) |
| **Hierarchical Classification of Financial Transactions / Two-headed DragoNet** (2023) | 금융 거래를 macro category와 micro category로 동시에 분류하는 계층형 거래 분류 문제 | merchant name과 business activity라는 두 개의 짧은 textual descriptor | Transformer encoder + context fusion + macro/micro two-head classifier + taxonomy-aware attention layer | 대분류와 소분류 간 계층 불일치 오류를 줄일 수 있음. card dataset macro-category F1 93%, current account dataset F1 95% 보고 | taxonomy 정비와 Transformer 운영 비용 필요 | 내부 카테고리가 대분류/중분류/소분류 구조라면 **hierarchical multi-head model**이 적합 | [arXiv](https://arxiv.org/abs/2312.07730) |
| **Identifying Banking Transaction Descriptions via SVM Short-Text Classification** (2024) | 짧고 약어가 많은 banking transaction description을 분류하는 문제 | 실제 고객 거래 설명으로 구성된 labeled corpus | character n-gram, word n-gram, SVM, short-text similarity detector 기반 2-stage classifier | 짧은 거래명에서는 n-gram + SVM 같은 전통 ML이 여전히 강한 baseline이 될 수 있음 | semantic generalization과 복잡한 merchant context 반영은 제한적 | SubwordCNN 외에 **char n-gram + SVM / LightGBM baseline**을 반드시 비교해야 함 | [arXiv](https://arxiv.org/abs/2404.08664) |
| **Transaction Categorization with Relational Deep Learning in QuickBooks / Rel-Cat** (2025) | QuickBooks의 transaction categorization을 relational database 위의 link prediction 문제로 재정의 | QuickBooks transaction data, relational database tables | Txn-BERT text encoder + GNN 기반 heterogeneous graph / link prediction | transaction description의 언어적 특성과 관계형 DB 구조를 함께 활용. cold-start와 scale 문제를 고려 | graph schema 설계와 production pipeline 난이도 높음 | 가맹점, 사용자, 거래, 카테고리, MCC, 사업자 정보를 graph로 연결할 수 있다면 **relational graph model** 후보로 적합 | [arXiv](https://arxiv.org/abs/2506.09234) |
| **Categorising SME Bank Transactions with Machine Learning and Synthetic Data Generation** (2025) | SME 거래 설명의 비표준성, 라벨 부족, 클래스 불균형 문제 해결 | SME bank transaction data와 synthetic transaction data | synthetic data generation + fine-tuned classifier + calibration methodology | held-out accuracy 73.49%, high-confidence accuracy 90.36% 보고. 데이터 부족/불균형에 대응 | synthetic data 품질과 calibration 품질에 민감 | LLM을 직접 분류기로 쓰기보다 **synthetic data 생성, 라벨 보강, confidence calibration**에 활용 가능 | [arXiv](https://arxiv.org/abs/2508.05425) |
| **Plaid Enrich** | unstructured transaction data를 merchant details, category, location, counterparty 등으로 보강하는 transaction enrichment 문제 | 카드/은행 거래 데이터 및 비정형 transaction data | ML 기반 transaction enrichment API | 산업에서는 category classification을 merchant details, location, counterparty와 함께 제공 | 외부 taxonomy가 내부 taxonomy와 다를 수 있음 | 우리 서비스도 최종적으로는 category만이 아니라 **category + confidence + merchant metadata**를 함께 출력하는 구조가 적합 | [Product](https://plaid.com/products/enrich/), [API Docs](https://plaid.com/docs/api/products/enrich/) |
| **Yodlee Transaction Data Enrichment** | 사용자가 이해하기 어려운 거래 설명을 merchant name, category, geolocation 등으로 보강 | bank/card transaction data | proprietary ML engine 기반 transaction enrichment | simple description, merchant name, category, geolocation을 contextualized manner로 제공 | 외부 솔루션이므로 내부 데이터와 taxonomy에 맞게 재학습하기 어려움 | 모델 출력 설계 시 **simple description, merchant name, category, geolocation, confidence**를 함께 고려 | [Docs](https://developer.yodlee.com/resources/yodlee/transaction-data-enrichment/docs), [Retail Category](https://developer.yodlee.com/resources/yodlee/transaction-data-enrichment/docs/retail-category) |
| **Salt Edge Data Enrichment** | raw transactional data를 개인/비즈니스 거래 카테고리와 actionable insight로 변환 | account aggregation 및 transaction data | self-learning engine 기반 categorisation API, user-defined category 지원 | personal/business categorisation, user-defined categories, merchant identification API 지원 | 내부 category 체계와 맞추기 위한 mapping 필요 | 개인 소비 분석과 사업자/법인 거래 분석을 함께 한다면 **multi-task 또는 multi-taxonomy output** 구조에 참고 | [Product](https://www.saltedge.com/products/data_enrichment), [Docs](https://docs.saltedge.com/data_enrichment/v5/) |

---

## 2. 가맹점 카테고리 분류 모델링 기술 카테고리 정리

가맹점 카테고리 분류 모델링은 하나의 알고리즘으로 해결하기보다 다음 요소의 조합으로 설계하는 것이 적합하다.

```text
1. Short-text / Subword Modeling
2. Tabular Feature Fusion
3. Hierarchical Multi-head Classification
4. Weak Supervision / Label Generation
5. Transaction Pattern / Time-series Modeling
6. Merchant Affinity / Graph Modeling
7. Relational Deep Learning
8. LLM / Synthetic Data / Fallback
9. Confidence-based Routing
```

---

### 2.1 Short-text / Subword 기반 분류 모델

#### 설명

가맹점명과 거래 설명은 일반 문서보다 훨씬 짧고, 표기 변형이 많다.

예시:

```text
STARBUCKS 1234 SEOUL
스타벅스강남2호
SBUX GANGNAM
네이버페이_스마트스토어
쿠팡페이먼츠
OO상사
ABC1234
```

이런 데이터에서는 word-level token보다 character n-gram, subword, 음절/자모 단위 표현이 더 효과적일 수 있다.

#### 대표 방법

- char n-gram + Logistic Regression
- char n-gram + SVM
- char n-gram + LightGBM
- fastText
- SubwordCNN
- character-level CNN
- CNN + rule score ensemble

#### 장점

- 짧은 텍스트에 강함
- 오타, 약어, 띄어쓰기 오류에 비교적 강함
- inference 비용이 낮음
- 대량 실시간 처리에 적합
- 현재 SubwordCNN 구조와 연결하기 쉬움

#### 한계

- 텍스트 외 거래 패턴을 충분히 반영하기 어려움
- `OO상사`, `ABC`, `스마트스토어`처럼 이름만으로 모호한 가맹점에는 한계
- 신규 브랜드 또는 플랫폼 거래에 대한 의미적 일반화가 제한적

#### 추천 적용

현재 SubwordCNN이 이미 있다면 바로 폐기하지 말고, 다음 baseline과 비교하는 것이 좋다.

```text
Baseline A: char n-gram + Linear SVM
Baseline B: char n-gram + LightGBM
Baseline C: fastText
Baseline D: existing SubwordCNN
Baseline E: SubwordCNN + tabular feature ensemble
```

#### 추천 입력 feature

| feature | 설명 |
|---|---|
| `raw_merchant_name` | 원천 가맹점명 |
| `clean_merchant_name` | 정제된 가맹점명 |
| `raw_transaction_description` | 카드/계좌 거래 원문 |
| `merchant_alias_tokens` | 별칭, 축약어, 영문/한글 변형 |
| `mcc` | 기존 MCC code |
| `rule_category` | 기존 룰 기반 예측 결과 |
| `rule_confidence` | 룰 매칭 신뢰도 |
| `wordnet_similarity_score` | WordNet 또는 synonym 기반 유사도 점수 |

---

### 2.2 Tabular Feature Fusion 모델

#### 설명

가맹점 분류는 텍스트만으로 끝나지 않는다. 거래 금액, 시간대, 반복 결제, 고객군, MCC, 기존 룰 결과 등 tabular feature를 함께 사용하면 분류 성능이 크게 개선될 수 있다.

#### 대표 방법

- LightGBM / XGBoost
- CatBoost
- Wide & Deep
- DeepFM
- MLP + categorical embedding
- text model output + tabular model ensemble
- late fusion / stacking

#### 모델 구조 예시

```text
Text Encoder:
    clean_merchant_name, transaction_description
        → text_embedding

Categorical Embedding:
    mcc, region, brand_id, rule_category
        → categorical_embedding

Numerical Feature Encoder:
    avg_amount, median_amount, weekend_ratio, repeat_ratio
        → numerical_embedding

Fusion:
    concat(text_embedding, categorical_embedding, numerical_embedding)
        → MLP / LightGBM / XGBoost
        → category prediction
```

#### 장점

- 가맹점명만으로 모호한 케이스를 보완 가능
- 기존 rule 결과와 MCC를 feature로 자연스럽게 활용 가능
- 실무 데이터 구조와 잘 맞음
- feature importance 분석이 가능함

#### 한계

- feature leakage 관리 필요
- batch aggregation feature 생성 주기 관리 필요
- 신규 가맹점은 거래 패턴 feature가 부족할 수 있음

#### 추천 적용

SubwordCNN 단독 모델보다 다음 형태가 실무적으로 더 안정적이다.

```text
Model 1: SubwordCNN only
Model 2: SubwordCNN probability + LightGBM
Model 3: char n-gram TF-IDF + tabular feature + LightGBM
Model 4: Transformer embedding + tabular feature + MLP
Model 5: Stacking ensemble
```

---

### 2.3 Hierarchical Multi-head Classification 모델

#### 설명

가맹점 카테고리는 일반적으로 계층형 구조를 가진다.

예시:

```text
식음료
  ├── 카페
  ├── 음식점
  ├── 주점
  └── 배달

쇼핑
  ├── 편의점
  ├── 마트
  ├── 백화점
  └── 온라인쇼핑

생활
  ├── 미용
  ├── 세탁
  ├── 수리
  └── 반려동물
```

이때 모든 카테고리를 하나의 flat label로 예측하면 다음과 같은 문제가 생긴다.

```text
대분류: 쇼핑
소분류: 카페
```

즉, 대분류와 소분류가 서로 맞지 않는 계층 불일치 오류가 발생할 수 있다.

#### 대표 사례

- Two-headed DragoNet
- macro-category head + micro-category head
- taxonomy-aware attention layer

#### 추천 모델 구조

```text
Shared Encoder:
    merchant_text + transaction_features + mcc_feature + rule_feature
        → shared_representation

Output Head 1:
    대분류 예측

Output Head 2:
    중분류 예측

Output Head 3:
    소분류 예측

Consistency Layer:
    hierarchy constraint
    parent-child valid mapping
    invalid combination penalty
```

#### Loss 설계 예시

```text
L_total =
    L_major
  + α * L_middle
  + β * L_minor
  + γ * L_hierarchy_consistency
```

#### 장점

- 대분류/중분류/소분류를 동시에 최적화 가능
- category hierarchy를 모델에 반영 가능
- 신규 소분류 추가 시 구조적으로 확장 가능
- 마케팅 카테고리와 소비 카테고리를 multi-task로 함께 다루기 좋음

#### 한계

- label hierarchy 품질이 중요함
- 특정 소분류에 데이터가 부족하면 head별 imbalance가 커짐
- 모델 평가가 복잡해짐

#### 추천 적용

룰/카테고리 체계 정비가 별도로 진행된다면, 모델링 관점에서는 다음처럼 활용하는 것이 좋다.

```text
Input:
    clean_merchant_name
    mcc
    rule_category
    transaction_pattern
    merchant_affinity_feature

Output:
    consumer_major_category
    consumer_middle_category
    consumer_minor_category
    marketing_category(optional)
```

---

### 2.4 Weak Supervision / Label Generation 기반 모델

#### 설명

가맹점 카테고리 분류는 수작업 라벨을 충분히 확보하기 어렵다. 또한 신규 가맹점과 롱테일 카테고리는 라벨이 부족하다. 이때 기존 룰, WordNet, MCC, 외부 업종 정보, 사용자 수정 라벨 등을 이용해 weak label을 만들고 supervised model을 학습할 수 있다.

#### 대표 사례

- Scalable and Weakly Supervised Bank Transaction Classification
- Building Payment Classification Models from Rules and Crowdsourced Labels

#### 추천 파이프라인

```text
1. 기존 룰/WordNet/MCC/외부 업종 정보로 weak label 생성
2. weak label별 confidence score 부여
3. 충돌 label 제거 또는 soft label화
4. high-confidence weak label로 1차 모델 학습
5. low-confidence / disagreement sample human review
6. 검수 결과로 gold label set 구축
7. gold + weak label 혼합 학습
```

#### Label source 예시

| label source | 활용 방식 |
|---|---|
| 기존 rule result | high-confidence weak label |
| WordNet / synonym score | 후보 category score |
| MCC mapping result | prior label 또는 auxiliary feature |
| 외부 POI category | 보조 label |
| 사용자 수정 category | feedback label |
| 상담/CS correction | high-value gold label |
| LLM generated category | 검수 전 후보 label |

#### 장점

- 라벨 부족 문제 완화
- 기존 룰/사전을 학습 데이터로 전환 가능
- 신규 카테고리 추가 시 초기 데이터 확보가 쉬움
- human review 비용을 줄일 수 있음

#### 한계

- label noise 관리 필요
- weak label을 그대로 정답으로 쓰면 모델이 룰 오류를 학습할 수 있음
- confidence calibration이 중요함

#### 추천 적용

현재 룰 기반 분류 로직이 이미 있다면, 이를 다음 세 가지로 분해해 모델에 연결하는 것이 좋다.

```text
rule_category       → categorical feature
rule_confidence     → numerical feature
rule_generated_label → weak label
```

---

### 2.5 Transaction Pattern / Time-series 기반 모델

#### 설명

가맹점의 업종은 거래 패턴에서 강하게 드러날 수 있다.

예시:

| 업종 | 거래 패턴 예시 |
|---|---|
| 카페 | 소액 결제, 오전/오후 피크, 재방문 빈도 높음 |
| 주점 | 야간 결제 비중 높음, 주말 비중 높음 |
| 병원 | 평일 주간 결제, 금액 분산 큼 |
| 주유소 | 차량 이용 패턴, 특정 금액대 반복 |
| 구독 서비스 | 월 1회 반복, 금액 고정 |
| 배달 | 점심/저녁 피크, 소액~중간 금액 |
| 편의점 | 소액, 전 시간대 분포 |

#### 대표 사례

- Merchant Category Identification Using Credit Card Transactions

#### 추천 feature

| feature group | 예시 |
|---|---|
| amount pattern | `avg_amount`, `median_amount`, `amount_std`, `amount_percentile` |
| time pattern | `hour_distribution`, `daypart_ratio`, `night_ratio` |
| weekday pattern | `weekday_ratio`, `weekend_ratio`, `holiday_ratio` |
| recurrence | `repeat_payment_ratio`, `fixed_amount_ratio`, `monthly_cycle_score` |
| customer behavior | `repeat_user_ratio`, `unique_user_count`, `new_user_ratio` |
| refund/cancel | `refund_ratio`, `cancel_ratio` |
| channel | `online_offline_flag`, `pg_flag`, `app_payment_flag` |

#### 모델 구조 예시

```text
Merchant-level Aggregation:
    transaction logs over last 30/60/90 days
        → merchant_behavior_vector

Temporal Encoder:
    hourly / daily amount-count sequence
        → temporal_embedding

Fusion:
    merchant_text_embedding
    + merchant_behavior_vector
    + temporal_embedding
        → category prediction
```

#### 장점

- 이름만으로 모호한 가맹점에 강함
- MCC 오류 또는 비정상 업종 신고 탐지에도 활용 가능
- 온라인/오프라인, 구독/일회성, 식음료/생활/교통 등 구분에 유용

#### 한계

- 신규 가맹점 cold-start에는 약함
- aggregation window 선택이 중요함
- 시즌성, 이벤트, 프로모션 영향을 받을 수 있음

---

### 2.6 Merchant Affinity / Graph 기반 모델

#### 설명

같은 고객이 함께 이용하는 가맹점, 같은 지역/상권에 있는 가맹점, 같은 브랜드/사업자에 속하는 가맹점은 업종이 유사하거나 보완적인 관계를 가질 수 있다.

예시:

```text
User-Merchant bipartite graph
Merchant-Merchant co-visit graph
Merchant-Category graph
Merchant-MCC graph
Merchant-Brand graph
Merchant-Region graph
```

#### 대표 사례

- Merchant Category Identification Using Credit Card Transactions
- Transaction Categorization with Relational Deep Learning in QuickBooks

#### 추천 그래프 구조

```text
Node:
    user
    merchant
    brand
    category
    mcc
    region
    transaction

Edge:
    user - paid_at - merchant
    merchant - belongs_to - brand
    merchant - has_mcc - mcc
    merchant - located_in - region
    merchant - labeled_as - category
    merchant - similar_to - merchant
```

#### 모델 방법

- Node2Vec / DeepWalk
- GraphSAGE
- GAT
- Heterogeneous GNN
- Link prediction
- Graph embedding + LightGBM
- Text embedding + graph embedding fusion

#### 장점

- 텍스트만으로 모호한 가맹점 보완
- 롱테일 가맹점도 인접 노드 정보로 보완 가능
- 관계형 데이터베이스 구조를 모델에 반영 가능
- Merchant-merchant affinity를 직접 학습 가능

#### 한계

- 그래프 구축 비용과 운영 복잡도가 큼
- 온라인 실시간 추론보다 offline/batch embedding 생성에 적합
- 개인정보 및 사용자 행동 그래프 사용 시 보안/거버넌스 필요

#### 추천 적용

가맹점명 기반 모델의 오류가 다음 유형에서 많다면 graph feature를 추가할 가치가 크다.

```text
- OO상사, ABC 등 generic merchant name
- 네이버페이/카카오페이/토스페이 등 payment gateway 거래
- 스마트스토어/마켓플레이스 거래
- 사업자명만 있고 업종 단서가 약한 거래
- 동일 상호가 여러 업종에서 사용되는 경우
```

---

### 2.7 Relational Deep Learning 모델

#### 설명

QuickBooks Rel-Cat 사례처럼 transaction categorization을 관계형 DB 위의 graph/link prediction 문제로 볼 수 있다. 이 접근은 단순히 merchant graph를 만드는 것보다 더 넓게, 여러 테이블 간 관계를 모델링한다.

#### 관계형 데이터 예시

```text
merchant_table
transaction_table
user_table
mcc_table
category_table
brand_table
region_table
rule_result_table
review_label_table
```

#### 모델 구조 예시

```text
Text Encoder:
    transaction_description, merchant_name
        → text_embedding

Relational Graph Encoder:
    merchant-user-category-mcc-rule-review graph
        → graph_embedding

Prediction:
    transaction/merchant node
        → category link prediction
```

#### 장점

- 기존 DW/DM 구조를 모델에 직접 활용 가능
- transaction, merchant, user, category 간 관계를 통합적으로 학습
- cold-start 상황에서도 일부 관계 정보가 있으면 보완 가능

#### 한계

- feature store / graph store / training pipeline 설계 난이도 높음
- 전체 시스템 아키텍처 변경이 필요할 수 있음
- 단순 텍스트 분류 대비 개발 기간이 길 수 있음

#### 추천 적용

다음 조건을 만족할 때 중장기 고도화 후보로 적합하다.

```text
- 가맹점/거래/사용자/카테고리/룰/검수 결과가 관계형 DB에 잘 정리되어 있음
- 단순 텍스트 모델의 성능 한계가 명확함
- 신규 가맹점 cold-start와 롱테일 문제가 큼
- batch training과 graph embedding serving이 가능함
```

---

### 2.8 LLM / Synthetic Data / Fallback 모델

#### 설명

LLM은 가맹점 카테고리 분류에서 다음 용도로 사용할 수 있다.

```text
1. zero-shot / few-shot category candidate generation
2. low-confidence case fallback
3. synthetic training data generation
4. 라벨 설명 생성
5. 검수자 보조
6. 신규 카테고리 초기 데이터 생성
```

대량 실시간 거래 전체를 LLM으로 분류하는 것은 비용, 지연시간, 재현성, 개인정보 측면에서 부담이 크다. 따라서 실무적으로는 **전면 대체보다 보조 모델**로 사용하는 것이 적합하다.

#### 대표 사례

- Categorising SME Bank Transactions with Machine Learning and Synthetic Data Generation
- LLMs for SME transaction categorisation 관련 연구
- 산업 transaction enrichment solution의 AI/ML 보강 흐름

#### 추천 활용 방식

```text
Primary Model:
    SubwordCNN / Transformer / LightGBM / Graph model

Routing:
    if confidence < threshold
    or rule-model conflict
    or new merchant
    or high-risk category:
        call LLM fallback

LLM Output:
    candidate_category_top3
    reasoning
    uncertainty
    required_review_flag
```

#### Prompt 입력 예시

```text
Input:
- clean merchant name
- raw transaction description
- MCC
- existing rule category
- transaction amount pattern
- time pattern
- candidate categories
- category definitions

Task:
- Choose top-3 likely categories.
- Explain the reason.
- Mark whether human review is required.
```

#### Synthetic data 생성 예시

```text
For category = "반려동물":
    generate merchant-like transaction descriptions
    include Korean/English variants
    include PG/payment gateway noise
    include branch suffix
    include amount/time pattern metadata
```

#### 장점

- 신규/롱테일 가맹점 대응
- 라벨 부족 카테고리 보강
- 검수자에게 설명 제공 가능
- 카테고리 정의 변경 시 빠르게 후보 라벨 생성 가능

#### 한계

- inference 비용
- 개인정보 처리 이슈
- 출력 일관성
- hallucination 가능성
- prompt/taxonomy 버전 관리 필요

#### 추천 적용

LLM은 아래처럼 limited-scope로 적용하는 것이 바람직하다.

```text
- 전체 거래의 5~20% 이하 low-confidence case에만 적용
- LLM 결과를 바로 정답으로 확정하지 않고 candidate로 사용
- human review와 결합
- 검수된 LLM 결과만 학습 데이터로 반영
- 민감 정보는 masking/tokenization 후 전달
```

---

### 2.9 Confidence-based Routing 모델

#### 설명

모든 거래를 자동 분류하는 것보다, 모델 신뢰도에 따라 자동 확정 / 후보 추천 / LLM fallback / human review를 분리하는 것이 운영적으로 안전하다.

#### 추천 routing rule

```text
High confidence + rule agreement:
    자동 확정

Medium confidence:
    자동 확정 + 샘플링 검수

Low confidence:
    LLM fallback 또는 human review

Rule-model conflict:
    human review 또는 LLM reasoning

High-risk category:
    threshold 상향
```

#### confidence feature

| feature | 설명 |
|---|---|
| `max_probability` | softmax 최대 확률 |
| `entropy` | 예측 분포 불확실성 |
| `top1_top2_margin` | 1위와 2위 점수 차이 |
| `rule_model_agreement` | 룰 결과와 모델 결과 일치 여부 |
| `model_ensemble_agreement` | 여러 모델 간 예측 일치 여부 |
| `category_risk_level` | 카테고리별 오류 비용 |
| `new_merchant_flag` | 신규 가맹점 여부 |
| `low_data_category_flag` | 학습 데이터 부족 카테고리 여부 |

#### 평가 지표

| 지표 | 의미 |
|---|---|
| `Accuracy` | 전체 정확도 |
| `Macro-F1` | 소수 카테고리 성능 |
| `Weighted-F1` | 거래량이 많은 카테고리 중심 성능 |
| `Top-3 Accuracy` | 후보 추천 운영 가능성 |
| `Coverage @ Confidence` | 특정 confidence 이상에서 자동 처리 가능한 비율 |
| `Error Rate @ Auto-approved` | 자동 확정 구간 오류율 |
| `ECE` | confidence calibration 품질 |
| `New Merchant Accuracy` | 신규 가맹점 일반화 성능 |
| `Long-tail Category F1` | 롱테일 카테고리 성능 |
| `Review Reduction Rate` | 검수량 감소 효과 |
| `LLM Fallback Success Rate` | LLM fallback 후 정답 개선율 |

---

## 3. 기술 카테고리별 대표 사례 매핑

| 기술 카테고리 | 대표 사례 | 핵심 키워드 | 적용 우선도 |
|---|---|---|---|
| Short-text / Subword | SVM short-text classification, SubwordCNN, fastText | char n-gram, subword, merchant name, short text | 높음 |
| Tabular Feature Fusion | LightGBM/XGBoost, Wide & Deep, ensemble | transaction pattern, mcc feature, rule score, feature fusion | 높음 |
| Hierarchical Multi-head | Two-headed DragoNet | macro/micro category, taxonomy-aware, multi-head | 높음 |
| Weak Supervision | Scalable and Weakly Supervised Bank Transaction Classification | weak label, heuristic, rule-generated label, label generation | 높음 |
| Transaction Pattern | Merchant Category Identification Using Credit Card Transactions | time-series, amount pattern, customer behavior | 중간~높음 |
| Merchant Affinity / Graph | Merchant affinity encoder, Rel-Cat | graph, co-visit, heterogeneous GNN, link prediction | 중장기 |
| Relational Deep Learning | QuickBooks Rel-Cat | relational DB, Txn-BERT, GNN, link prediction | 중장기 |
| LLM / Synthetic Data | SME transaction synthetic data | LLM, synthetic data, few-shot, calibration | 제한적 fallback |
| Confidence Routing | calibration, high-confidence prediction, human review | confidence, entropy, margin, review routing | 매우 높음 |

---

## 4. 한 줄 정리

현재 가맹점 카테고리 분류 모델링은 아래 흐름으로 발전하고 있다고 볼 수 있다.

1. **Short-text / Subword 기반**
2. **Tabular Feature Fusion 기반**
3. **Hierarchical Multi-head 기반**
4. **Weak Supervision 기반**
5. **Transaction Pattern / Time-series 기반**
6. **Merchant Affinity / Graph 기반**
7. **Relational Deep Learning 기반**
8. **LLM / Synthetic Data / Fallback 기반**
9. **Confidence-based Routing 기반**

즉, 초기에는 `가맹점명 + 짧은 텍스트 분류` 중심이었다면, 최근에는 `거래 패턴`, `merchant affinity`, `관계형 DB graph`, `계층형 taxonomy`, `weak supervision`, `LLM fallback`, `confidence routing`을 결합하는 방향으로 발전하고 있다.

---

## 5. 우리 서비스 관점의 적용 우선순위

현재 서비스에 이미 다음 요소가 있다고 가정한다.

```text
- SubwordCNN 기반 가맹점 카테고리 분류 모델
- WordNet 기반 유사어/동의어 매칭
- 룰 기반 분류 로직
- MCC / 내부 소비 카테고리 / 마케팅 카테고리 정비 계획
- Merchant normalization 정비 계획
```

따라서 본 문서에서는 rule/dictionary/normalization/taxonomy 정비 자체를 우선순위에서 제외하고, 모델링 고도화만 우선순위로 정리한다.

| 우선순위 | 추천 접근 | 이유 | 예상 난이도 |
|---|---|---|---|
| 1순위 | **SubwordCNN + char n-gram / SVM / LightGBM baseline 비교** | 짧은 거래명에서는 전통 ML이 여전히 강하므로 현재 모델의 실질적 우위 확인 필요 | 낮음 |
| 2순위 | **Text + Tabular Feature Fusion** | MCC, rule score, 거래 금액/시간대/반복성 feature를 결합하면 텍스트만 쓰는 모델보다 안정적 | 중간 |
| 3순위 | **Hierarchical Multi-head Classification** | 대분류/중분류/소분류 또는 소비/마케팅 카테고리를 동시에 예측 가능 | 중간 |
| 4순위 | **Weak Supervision Pipeline** | 기존 룰/WordNet/MCC 결과를 weak label과 feature로 활용해 라벨 부족 문제 완화 | 중간 |
| 5순위 | **Confidence-based Routing** | 자동 확정, 후보 추천, LLM fallback, human review를 분리해 운영 안정성 확보 | 중간 |
| 6순위 | **Transaction Pattern Feature 추가** | 가맹점명이 모호한 케이스에서 금액/시간대/반복성 패턴이 중요한 보조 신호 | 중간~높음 |
| 7순위 | **Merchant Affinity / Graph Feature 추가** | PG/마켓플레이스/롱테일 가맹점처럼 이름만으로 어려운 케이스 보완 | 높음 |
| 8순위 | **LLM Fallback / Synthetic Data Generation** | LLM은 전체 대체보다 low-confidence case 보정, synthetic data, 검수자 보조에 적합 | 중간~높음 |
| 9순위 | **Relational Deep Learning / GNN 고도화** | QuickBooks Rel-Cat처럼 관계형 DB 전체를 graph로 활용하는 장기 고도화 방향 | 높음 |

---

## 6. 추천 실험 설계

### 6.1 Baseline 실험

```text
Experiment 1:
    char n-gram + Linear SVM

Experiment 2:
    char n-gram + LightGBM

Experiment 3:
    fastText

Experiment 4:
    existing SubwordCNN

Experiment 5:
    SubwordCNN + WordNet score + rule score ensemble
```

목표:

```text
현재 SubwordCNN이 짧은 거래명 baseline 대비 실제로 우수한지 검증
```

---

### 6.2 Feature Fusion 실험

```text
Text features:
    clean_merchant_name
    raw_transaction_description

Categorical features:
    mcc
    rule_category
    region
    online_offline_flag

Numerical features:
    rule_confidence
    wordnet_similarity_score
    avg_amount
    median_amount
    weekend_ratio
    night_ratio
    repeat_payment_ratio
    refund_ratio
```

모델 후보:

```text
Model A: LightGBM only
Model B: Text embedding + LightGBM
Model C: SubwordCNN output probability + LightGBM
Model D: Transformer embedding + MLP
Model E: Stacking ensemble
```

---

### 6.3 Hierarchical Multi-head 실험

```text
Input:
    text_embedding
    + mcc_embedding
    + rule_feature
    + transaction_pattern_feature

Output heads:
    major_category
    middle_category
    minor_category
    marketing_category(optional)
```

Loss:

```text
L_total =
    L_major
  + 0.7 * L_middle
  + 0.5 * L_minor
  + 0.3 * L_marketing
  + 0.2 * L_hierarchy_consistency
```

평가:

```text
- Major category accuracy
- Middle category macro-F1
- Minor category macro-F1
- Hierarchy violation rate
- Top-3 minor category accuracy
```

---

### 6.4 Weak Supervision 실험

```text
Label sources:
    rule result
    WordNet match
    MCC mapping
    external POI/business category
    user correction
    human review

Training set:
    high-confidence weak label
    + gold label
    + sampled low-confidence review label
```

실험 비교:

```text
Model A: gold label only
Model B: weak label only
Model C: gold + weak label
Model D: gold + weak label + confidence weighting
Model E: gold + weak label + label smoothing
```

평가:

```text
- 전체 macro-F1
- long-tail category F1
- noisy label robustness
- review sample efficiency
```

---

### 6.5 Confidence Routing 실험

```text
Confidence features:
    max_probability
    entropy
    top1_top2_margin
    rule_model_agreement
    ensemble_agreement
```

Routing policy 예시:

| 구간 | 조건 | 액션 |
|---|---|---|
| Auto approve | confidence >= 0.90 and rule_model_agreement = true | 자동 확정 |
| Soft approve | confidence >= 0.80 | 자동 확정 + 샘플링 검수 |
| Candidate mode | 0.60 <= confidence < 0.80 | Top-3 후보 저장 |
| LLM fallback | confidence < 0.60 or rule conflict | LLM 후보 분류 |
| Human review | high-risk category or LLM uncertain | 검수 큐 전달 |

평가:

```text
- 자동 처리율
- 자동 처리 구간 오류율
- 검수량 감소율
- LLM 호출 비율
- LLM fallback 후 오류 개선율
```

---

## 7. 추천 시스템 아키텍처

```text
[Raw Transaction / Merchant Data]
        |
        v
[Prepared Modeling Features]
    - clean_merchant_name
    - standard_merchant_name
    - raw_transaction_description
    - mcc
    - rule_category
    - rule_confidence
    - wordnet_similarity_score
    - amount/time/repeat pattern
    - merchant/user/category relation features
        |
        v
[Short-text Encoder]
    - char n-gram
    - fastText
    - SubwordCNN
    - Transformer encoder
        |
        v
[Feature Fusion Layer]
    - text embedding
    - categorical embedding
    - numerical transaction features
    - rule/MCC features
        |
        v
[Category Prediction Layer]
    - flat classifier
    - hierarchical multi-head classifier
    - multi-task consumer/marketing category classifier
        |
        v
[Pattern / Affinity / Graph Enhancement]
    - transaction time-series
    - merchant affinity
    - heterogeneous graph embedding
        |
        v
[Confidence & Conflict Checker]
    - max probability
    - entropy
    - top1-top2 margin
    - rule-model agreement
    - category risk level
        |
        +-----------------------------+
        |                             |
        v                             v
[Auto Approval]              [LLM / Human Review Fallback]
        |                             |
        v                             v
[Final Category Output]      [Corrected Label + Reason]
        |                             |
        +-------------+---------------+
                      v
              [Feedback Training Data]
```

---

## 8. 최종 제안

현재 상황에서는 다음 방향이 가장 현실적이다.

```text
SubwordCNN 유지
    + char n-gram / SVM / LightGBM baseline 검증
    + Text & Tabular Feature Fusion
    + Hierarchical Multi-head Classification
    + Weak Supervision
    + Confidence Routing
    + Transaction Pattern / Merchant Affinity Feature
    + LLM Fallback
```

특히 LLM은 다음 이유로 전면 교체보다 fallback이 적합하다.

```text
- 전체 거래에 적용하기에는 비용과 지연시간 부담이 큼
- 룰과 경량 모델이 이미 잘 맞히는 고신뢰 케이스가 많음
- 개인정보/거래정보를 외부 LLM API로 보낼 경우 보안 검토 필요
- 분류 일관성과 재현성 관리가 필요함
- low-confidence, 신규 가맹점, 룰-모델 충돌 케이스에서만 호출해도 효과가 큼
```

따라서 최종적으로는 다음과 같은 구조를 권장한다.

```text
Primary classifier:
    SubwordCNN / char n-gram / LightGBM / Transformer

Feature fusion:
    text + mcc + rule score + transaction pattern

Advanced feature:
    merchant affinity / graph embedding

Output:
    hierarchical category + confidence score + top-k candidates

Operation:
    confidence-based auto approval / LLM fallback / human review

Feedback:
    reviewed label → weak/gold training data → periodic retraining
```

---

## 9. 참고 링크

### Research Papers

- [Merchant Category Identification Using Credit Card Transactions](https://arxiv.org/abs/2011.02602)
- [Building Payment Classification Models from Rules and Crowdsourced Labels: A Case Study](https://lepo.it.da.ut.ee/~dumas/pubs/caise2018PaymentClassification.pdf)
- [Scalable and Weakly Supervised Bank Transaction Classification](https://arxiv.org/abs/2305.18430)
- [Hierarchical Classification of Financial Transactions Through Context-Fusion of Transformer-based Embeddings and Taxonomy-aware Attention Layer](https://arxiv.org/abs/2312.07730)
- [Identifying Banking Transaction Descriptions via Support Vector Machine Short-Text Classification Based on a Specialized Labelled Corpus](https://arxiv.org/abs/2404.08664)
- [Transaction Categorization with Relational Deep Learning in QuickBooks](https://arxiv.org/abs/2506.09234)
- [Categorising SME Bank Transactions with Machine Learning and Synthetic Data Generation](https://arxiv.org/abs/2508.05425)

### Industry / Product References

- [Plaid Enrich](https://plaid.com/products/enrich/)
- [Plaid Enrich API Docs](https://plaid.com/docs/api/products/enrich/)
- [Yodlee Transaction Data Enrichment](https://developer.yodlee.com/resources/yodlee/transaction-data-enrichment/docs)
- [Yodlee Retail Transaction Enrichment](https://developer.yodlee.com/resources/yodlee/transaction-data-enrichment/docs/retail-category)
- [Salt Edge Data Enrichment](https://www.saltedge.com/products/data_enrichment)
- [Salt Edge Data Enrichment Docs](https://docs.saltedge.com/data_enrichment/v5/)

### Reference README Style

- [Look-Alike / README.md](https://github.com/Branden-Kang/recommendation-system/blob/main/Look-Alike/README.md)
