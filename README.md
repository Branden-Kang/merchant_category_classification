# 가맹점 카테고리 분류 모델링 논문 / 산업 사례 비교 및 고도화 제안

> 작성일: 2026-07-13  
> 목적: 기존 ML 모델과 룰 기반 분류 로직을 보유하고 있으며, LLM 적용 효과가 확인된 상황에서 가맹점 카테고리 분류 모델을 어떤 방향으로 고도화할지 검토하기 위한 모델링 중심 리서치 문서

---

## 0. 문서 범위

본 문서는 **가맹점명, 거래 설명, 거래 패턴, 관계 정보, 기존 룰 결과, MCC 등을 이용한 가맹점 카테고리 분류 모델링**에 직접 또는 간접적으로 관련된 연구와 산업 사례를 정리한다.

다음 항목은 이미 별도 정비 계획이 있다고 가정하고 독립적인 우선순위 과제에서는 제외한다.

- Rule / Dictionary / Merchant Normalization 정비
- MCC / 내부 소비 카테고리 / 마케팅 카테고리 분리 및 taxonomy 정비

다만 위 항목은 모델에서 제외되는 것이 아니라 다음과 같이 입력 피처, 후보 생성 신호, 학습 라벨 또는 운영 신호로 활용한다.

| 항목 | 본 문서에서의 활용 방식 |
|---|---|
| Rule / Dictionary | `rule_category`, `rule_score`, `rule_type`, `rule_model_agreement`, hard-rule early exit, weak label |
| Merchant Normalization | `clean_merchant_name`, `standard_merchant_name`, `brand_id`, 임베딩 검색 키, 중복 제거 키 |
| MCC | `mcc`, `mcc_embedding`, `mcc_prior`, 후보 카테고리 제한, LLM 입력 근거 |
| 내부 소비 / 마케팅 카테고리 | hierarchical 또는 multi-task output label |
| 수기 라벨 | gold label, 유사 사례 검색 인덱스, hard-negative 구성, calibration 데이터 |

본 문서의 핵심 문제는 다음과 같다.

```text
입력:
    가맹점명, 거래명, MCC, 룰 결과, 거래 패턴, 관계 정보

출력:
    계층형 카테고리
    + confidence
    + Top-K 후보
    + 근거 및 유사 사례
    + LLM/검수 필요 여부
```

### 0.1 논문 해석 시 주의사항

검토 논문들은 동일한 문제를 다루지 않는다. 따라서 논문에서 보고한 수치를 가맹점 카테고리 분류 성능으로 직접 해석하면 안 된다.

| 연구 유형 | 우리 문제와의 관련성 | 해석 방법 |
|---|---:|---|
| Product categorization | 높음 | 대규모 taxonomy와 Top-K 후보 재순위화 구조를 직접 참고 |
| Bank transaction categorization | 높음 | 짧고 노이즈가 많은 거래명, 계층 분류, 약지도 학습을 참고 |
| Merchant normalization / matching | 중간 | 모델 크기·비용·도메인 특화 학습의 근거로 활용 |
| Transaction foundation model | 중간 | merchant/MCC 임베딩을 주 모델에 주입하는 방법을 참고 |
| Personalized accounting category | 중간 | 사용자·기업별 관계와 graph/link prediction 구조를 참고 |
| 외부 enrichment API | 간접 | 출력 스키마와 운영 형태를 참고 |

---

## 1. 논문 / 산업 사례 비교표

### 1.1 연구 논문

| 논문 | 문제 정의 | 데이터 / 입력 | 핵심 방법 | 주요 장점 | 주요 한계 | 우리 서비스 적용 포인트 |
|---|---|---|---|---|---|---|
| **Merchant Category Identification Using Credit Card Transactions** (2020) | 신고된 merchant business type의 타당성 식별 | 71,668 merchant, 대규모 고객 거래 관계 및 시계열 | temporal transaction encoder + merchant affinity encoder | 텍스트가 아닌 거래 패턴과 merchant 관계 활용 | 충분한 거래 로그와 affinity 구축 필요 | 이름이 모호한 가맹점에 거래 패턴·유사 가맹점 feature 추가 |
| **Building Payment Classification Models from Rules and Crowdsourced Labels** (2018) | 룰 coverage 한계와 사용자 수정 라벨 활용 | 초기 룰, 결제 데이터, 사용자 수정 | 룰 bootstrap + 수정 라벨 기반 ML | 기존 룰을 학습 자산으로 전환 | 사용자 라벨 일관성 관리 필요 | 룰 결과를 feature, weak label, feedback label로 분리 |
| **Scalable and Weakly Supervised Bank Transaction Classification** (2023) | 수기 라벨 부족 상황에서 확장 가능한 거래 분류 | transaction text, 시간·금액 패턴, heuristic | labeling functions + FastText anchoring + Snorkel label model + multimodal DNN | 라벨 부족 완화, 신규 분류 과제 확장 용이 | labeling function 의존성·확률 calibration 문제 | 룰/WordNet/MCC/LLM을 약지도 소스로 통합 |
| **Two-headed DragoNet** (2023) | 금융 거래의 macro/micro 계층 분류 | merchant name + business activity | Transformer + Context Fusion + two heads + Taxonomy-aware Attention | 부모-자식 불일치 감소, 보조 설명 정보 활용 | merchant name만 사용할 때 성능이 크게 낮음 | 가맹점명 + 업종 설명/MCC/검색 문맥의 multi-input fusion 필요 |
| **SVM Short-Text Classification** (2024) | 약어와 노이즈가 많은 짧은 거래 설명 분류 | 실제 거래 설명 corpus | short-text similarity detector + SVM | 낮은 비용, 강한 전통 baseline | 의미 일반화 제한 | char n-gram/SVM을 반드시 비교 baseline으로 유지 |
| **E-commerce Product Categorization with LLM-based Dual-Expert** (2024) | 수천 개 taxonomy에서 세밀한 product category 선택 | product text, taxonomy path, category definitions | domain expert Top-K + LLM general expert reranking | long-tail·유사 카테고리·노이즈 라벨에 강점 | 2단계 추론 비용, product와 merchant 도메인 차이 | 임베딩/소형 모델이 Top-K 생성 후 LLM이 후보만 비교 |
| **Transaction Categorization with Relational Deep Learning in QuickBooks** (2025) | personalized accounting category 예측 | transaction, category, code, company 관계형 DB | custom Txn-Bert + Top-K NN + heterogeneous GNN link prediction | 반복 거래 early exit, cold-start 및 unseen 보완 | graph pipeline 복잡성, 개인화 category와 우리 taxonomy 차이 | 수기 정답 유사 사례 검색 + graph 확장 구조에 참고 |
| **Categorising SME Bank Transactions with Synthetic Data** (2025) | 비표준 거래명, 데이터 부족, 불균형 | SME transaction + synthetic data | synthetic generation + fine-tuned classifier + calibration | high-confidence 구간의 운영 가능성 제시 | synthetic 품질과 분포 왜곡 위험 | 소수 카테고리·신규 패턴 augmentation에 제한적으로 활용 |
| **Better with Less** (2025) | raw transaction에서 merchant name 이해·매칭 | rule/ESD/raw transaction, 7.8M merchant 후보 | encoder/decoder/encoder-decoder 계열의 LLM과 proprietary small model 비교 | 작은 도메인 모델의 비용·속도 우위, 대규모 배포 사례 | merchant matching 과제이며 category 분류 직접 증거는 아님 | LLM 결과를 소형 Transformer로 이전하는 근거 |
| **Enhancing Foundation Models... with LLM-based Sentence Embeddings** (2025) | transaction foundation model의 categorical index embedding 의미 손실 | MCC, merchant, location, transaction sequences | 다중 소스 정보 보강 + LLM sentence embedding offline initialization | 런타임 LLM 비용 없이 의미 정보 주입 | static embedding, category classification 직접 실험 아님 | Label Card/MCC/brand embedding 초기화 및 lightweight model 학습 |

### 1.2 산업 사례

| 사례 | 제공 기능 | 모델링 관점의 시사점 | 주의점 |
|---|---|---|---|
| **Plaid Enrich** | merchant, category, location, counterparty 등 enrichment | category 단일 출력보다 merchant metadata와 confidence를 함께 제공 | 외부 taxonomy와 내부 taxonomy mapping 필요 |
| **Yodlee Transaction Data Enrichment** | simple description, merchant name, category, geolocation | 정제 merchant name을 중간 산출물로 관리 | proprietary system이므로 내부 재학습 불가 |
| **Salt Edge Data Enrichment** | personal/business category, merchant identification, user-defined category | 개인·법인·마케팅 taxonomy를 multi-task로 분리 가능 | taxonomy mapping과 사용자 수정 처리 필요 |
| **QuickBooks production systems** | 기업별 거래 category 추천 | global model + company history + Top-K 추천의 조합 | 개인화 회계 category와 소비 category는 목적이 다름 |

---

## 2. 가맹점 카테고리 분류 모델링 기술 카테고리 정리

가맹점 카테고리 분류 모델링은 입력 데이터와 운영 역할을 기준으로 다음 10개 영역으로 정리할 수 있다.

```text
1. Short-text / Subword / Domain-specific Small Transformer
2. Embedding Retrieval / Semantic Initialization
3. Dual-Expert LLM Reranking
4. Text & Tabular Context Fusion
5. Hierarchical Multi-head / Taxonomy Constraint
6. Weak Supervision / Label Generation
7. Transaction Pattern / Time-series
8. Merchant Affinity / Relational Graph
9. Synthetic Data / LLM Teacher / Distillation
10. Confidence-based Routing and Human Review
```

---

### 2.1 Short-text / Subword / Domain-specific Small Transformer

#### 문제 특성

가맹점명과 거래 설명은 자연어 문장과 다르게 짧고, 약어·숫자·지점명·PG 표기·잘린 문자열이 많다.

```text
STARBUCKS 1234 SEOUL
SBUX GANGNAM
NAVERPAY SMARTSTORE
COUPANGPAYMENTS
ABC1234
OO상사
```

따라서 범용 LLM의 지식만으로 처리하기보다 거래 문자열의 고유한 표기 규칙을 직접 학습하는 모델이 필요하다.

#### 비교할 모델

```text
A. char n-gram + Linear SVM
B. char n-gram + Logistic Regression
C. fastText
D. 기존 SubwordCNN
E. DistilBERT / MiniLM 계열 fine-tuning
F. 4~6 layer domain-specific Transformer
G. LLM pseudo-label로 학습한 student model
```

**Better with Less**는 merchant matching 과제에서 1.5M~11M parameter의 도메인 특화 Transformer가 대형 pretrained model과 유사하거나 더 높은 성능을 보일 수 있음을 제시한다. **QuickBooks Rel-Cat**도 거래 문자열에 맞춘 tokenizer와 6-layer Txn-Bert를 학습했으며, 더 큰 12-layer 모델만이 항상 필요한 것은 아니라는 결과를 보고한다. 두 연구는 가맹점명과 거래 설명이 일반 자연어와 다른 표기 체계를 가지므로, 범용 대형 모델과 함께 작은 도메인 특화 모델을 독립적인 핵심 후보로 평가해야 한다는 근거를 제공한다.

#### 추천 입력

| 입력 | 설명 |
|---|---|
| `raw_merchant_name` | 원천 문자열 보존 |
| `clean_merchant_name` | 노이즈 제거 문자열 |
| `standard_merchant_name` | 정규화된 브랜드 또는 사업자명 |
| `raw_transaction_description` | 거래 설명 원문 |
| `token_pattern` | 영문/숫자/한글/특수문자 패턴 |
| `mcc` | MCC categorical input |
| `rule_category`, `rule_score` | 룰 결과와 강도 |
| `brand_id` | 알려진 브랜드 식별자 |

#### 권장 판단

기존 SubwordCNN을 바로 교체하기보다 다음을 동일 데이터에서 비교한다.

```text
SubwordCNN
vs. char n-gram SVM
vs. DistilBERT
vs. 4~6 layer custom Transformer
```

대규모 운영에서는 모델 크기보다 다음 지표를 함께 봐야 한다.

```text
Macro-F1
Long-tail F1
New merchant accuracy
1M건 추론 시간
1M건 GPU/CPU 비용
모델 업데이트 및 인덱스 재구축 비용
```

---

### 2.2 Embedding Retrieval / Semantic Initialization

임베딩의 역할은 두 가지로 구분해야 한다.

```text
역할 A: 유사 정답 가맹점과 후보 카테고리를 검색
역할 B: LLM의 의미 지식을 경량 모델의 embedding layer 초기값으로 주입
```

#### 2.2.1 검색용 임베딩

수기 라벨된 가맹점을 임베딩 인덱스로 만들고 신규 가맹점과 유사한 사례를 검색한다.

```text
입력 가맹점
    ↓
Dense embedding search
    + lexical / char similarity
    + brand / MCC filter
    ↓
유사 수기 정답 Top-N
    ↓
카테고리별 점수 집계
    ↓
후보 카테고리 Top-K
```

권장 인덱스는 2개다.

| 인덱스 | 내용 | 목적 |
|---|---|---|
| Merchant Reference Index | 수기 정답 가맹점명, 정규화명, 브랜드, category | 유사 정답 사례 검색 |
| Category Label Card Index | category name, taxonomy path, 정의, 포함·제외, 대표 예시 | 후보 category semantic retrieval |

#### Label Card 예시

```yaml
category_id: C102
category_name: 단체급식
taxonomy_path: 식음료 > 급식 > 단체급식
definition: 기업, 학교, 병원 등의 구내식당을 운영하거나 다수 인원에게 식사를 제공하는 업종
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

#### 검색 점수 예시

```text
retrieval_score(category) =
    0.35 × max_reference_similarity
  + 0.20 × top3_reference_mean
  + 0.20 × category_card_similarity
  + 0.15 × lexical_similarity
  + 0.10 × neighbor_support_ratio
```

가중치는 예시이며 validation set으로 학습하거나 조정한다.

#### Hybrid Retrieval 권장

가맹점명은 매우 짧으므로 dense embedding만 사용하면 중요한 표면 문자열이 약화될 수 있다.

```text
Dense semantic score
    + char n-gram / BM25 score
    + exact brand match
    + MCC prior
    + rule score
```

최종 후보 생성기는 Logistic Regression 또는 LightGBM으로 결합할 수 있다.

#### 2.2.2 LLM 기반 Semantic Initialization

**Enhancing Foundation Models in Transaction Understanding with LLM-based Sentence Embeddings**는 merchant, MCC, location 정보를 외부 문맥으로 보강한 뒤 LLM 임베딩을 오프라인 생성하여 경량 foundation model의 embedding layer 초기값으로 사용한다.

우리 서비스에서는 다음 실험이 가능하다.

```text
Category Card / Merchant Brand / MCC 설명
    ↓
LLM sentence embedding 오프라인 생성
    ↓
경량 Transformer의 category / merchant / MCC embedding 초기화
    ↓
내부 라벨로 task-specific fine-tuning
```

비교군:

```text
A. random initialization
B. 일반 sentence embedding initialization
C. LLM-generated semantic initialization
D. LLM initialization + trainable
E. LLM initialization + frozen
```

장점:

- 온라인 LLM 호출 없이 의미 지식 활용
- category label이 적은 long-tail에서 초기 표현 개선 가능
- MCC 코드의 숫자형 index embedding보다 의미 정보가 풍부함

주의점:

- 정적 임베딩은 신규 브랜드와 의미 변화 반영이 느림
- LLM 임베딩이 항상 분류 친화적이라는 보장은 없음
- 주기적 재생성 또는 fine-tuning이 필요

#### 임베딩 평가 지표

| 지표 | 의미 |
|---|---|
| `Recall@3/5/10` | 정답 category가 후보에 포함되는 비율 |
| `MRR` | 정답 후보의 평균 순위 |
| `Neighbor Purity` | 상위 이웃의 category 일관성 |
| `Candidate Miss Rate` | LLM 후보 목록에 정답이 없는 비율 |
| `New Merchant Recall@K` | 신규 가맹점에서 후보 생성 성능 |
| `Long-tail Recall@K` | 소수 category 후보 회수 성능 |

임베딩 단계의 목표는 Top-1 accuracy보다 **Recall@K를 높이는 것**이다.

---

### 2.3 Dual-Expert LLM Reranking

LLM 적용 효과가 이미 확인되었다면 가장 우선적으로 검토할 구조다.

#### Amazon Dual-Expert 구조

```text
Domain Expert
    - 도메인 데이터로 fine-tuning
    - Top-K candidate category 생성

General Expert
    - LLM
    - 후보별 미묘한 차이 비교
    - 최종 category 선택 및 이유 생성
```

Amazon 연구에서는 Domain Expert가 K=10 후보를 생성하고, LLM이 taxonomy path 및 LLM이 요약한 category definition을 이용해 최종 후보를 선택했다.

#### 우리 서비스에 맞춘 구조

```text
[Rule / Dictionary]
        ↓
[Small Domain Classifier]
        +
[Embedding Retriever]
        +
[MCC / Transaction Prior]
        ↓
[Candidate Fusion: Top-3~Top-10]
        ↓
[LLM General Expert]
        ↓
[Confidence / Conflict Checker]
```

Domain Expert는 단일 모델일 필요가 없다.

```text
Domain Expert =
    SubwordCNN probability
  + domain Transformer probability
  + embedding category score
  + rule score
  + MCC prior
```

#### LLM 입력 예시

```json
{
  "merchant": {
    "raw_name": "삼성웰스토리 판교구내식당",
    "normalized_name": "삼성웰스토리",
    "mcc": "...",
    "rule_evidence": ["구내식당"]
  },
  "candidates": [
    {
      "category": "단체급식",
      "taxonomy_path": "식음료 > 급식 > 단체급식",
      "definition": "기업·학교·병원 등에 단체 식사를 제공하거나 구내식당을 운영",
      "include": ["위탁급식", "사내식당"],
      "exclude": ["일반 음식점", "식자재 도매"],
      "similar_examples": ["아워홈 본사식당", "현대그린푸드 구내식당"]
    },
    {
      "category": "일반 음식점",
      "taxonomy_path": "식음료 > 음식점",
      "definition": "일반 소비자를 대상으로 조리 음식을 판매",
      "similar_examples": ["판교삼성식당"]
    }
  ]
}
```

#### 권장 출력 제한

```json
{
  "selected_category_id": "C102",
  "alternative_category_id": "C101",
  "decision": "selected | insufficient_information | review_required",
  "evidence": ["구내식당", "유사 정답 사례"],
  "ambiguity_type": "clear | category_boundary | insufficient_name | conflicting_evidence"
}
```

LLM이 후보 외 category를 자유 생성하지 못하도록 한다. 예외적으로 정답 누락 탐지를 위해 `none_of_candidates`만 허용한다.

#### 카테고리 정의 생성

Amazon 연구처럼 수기 라벨된 대표 사례를 LLM에 요약시켜 category definition을 생성할 수 있다.

```text
수기 정답 사례 100~1,000개
    ↓
대표 사례 / 하위 군집 샘플링
    ↓
LLM category summary 생성
    ↓
도메인 담당자 검수
    ↓
Versioned Label Card 저장
```

정의에는 반드시 다음이 포함되어야 한다.

```text
정의
포함 사례
제외 사례
혼동 category
대표 키워드
대표 브랜드
taxonomy path
```

---

### 2.4 Text & Tabular Context Fusion

Two-headed DragoNet 결과에서 merchant name만 사용한 경우 macro/micro F1이 낮았고, business activity를 함께 사용했을 때 성능이 크게 향상되었다. 이는 이름만으로 판단 불가능한 merchant가 많다는 점을 보여준다.

#### 권장 피처

| 그룹 | 예시 |
|---|---|
| Text | raw/clean merchant name, transaction description, business activity |
| Category context | MCC description, rule category, Label Card retrieval result |
| Numerical | avg/median amount, amount std, night ratio, weekend ratio |
| Behavioral | repeat ratio, unique customer count, refund ratio |
| Entity | brand_id, PG flag, online/offline, region |
| Retrieval | top1 similarity, top1-top2 margin, neighbor purity |

#### 모델 구조

```text
Text Encoder
    → text_embedding

MCC / Rule / Brand Embedding
    → categorical_embedding

Numerical Features
    → numerical_projection

Retrieval Evidence
    → retrieval_feature

concat
    → MLP / Transformer Fusion / LightGBM
    → category probabilities
```

#### 추천 비교

```text
Model A: text only
Model B: text + MCC
Model C: text + rule features
Model D: text + transaction pattern
Model E: text + retrieval features
Model F: all features
```

feature leakage를 피하기 위해 MCC나 rule이 실제 운영 시점에 사용할 수 있는 값인지 확인해야 한다.

---

### 2.5 Hierarchical Multi-head / Taxonomy Constraint

가맹점 category가 대분류·중분류·소분류 구조라면 flat classifier만으로는 계층 불일치가 발생한다.

#### 권장 구조

```text
Shared Encoder
    ↓
Major Head
Middle Head
Minor Head
Marketing Category Head(optional)
    ↓
Taxonomy Constraint
```

#### Loss 예시

```text
L_total =
    L_major
  + α × L_middle
  + β × L_minor
  + γ × L_marketing
  + δ × L_hierarchy_consistency
```

#### Taxonomy-aware 처리 방식

1. **Inference masking**

```text
예측된 대분류의 하위 category만 소분류 후보로 허용
```

2. **Hierarchy penalty**

```text
부모-자식 불일치 확률에 추가 loss 부여
```

3. **Path classification**

```text
유효한 taxonomy path 자체를 label로 예측
```

4. **LLM constrained selection**

```text
후보에 유효한 path만 제공
```

#### 평가 지표

```text
Major / Middle / Minor Macro-F1
Exact Path Accuracy
Hierarchy Violation Rate
Parent Correct but Child Wrong Rate
Top-K Leaf Recall
```

Two-headed DragoNet의 Taxonomy-aware Attention은 micro-category F1을 약 1% 개선했으므로, 계층 제약은 단독으로 큰 성능 향상보다 **잘못된 조합 제거와 운영 안정성**에 의미가 있다.

---

### 2.6 Weak Supervision / Label Generation

수기 라벨이 부족하거나 long-tail category가 많은 경우 기존 룰과 보조 정보를 약지도 학습에 활용한다.

#### Label Source

| 소스 | 용도 | 예상 신뢰도 |
|---|---|---:|
| exact brand rule | positive weak label | 매우 높음 |
| 정규식/keyword rule | weak label | 중~높음 |
| MCC mapping | prior / weak label | 중간 |
| WordNet / synonym | candidate evidence | 중간 이하 |
| embedding neighbor majority | weak label | similarity 구간별 상이 |
| LLM agreement label | pseudo-label | 검증 필요 |
| 사용자/CS 수정 | gold 또는 silver label | 중~높음 |
| human review | gold label | 높음 |

#### 권장 파이프라인

```text
1. Labeling Functions 생성
2. coverage / overlap / conflict 분석
3. label source별 정확도 추정
4. probabilistic 또는 hard weak label 생성
5. gold + weak label 혼합 학습
6. low-confidence / disagreement human review
7. 오류 사례를 labeling function 개선에 반영
```

#### FastText Anchoring 응용

Scalable Weak Supervision 연구는 소수 anchor 단어에서 FastText 유사 단어를 확장해 labeling function coverage를 높인다.

우리 서비스에서는 다음과 같이 활용할 수 있다.

```text
anchor: 약국
    → 온누리, 메디팜, pharmacy, pharm, drugstore ...

anchor: 카페
    → coffee, roastery, espresso, cafe, 커피 ...
```

다만 유사어 확장은 false positive가 많을 수 있으므로 수기 검수와 word boundary가 필요하다.

#### 주의점

- labeling function이 서로 강하게 상관되면 독립 가정이 깨질 수 있음
- label model 확률은 calibration되지 않을 수 있음
- weak label 오류를 DNN이 과적합할 수 있음
- confidence weighting이 항상 성능 향상을 보장하지 않음

따라서 small gold validation set은 필수다.

---

### 2.7 Transaction Pattern / Time-series

가맹점명 정보가 부족한 경우 거래 패턴이 보조 신호가 된다.

| 업종 | 패턴 예시 |
|---|---|
| 카페 | 소액, 오전·오후 피크, 높은 재방문 |
| 주점 | 야간·주말 비중 증가 |
| 병원 | 평일 주간, 금액 분산 큼 |
| 구독 | 월 단위 반복, 고정 금액 |
| 배달 | 점심·저녁 피크 |
| 편의점 | 소액, 전 시간대 분산 |

#### 추천 feature

```text
amount: mean, median, std, quantile
hour: daypart distribution, night ratio
calendar: weekday/weekend/holiday ratio
recurrence: fixed amount, cycle score, repeat ratio
customer: unique users, repeat users, new user ratio
operation: refund, cancel, online/offline, PG flag
```

#### 추천 활용

```text
Text model probability
    + transaction pattern features
    → LightGBM / MLP fusion
```

신규 merchant는 거래 이력이 적으므로 cold-start flag를 별도로 두고 text/embedding 가중치를 높여야 한다.

---

### 2.8 Merchant Affinity / Relational Graph

#### 기본 graph

```text
Node:
    merchant, brand, user, category, MCC, region, transaction

Edge:
    user - paid_at - merchant
    merchant - belongs_to - brand
    merchant - has_mcc - MCC
    merchant - located_in - region
    merchant - labeled_as - category
    transaction - generated_by - merchant
```

#### Rel-Cat에서 얻는 구조적 시사점

```text
1. 거래 문자열용 Txn-Bert 임베딩 생성
2. 같은 기업/사용자의 과거 거래에서 Top-K NN 검색
3. 충분히 유사하면 early exit
4. 그렇지 않으면 heterogeneous GNN으로 category link prediction
```

QuickBooks 연구에서 Top-K NN은 반복된 historical seen category에 강했지만 unseen category에는 거의 예측력이 없었다. GNN은 unseen subset을 보완했다.

우리 서비스에 대응하면 다음과 같다.

```text
유사 merchant/brand가 충분함
    → embedding retrieval 결과로 빠르게 처리

유사 사례가 없거나 category가 분산됨
    → transaction/brand/MCC/region graph로 보완
```

#### 적용 우선 조건

- 가맹점·브랜드·MCC·지역·거래 관계가 안정적으로 구축되어 있음
- text + embedding 모델의 오류가 generic merchant, PG, marketplace에 집중됨
- graph embedding을 batch로 생성하고 serving할 수 있음

그래프는 초기 MVP보다 중장기 과제로 두는 것이 적절하다.

---

### 2.9 Synthetic Data / LLM Teacher / Distillation

LLM이 실제로 높은 분류 효과를 보였다면 LLM 호출 결과를 일회성으로 소비하지 말고 학습 자산으로 전환한다.

#### Teacher-Student 구조

```text
Gold labels
    + high-confidence LLM labels
    + rule/model/LLM agreement labels
    ↓
Student model training
    ↓
대부분 student가 처리
    ↓
불확실한 일부만 LLM 호출
```

#### LLM pseudo-label 채택 기준

```text
- 동일 입력 반복 시 결과 일치
- 서로 다른 prompt에서 결과 일치
- candidate set 내에서 선택
- retrieval 근거와 일치
- category definition과 모순 없음
- 기존 모델과 LLM이 일치하거나 human 검수 완료
```

#### 제외 또는 낮은 가중치 대상

```text
- 실행마다 category 변경
- none_of_candidates 빈번
- 입력 정보 부족
- rule/embedding/LLM이 모두 충돌
- long-tail인데 유사 근거가 없음
```

#### Synthetic Data 권장 범위

합성 데이터는 전체 데이터를 대체하지 않고 다음에 제한한다.

```text
- 신규 category 초기 샘플
- 철자/언어/지점명 변형
- PG·결제대행 노이즈 변형
- hard-negative pair
- minority category 보강
```

합성 데이터는 실제 validation/test set과 분리하고, 합성 비율별 ablation을 수행한다.

---

### 2.10 Confidence-based Routing and Human Review

최종 운영은 단일 모델 정확도보다 **정확도-coverage-비용 trade-off**를 최적화해야 한다.

#### Confidence 입력

| 신호 | 설명 |
|---|---|
| `model_max_probability` | 주 분류 모델 최대 확률 |
| `entropy` | 예측 분포 불확실성 |
| `top1_top2_margin` | 1위·2위 차이 |
| `embedding_top1_similarity` | 가장 유사한 정답 사례 점수 |
| `neighbor_purity` | 이웃 category 일관성 |
| `candidate_recall_proxy` | 후보 생성 안정성 |
| `rule_model_agreement` | 룰과 모델 일치 |
| `model_llm_agreement` | 주 모델과 LLM 일치 |
| `new_merchant_flag` | 신규 merchant 여부 |
| `category_risk` | category별 오류 비용 |

#### 별도 Correctness Model

LLM이 직접 출력한 confidence만 사용하지 말고, 검증 라벨로 다음 확률을 학습한다.

```text
P(final prediction is correct | all confidence signals)
```

모델 후보:

```text
Logistic Regression
LightGBM
Calibrated classifier
```

#### Routing 예시

| 구간 | 예시 조건 | 처리 |
|---|---|---|
| Hard auto | exact rule + model/embedding 일치 | 자동 확정 |
| Model auto | calibrated correctness가 목표 precision 이상 | 자동 확정 |
| LLM rerank | 후보 Recall이 높고 Top-1 margin이 낮음 | LLM 후보 비교 |
| Human review | LLM도 불확실하거나 high-risk | 검수 큐 |
| Unknown | 정보 부족·OOD | 기타/미분류 유지 |

threshold는 고정값을 임의로 정하지 말고 validation set에서 category별 precision 목표에 맞춰 선택한다.

---

## 3. 권장 최종 시스템 아키텍처

### 3.1 오프라인 학습·준비 영역

```text
[Gold / Reviewed Labels]
        |
        +----------------------------+
        |                            |
        v                            v
[Reference Merchant Index]    [Category Label Cards]
        |                            |
        |                     LLM summary + human review
        |                            |
        +-------------+--------------+
                      v
             [Embedding Index]
                      |
                      v
[Rule / MCC / WordNet / LLM Pseudo Labels]
                      |
                      v
             [Weak Label Pipeline]
                      |
                      v
[Domain-specific Small Model Training]
    - SubwordCNN
    - char n-gram baseline
    - 4~6 layer Transformer
    - hierarchical heads
                      |
                      v
             [Calibration Model]
```

### 3.2 온라인 또는 대규모 배치 추론 영역

```text
[Raw Transactions / Merchant Data]
        |
        v
[Merchant Entity Deduplication]
    - merchant_id
    - normalized merchant
    - brand cluster
        |
        v
[Hard Rule Early Exit]
        |
        +---- confident match ----> [Final Category]
        |
        v
[Domain Model]
    - category probabilities
        |
        +
[Embedding Retrieval]
    - similar gold merchants
    - category Label Cards
        |
        +
[Soft Rule / MCC / Pattern Features]
        |
        v
[Candidate Fusion]
    - Top-K category
    - retrieval evidence
    - candidate margin
        |
        v
[Correctness / Conflict Checker]
        |
        +---- high confidence ----> [Auto Approval]
        |
        +---- resolvable ambiguity -> [LLM General Expert]
        |
        +---- insufficient evidence -> [Human Review / Unknown]
                                      |
                                      v
                              [Feedback Labels]
                                      |
                                      v
                              [Periodic Retraining]
```

### 3.3 LLM 호출 단위

1천만 건 원천 거래를 그대로 LLM에 보내지 않는다.

```text
원천 거래 10,000,000건
    ↓ merchant_id / normalized name 중복 제거
고유 merchant entity
    ↓ brand/branch clustering
대표 merchant unit
    ↓ rule/model/embedding routing
실제 LLM 호출 대상만 선별
```

동일 가맹점명·동일 후보 집합에 대한 LLM 결과는 cache한다.

---

## 4. 우리 서비스 관점의 적용 우선순위

현재 다음 요소가 있다고 가정한다.

```text
- SubwordCNN 기반 분류 모델
- WordNet 기반 유사어/동의어 매칭
- 룰 기반 분류
- 수기 라벨 데이터
- MCC 및 merchant normalization 정비 계획
- LLM 재분류의 유의미한 효과 확인
```

현재 LLM의 분류 효과가 확인되었으므로 LLM을 단순한 최후순위 fallback으로만 두지 않는다. 다만 전체 건을 LLM으로 직접 처리하는 구조도 권장하지 않는다. 임베딩·도메인 모델이 생성한 Top-K 후보를 재순위화하고, 고품질 pseudo-label과 오프라인 의미 임베딩을 생성하며, 저신뢰 사례만 보정하도록 역할을 분리한다. 이를 반영한 모델링 우선순위는 다음과 같다.

| 우선순위 | 추천 접근 | 이유 | 난이도 |
|---:|---|---|---:|
| 1 | **평가셋·오류 유형·비용 지표 정비** | 이후 모든 실험의 기준. 신규/long-tail/기타/충돌 subset 분리 필요 | 낮음~중간 |
| 2 | **Embedding Top-K 후보 검색** | 수기 정답을 직접 활용하며 LLM 자유도를 줄이고 Recall@K를 높임 | 중간 |
| 3 | **Dual-Expert LLM reranking** | 이미 확인된 LLM 효과를 비용 효율적으로 확대 | 중간 |
| 4 | **Confidence / correctness routing** | 자동 승인, LLM, human review를 분리해 운영 위험 통제 | 중간 |
| 5 | **도메인 특화 소형 Transformer 실험** | Better with Less와 Rel-Cat이 제시한 비용·속도·도메인 적합성 검증 | 중간 |
| 6 | **LLM Teacher / Distillation** | 반복 LLM 호출을 student model로 이전 | 중간~높음 |
| 7 | **Hierarchical Multi-head + taxonomy constraint** | 대·중·소분류 일관성과 long-tail 개선 | 중간 |
| 8 | **Text + Transaction Feature Fusion** | 이름만으로 불가능한 merchant 보완 | 중간~높음 |
| 9 | **Weak Supervision Pipeline** | 기존 룰/WordNet/MCC/LLM을 학습 데이터로 전환 | 중간 |
| 10 | **Merchant Affinity / Relational GNN** | generic/PG/marketplace 및 unseen 보완 | 높음 |

## 5. 추천 실험 설계

### 5.1 평가 데이터 구성

무작위 test set 하나만으로는 충분하지 않다.

| 평가 subset | 목적 |
|---|---|
| Random holdout | 전체 평균 성능 |
| Time-based holdout | 신규 시점 일반화 |
| New merchant | cold-start 성능 |
| Long-tail category | 소수 category 성능 |
| Other-to-category | 기존 기타 재분류 정확도 |
| Rule-model conflict | 하이브리드 판단 성능 |
| Similar category pairs | 경계 category 구분 |
| Generic merchant names | 이름 정보 부족 대응 |
| PG / marketplace | payment intermediary 처리 |
| High-volume merchant | 사업 영향도가 큰 사례 |

### 5.2 Baseline / Small Model 실험

```text
B0. Existing SubwordCNN
B1. char n-gram + Linear SVM
B2. char n-gram + Logistic Regression
B3. fastText
B4. DistilBERT fine-tuning
B5. 4-layer custom Transformer
B6. 6-layer custom Transformer
B7. Student model trained with LLM pseudo-labels
```

평가:

```text
Accuracy / Macro-F1 / Weighted-F1
Long-tail F1
New merchant F1
1M records latency and cost
Model size / memory
```

### 5.3 Embedding Retrieval 실험

```text
R0. lexical only: char TF-IDF / BM25
R1. multilingual sentence embedding
R2. BGE/E5 계열 dense embedding
R3. domain fine-tuned embedding
R4. dense + lexical hybrid
R5. hybrid + MCC/rule prior
R6. LLM-generated semantic initialization model
```

검색 단위 비교:

```text
merchant raw name
merchant raw + normalized name
merchant + MCC description
merchant + business activity
merchant + transaction context
```

평가:

```text
Recall@3 / Recall@5 / Recall@10
MRR
Neighbor purity
Candidate miss rate
Search latency
Index size
```

### 5.4 Dual-Expert 실험

```text
D0. Current LLM direct classification
D1. Model Top-5 + LLM selection
D2. Embedding Top-5 + LLM selection
D3. Rule + Model + Embedding Top-5 + LLM selection
D4. D3 + Label Card
D5. D4 + similar positive examples
D6. D5 + hard-negative examples
D7. D6 + none_of_candidates
```

K 비교:

```text
K = 3, 5, 10
```

평가:

```text
Candidate Recall@K
LLM conditional accuracy given gold in candidates
End-to-end accuracy
Long-tail accuracy
Candidate miss recovery rate
Average input tokens
Cost per 1M unique merchants
P50 / P95 latency
Repeat prediction consistency
```

중요한 분해 지표:

```text
End-to-end error =
    candidate generator miss
  + LLM reranking error
  + routing error
```

### 5.5 LLM Embedding Initialization 실험

```text
I0. Random embedding initialization
I1. Learned category index embedding
I2. General sentence embedding initialization
I3. LLM-generated Label Card embedding initialization
I4. LLM-generated MCC + merchant + category initialization
```

각 설정에서 embedding을 다음처럼 비교한다.

```text
frozen
trainable
partial fine-tuning
```

### 5.6 Hierarchical Multi-head 실험

```text
H0. Flat minor-category classifier
H1. Major + Minor multi-head
H2. Major + Middle + Minor multi-head
H3. H2 + hierarchy penalty
H4. H2 + inference mask
H5. H2 + LLM path reranking
```

평가:

```text
Major/Middle/Minor Macro-F1
Exact Path Accuracy
Hierarchy Violation Rate
Top-K Leaf Recall
```

### 5.7 Weak Supervision 실험

```text
W0. Gold only
W1. Rule weak labels only
W2. Gold + rule weak labels
W3. Gold + rule + MCC + WordNet
W4. W3 + embedding pseudo-label
W5. W4 + LLM pseudo-label
W6. W5 + source confidence weighting
```

반드시 다음을 기록한다.

```text
label source coverage
label source empirical accuracy
source conflict rate
category별 noise rate
```

### 5.8 Confidence Routing 실험

```text
C0. max softmax only
C1. temperature scaling
C2. model + embedding signals
C3. model + embedding + rule agreement
C4. C3 + LLM agreement
C5. LightGBM correctness model
```

평가:

```text
Coverage @ target precision
Error Rate @ Auto-approved
Risk-Coverage curve
ECE / Brier score
LLM call reduction
Human review reduction
Cost per correct classification
```

### 5.9 Scale 실험

```text
- 원천 건수 vs 고유 merchant 수
- deduplication ratio
- embedding batch throughput
- exact search vs ANN Recall@K
- LLM cache hit ratio
- category별 처리 시간
- 재처리 필요 비율
```

---

## 6. 모델 선택 및 운영 판단 기준

### 6.1 작은 모델과 LLM의 역할 분리

**Better with Less**의 결과를 그대로 가맹점 category 분류에 일반화할 수는 없지만, 다음 가설은 충분히 검증할 가치가 있다.

```text
가설:
가맹점명/거래명처럼 제한된 도메인 문자열은
대형 범용 LLM보다 작은 도메인 모델이
비슷한 정확도와 훨씬 낮은 비용을 달성할 수 있다.
```

따라서 운영 모델 선택은 다음처럼 한다.

| 상황 | 우선 모델 |
|---|---|
| 규칙적으로 반복되는 상호 | rule / dictionary |
| 수기 정답과 유사한 신규 지점 | embedding retrieval |
| 패턴이 학습된 일반 merchant | small domain classifier |
| 후보 간 의미 차이가 미세함 | LLM reranker |
| 정보가 부족하거나 충돌함 | human review / unknown |
| 관계 정보가 풍부하고 text가 모호함 | graph / relational model |

### 6.2 LLM이 특히 유리한 사례

```text
- category 이름이 의미적으로 겹침
- taxonomy path를 이해해야 함
- 소수 category에 학습 데이터가 적음
- 기존 라벨에 노이즈가 있음
- 상호명에서 업종을 상식적으로 추론할 수 있음
- 후보별 포함·제외 기준 비교가 필요함
```

### 6.3 LLM이 불리한 사례

```text
- 상호명이 정보가 없는 고유명사
- 실제 사업 내용이 이름과 다름
- PG/marketplace가 실제 seller를 가림
- category 후보에 정답이 없음
- 최신 로컬 사업체 정보가 필요함
- 수천만 건을 반복 처리해야 함
```

이런 사례에는 거래 패턴, 관계 정보, 외부 사업자 정보 또는 human review가 필요하다.

---

## 7. 단계별 구현 로드맵

### Phase 1: 4~8주 — 후보 검색 + Dual-Expert MVP

```text
1. 수기 정답 merchant reference index 구축
2. Category Label Card 작성
3. dense + lexical hybrid retrieval
4. Top-5 candidate 생성
5. LLM candidate-only reranking
6. current LLM direct 방식과 A/B 비교
```

성공 조건 예시:

```text
Recall@5가 충분히 높음
현재 direct LLM 대비 정확도 유지 또는 향상
토큰/비용 감소
잘못된 category 자유 생성 감소
```

### Phase 2: 4~8주 — Confidence Routing

```text
1. confidence feature 저장
2. correctness model 학습
3. auto / LLM / human review 구간 분리
4. category별 threshold 최적화
5. risk-coverage dashboard 구축
```

### Phase 3: 6~12주 — Small Domain Model / Distillation

```text
1. 4~6 layer Transformer 학습
2. LLM high-confidence pseudo-label 추가
3. hard-negative contrastive learning
4. student model과 LLM 성능·비용 비교
5. LLM 호출 비율 축소
```

### Phase 4: 8~16주 — Hierarchical / Feature Fusion

```text
1. major/middle/minor multi-head
2. taxonomy constraint
3. MCC / transaction pattern fusion
4. new merchant / long-tail 개선
```

### Phase 5: 중장기 — Graph / Relational Deep Learning

```text
1. merchant-brand-MCC-region graph
2. merchant similarity edge
3. graph embedding feature
4. Top-K retrieval early exit + GNN cascade
```

---

## 8. 최종 제안

현재 상황에서 가장 현실적인 목표 구조는 다음과 같다.

```text
Rule / Dictionary Early Exit
    ↓
Domain-specific Small Classifier
    + Embedding Retrieval
    + MCC / Rule / Pattern Features
    ↓
Top-K Candidate Generator
    ↓
High confidence → Auto Approval
Ambiguous but resolvable → LLM General Expert
Insufficient / high risk → Human Review or Unknown
    ↓
Reviewed labels + LLM labels → Student Retraining
```

핵심 제안은 다음과 같다.

1. **임베딩은 최종 분류기보다 Top-K 후보 검색기로 우선 사용한다.**
2. **LLM은 전체 taxonomy 자유 선택보다 후보 재순위화에 사용한다.**
3. **카테고리명만 주지 말고 taxonomy path와 Label Card를 제공한다.**
4. **LLM의 의미 지식은 오프라인 임베딩 초기화와 distillation으로 경량 모델에 이전한다.**
5. **대규모 운영의 주 모델은 도메인 특화 소형 Transformer를 포함해 비교한다.**
6. **룰·MCC·WordNet·LLM 결과는 약지도 라벨 소스로 재활용한다.**
7. **accuracy만이 아니라 Recall@K, coverage, 자동 승인 오류율, 비용을 함께 최적화한다.**
8. **merchant name만으로 판단 불가능한 케이스는 transaction pattern과 graph로 분리 대응한다.**

초기 운영 구조인 다음 조합:

```text
SubwordCNN
+ Text & Tabular Fusion
+ Hierarchical Classification
+ Weak Supervision
+ Confidence Routing
+ LLM Fallback
```

은 임베딩 검색과 LLM의 역할을 포함해 다음 구조로 확장하는 것이 적절하다.

```text
Rule / Dictionary
+ Domain-specific Small Transformer
+ Embedding Top-K Retrieval
+ Dual-Expert LLM Reranking
+ Hierarchical / Feature Fusion
+ Confidence Routing
+ LLM Teacher / Offline Semantic Initialization
+ Transaction Pattern / Relational Graph
```

---

## 9. 참고 링크

### Research Papers

- [E-commerce Product Categorization with LLM-based Dual-Expert Classification Paradigm](https://www.amazon.science/publications/e-commerce-product-categorization-with-llm-based-dual-expert-classification-paradigm)
- [Better with Less: Small Proprietary Models Surpass Large Language Models in Financial Transaction Understanding](https://arxiv.org/abs/2509.25803)
- [Hierarchical Classification of Financial Transactions Through Context-Fusion of Transformer-based Embeddings and Taxonomy-aware Attention Layer](https://arxiv.org/abs/2312.07730)
- [Transaction Categorization with Relational Deep Learning in QuickBooks](https://arxiv.org/abs/2506.09234)
- [Scalable and Weakly Supervised Bank Transaction Classification](https://arxiv.org/abs/2305.18430)
- [Enhancing Foundation Models in Transaction Understanding with LLM-based Sentence Embeddings](https://aclanthology.org/2025.emnlp-industry.61/)
- [Merchant Category Identification Using Credit Card Transactions](https://arxiv.org/abs/2011.02602)
- [Building Payment Classification Models from Rules and Crowdsourced Labels](https://lepo.it.da.ut.ee/~dumas/pubs/caise2018PaymentClassification.pdf)
- [Identifying Banking Transaction Descriptions via SVM Short-Text Classification](https://arxiv.org/abs/2404.08664)
- [Categorising SME Bank Transactions with Machine Learning and Synthetic Data Generation](https://arxiv.org/abs/2508.05425)

### Industry / Product References

- [Plaid Enrich](https://plaid.com/products/enrich/)
- [Plaid Enrich API Docs](https://plaid.com/docs/api/products/enrich/)
- [Yodlee Transaction Data Enrichment](https://developer.yodlee.com/resources/yodlee/transaction-data-enrichment/docs)
- [Salt Edge Data Enrichment](https://www.saltedge.com/products/data_enrichment)

---

## 부록 A. 최소 권장 LLM Prompt 구조

```text
System:
너는 가맹점 카테고리 taxonomy를 따르는 분류 전문가다.
반드시 제공된 후보 중 하나를 선택하거나 none_of_candidates를 반환한다.
상호명에 없는 사실을 임의로 생성하지 않는다.

Input:
- 원본 가맹점명
- 정규화 가맹점명
- MCC와 설명
- 룰 근거
- 후보 category Top-K
- 후보별 taxonomy path
- 후보별 정의 / 포함 / 제외
- 유사 수기 정답 사례
- 혼동 사례

Output JSON:
- selected_category_id
- alternative_category_id
- decision
- evidence
- ambiguity_type
- review_required
```

## 부록 B. 최소 권장 저장 스키마

```yaml
merchant_id: string
raw_merchant_name: string
normalized_merchant_name: string
brand_id: string | null
rule_category_id: string | null
rule_score: float
model_top1_category_id: string
model_top1_probability: float
model_top2_category_id: string
model_margin: float
embedding_top1_similarity: float
neighbor_purity: float
candidate_categories: array
llm_selected_category_id: string | null
llm_decision: string | null
final_category_id: string
final_correctness_probability: float
routing_action: auto | llm | review | unknown
model_version: string
embedding_version: string
label_card_version: string
prompt_version: string
```
