# 신규 가맹점 업종 분류 모델 추천안

## 1. 제안 요약

현재 사용 중인 `SubwordCNN + WordNet + Ensemble` 구조를 대체하기 위한 신규 모델로 다음 구조를 권장합니다.

```text
가맹점명 원문
   │
   ├─ 한국어 Subword Encoder
   │    └─ KLUE-RoBERTa 또는 mDeBERTa-v3
   │
   ├─ UTF-8 Byte·문자 Encoder
   │    └─ 경량 CNN + Tiny Transformer
   │
   └─ 선택적 메타데이터 Encoder
        └─ 주소, 브랜드, 설명, 과거 업종, 지역 정보
                 │
             Gated Fusion
                 │
       ┌─────────┼─────────┐
       │         │         │
     대분류     중분류     세분류
       │         │         │
       └────── 최종 업종 ───┘
                 │
       Confidence 및 미분류 판정
```

권장 모델의 임시 명칭은 다음과 같습니다.

```text
MerchantHybridEncoder
```

핵심 구성은 다음과 같습니다.

1. 사전학습 한국어 Encoder로 문맥과 단어 의미를 학습합니다.
2. Byte·문자 Encoder로 오타, 특수문자, 영문·한글 혼합, 신규 브랜드를 처리합니다.
3. 두 Encoder의 최종 Logit이 아니라 중간 특징을 결합합니다.
4. 대분류, 중분류, 세분류를 동시에 학습합니다.
5. Confidence 보정과 미분류 탐지를 별도 기능으로 구성합니다.
6. 운영 단계에서는 작은 Student 모델로 경량화합니다.

---

## 2. 기존 모델의 한계

현재 구조는 다음과 같습니다.

```text
문자 입력
→ SubwordCNN
→ Class Logit

단어 입력
→ Embedding
→ Flatten
→ MLP
→ Class Logit

두 Logit 결합
→ 최종 분류
```

### 2.1 WordNet의 문맥 학습 한계

현재 WordNet은 단어 Embedding 전체를 펼친 후 MLP로 처리합니다.

```python
embedding
→ flatten
→ linear
→ classification
```

단어 사이의 관계를 명시적으로 계산하는 Attention 구조가 없기 때문에 다음 표현의 차이를 충분히 학습하기 어렵습니다.

```text
서울 약국 카페
서울 카페 옆 약국
약국용품 판매점
```

단어 위치는 Flatten된 입력에 일부 반영되지만 문맥적 상호작용을 직접 모델링하지는 않습니다.

### 2.2 SubwordCNN의 지역적 패턴 중심 학습

현재 CNN Kernel 크기가 2~5라면 다음과 같은 짧은 문자 패턴을 찾는 데 강점이 있습니다.

```text
약국
카페
편의
한식당
```

그러나 문자열 전체 관계나 멀리 떨어진 표현 사이의 관계를 학습하는 데는 한계가 있습니다.

### 2.3 늦은 단계의 Ensemble

현재 Ensemble은 두 모델의 Class Logit을 결합합니다.

```text
SubwordCNN Logit
+
WordNet Logit
→ Linear
```

이 방식에서는 문자 모델과 단어 모델의 중간 특징이 서로 상호작용하지 못합니다.

예를 들어 문자 Encoder가 발견한 브랜드 특징과 단어 Encoder가 발견한 업종 특징을 Feature 수준에서 결합하기 어렵습니다.

### 2.4 Vocabulary 의존성

다음과 같은 입력은 기존 Vocabulary에서 제대로 처리되지 않을 수 있습니다.

```text
CAFE봄
메가MGC커피
GS25강남2호점
OO약국_24H
```

문자 CNN이 일부 보완할 수 있지만 단어 기반 모델에서는 미등록 단어가 `<unk>`로 변환될 가능성이 높습니다.

### 2.5 업종 계층 구조 미활용

현재 메타데이터에는 다음과 같은 계층 정보가 존재할 수 있습니다.

```text
대분류
→ 중분류
→ 세분류
```

예:

```text
음식
→ 카페
→ 커피전문점
```

기존 모델이 최종 세분류만 예측하면 이러한 계층 관계를 학습에 충분히 사용하지 못합니다.

---

## 3. 추천 모델: MerchantHybridEncoder

## 3.1 한국어 Subword Encoder

첫 번째 Branch는 한국어 또는 다국어 사전학습 Encoder를 사용합니다.

권장 후보:

| 모델 | 장점 | 적합한 환경 |
|---|---|---|
| `KLUE-RoBERTa-base` | 한국어 분류 기준선 구축이 쉬움 | 한국어 가맹점명이 대부분인 경우 |
| `mDeBERTa-v3-base` | 다국어와 영문 혼합 대응 | 영문 브랜드와 외국어 입력이 많은 경우 |
| 경량 한국어 BERT 계열 | 추론 속도와 메모리 절감 | CPU 운영 또는 낮은 Latency가 필요한 경우 |

초기 개발에서는 `KLUE-RoBERTa-base`를 첫 기준선으로 사용하는 것이 좋습니다.

기본 구조:

```text
Tokenizer
→ Pretrained Encoder
→ Pooling
→ Dropout
→ Classification Head
```

### Pooling 방법

`[CLS]` Token만 사용하는 방법과 Mean Pooling을 비교합니다.

```python
token_features = encoder_output.last_hidden_state
mask = attention_mask.unsqueeze(-1)

mean_embedding = (
    (token_features * mask).sum(dim=1)
    / mask.sum(dim=1).clamp(min=1)
)
```

가맹점명처럼 짧은 텍스트에서는 Mean Pooling이 전체 토큰 정보를 안정적으로 모을 수 있습니다.

비교 대상:

```text
CLS Pooling
Mean Pooling
CLS + Mean Concatenation
Attention Pooling
```

---

## 3.2 UTF-8 Byte·문자 Encoder

두 번째 Branch는 원문을 UTF-8 Byte 또는 Unicode 문자 단위로 처리합니다.

```text
"GS25강남점"
→ UTF-8 Byte Sequence
→ Byte Embedding
→ CNN Downsampling
→ Tiny Transformer
→ Byte Feature
```

추천 구조:

```text
Byte Embedding
→ Conv1d Downsampling
→ Transformer Encoder 2~4개
→ Attention Pooling
```

초기 설정 예:

| 항목 | 권장 범위 |
|---|---:|
| Byte Vocabulary | 256 + 특수 토큰 |
| Embedding 크기 | 64~128 |
| Hidden 크기 | 128~256 |
| Transformer Layer | 2~4 |
| Attention Head | 4 |
| 최대 Byte 길이 | 64~128 |

Byte Branch의 목적은 다음과 같습니다.

- 미등록 브랜드 처리
- 띄어쓰기 오류 대응
- 영문·한글 혼합 처리
- 숫자와 지점명 처리
- 특수문자와 약칭 처리
- 형태소 분석 실패 보완
- 신규 단어와 오타 대응

예:

```text
GS25강남역점
gs25 강남역
GS_25강남
지에스25강남점
```

Subword Tokenizer만 사용하는 모델보다 원문 형태 변화에 강건한 Branch를 만들 수 있습니다.

---

## 3.3 Gated Feature Fusion

기존 방식처럼 최종 Logit끼리 결합하지 않고 두 Encoder의 중간 특징을 결합합니다.

```python
token_feature = self.token_encoder(...)
byte_feature = self.byte_encoder(...)

token_feature = self.token_projection(
    token_feature
)

byte_feature = self.byte_projection(
    byte_feature
)

gate_input = torch.cat(
    [token_feature, byte_feature],
    dim=-1,
)

gate = torch.sigmoid(
    self.gate(gate_input)
)

fused_feature = (
    gate * token_feature
    + (1.0 - gate) * byte_feature
)
```

Gate는 입력마다 두 Branch의 중요도를 다르게 조정합니다.

예:

```text
"김치찌개 전문점"
→ 한국어 Encoder 비중 증가

"GS25_강남역2호점"
→ Byte Encoder 비중 증가

"CAFE아띠"
→ 두 Encoder를 함께 사용
```

다른 Fusion 후보도 함께 비교할 수 있습니다.

```text
Concatenation + MLP
Gated Sum
Cross Attention
Mixture of Experts
Bilinear Fusion
```

첫 구현은 `Concatenation + MLP` 또는 `Gated Sum`이 가장 단순하고 안정적입니다.

---

## 4. 계층형 Multi-task Classification

업종 메타데이터에 대분류와 중분류가 존재한다면 세 개의 Head를 동시에 학습합니다.

```python
top_logits = self.top_classifier(
    fused_feature
)

mid_logits = self.mid_classifier(
    fused_feature
)

fine_logits = self.fine_classifier(
    fused_feature
)
```

예:

```text
대분류: 음식
중분류: 카페
세분류: 커피전문점
```

Loss 예:

```python
total_loss = (
    fine_loss
    + 0.3 * mid_loss
    + 0.2 * top_loss
)
```

가장 중요한 세분류 Loss에 가장 큰 비중을 두고, 대분류와 중분류는 보조 학습 신호로 사용합니다.

### 계층 일관성 검사

추론 시 다음과 같은 결과를 검사할 수 있습니다.

정상:

```text
대분류: 의료
중분류: 약국
세분류: 일반의약품판매점
```

불일치:

```text
대분류: 음식
중분류: 약국
세분류: 일반의약품판매점
```

불일치 처리 방법:

1. 세분류가 속한 실제 대·중분류로 보정합니다.
2. 계층이 일치하는 후보 중 최고 점수를 선택합니다.
3. 계층 불일치가 크면 미분류로 처리합니다.
4. 대분류·중분류 확률을 세분류 점수에 결합합니다.

예:

```python
final_score = (
    fine_probability
    * mid_probability
    * top_probability
)
```

---

## 5. 추가 메타데이터 활용

가맹점명 외에 다음 정보가 있다면 별도 Encoder로 추가할 수 있습니다.

```text
주소
시·도
시·군·구
브랜드명
법인명
사업자 설명
과거 업종
결제 적요
홈페이지 설명
전화번호 지역
상권 정보
```

예:

```text
가맹점명: 행복
주소: 서울 종로구
사업자 설명: 의약품 및 의료용품 판매
```

가맹점명만으로는 업종 판단이 어렵지만 사업자 설명과 주소가 추가되면 분류가 쉬워질 수 있습니다.

메타데이터 구성 예:

```text
Text Encoder Feature
+
Byte Encoder Feature
+
Region Embedding
+
Brand Embedding
+
Numeric Feature
→ Fusion
```

숫자형 정보가 있다면 정규화한 뒤 작은 MLP로 처리할 수 있습니다.

---

## 6. 추천 학습 Loss

## 6.1 Class-Balanced Cross-Entropy

업종별 데이터 개수가 불균형한 경우 단순 Cross-Entropy보다 Class Weight를 적용합니다.

기본 형태:

```python
fine_loss = F.cross_entropy(
    fine_logits,
    fine_labels,
    weight=class_weights,
)
```

현재의 단순 역빈도 방식 외에 다음을 비교할 수 있습니다.

```text
일반 Cross-Entropy
Inverse Frequency Weight
Class-Balanced Cross-Entropy
Focal Loss
Class-Balanced Focal Loss
Logit Adjustment
```

첫 단계에서는 `Class-Balanced Cross-Entropy`를 권장합니다.

---

## 6.2 Supervised Contrastive Loss

같은 업종의 표현은 Embedding 공간에서 가깝게 만들고, 다른 업종은 멀리 떨어뜨립니다.

예:

```text
스타벅스 강남점
스타벅스 역삼점
커피전문점 스타벅스
→ 가까운 Embedding

행복약국
건강온누리약국
→ 가까운 Embedding

스타벅스 ↔ 행복약국
→ 먼 Embedding
```

전체 Loss 예:

```python
total_loss = (
    fine_loss
    + 0.3 * mid_loss
    + 0.2 * top_loss
    + 0.1 * contrastive_loss
)
```

특히 이름이 비슷하지만 업종이 다른 경우의 구분에 도움이 될 수 있습니다.

```text
행복카페
행복약국
행복마트
```

---

## 6.3 Label Smoothing

Label 오류가 있거나 유사 업종 경계가 모호한 경우 Label Smoothing을 적용할 수 있습니다.

```python
loss_function = nn.CrossEntropyLoss(
    label_smoothing=0.05
)
```

지나치게 큰 값은 Class 구분을 약하게 만들 수 있으므로 `0.02~0.1` 범위에서 검증합니다.

---

## 7. 도메인 추가 사전학습

한국어 범용 Encoder를 바로 Fine-tuning하는 것보다 Label이 없는 가맹점명으로 추가 사전학습하는 방식을 권장합니다.

사용 가능한 비라벨 데이터 예:

```text
가맹점 등록명
카드 결제 가맹점명
사업자 상호
브랜드명
정제 전 상호 원문
지점명
과거 상호명
사업자 설명
```

학습 흐름:

```text
범용 KLUE-RoBERTa
→ 가맹점명 데이터로 Masked Language Modeling
→ Merchant-RoBERTa
→ 업종 분류 Fine-tuning
```

가맹점명은 일반 문장과 형태가 다릅니다.

```text
GS25동탄센트럴점
씨유뉴강남타워
메가MGC커피
OOFASHION
행복온누리약국
```

따라서 모델 크기를 단순히 키우는 것보다 가맹점 도메인 데이터로 추가 사전학습하는 것이 중요한 성능 개선 요소가 될 수 있습니다.

### 추가 사전학습 데이터 정제

다음 항목을 관리해야 합니다.

- 중복 가맹점명 제거
- 개인정보 제거
- 지나치게 긴 문자열 제거
- 의미 없는 코드성 문자열 분리
- 인코딩 오류 제거
- 회사 형태 표현의 유지·제거 정책 결정
- 원문과 정규화 문장을 모두 활용할지 결정

---

## 8. Data Augmentation

가맹점명은 짧기 때문에 일반 문장용 Augmentation을 무리하게 적용하면 의미가 손상될 수 있습니다.

권장되는 Noise Augmentation:

```text
공백 삭제
공백 추가
특수문자 삽입
영문 대소문자 변경
숫자 지점명 추가
회사 형태 표현 추가
괄호와 하이픈 추가
일부 문자 오타
한글·영문 표기 혼용
```

예:

```text
원문: 메가MGC커피 강남점

증강:
메가MGC커피강남점
메가 MGC 커피 강남점
메가-MGC-커피 강남점
MEGAMGC커피강남점
메가MGC커피(강남점)
```

주의할 점:

- 업종을 나타내는 핵심 단어를 무작위로 삭제하지 않습니다.
- 서로 다른 업종으로 의미가 바뀌는 증강은 피합니다.
- 실제 운영 입력에서 관찰되는 Noise를 우선 사용합니다.

---

## 9. 미분류와 신규 업종 처리

현재 Raw Logit Threshold만 사용하면 Confidence 해석이 어렵습니다.

```python
if score < threshold:
    return unknown_index
```

Raw Logit은 확률이 아니며 모델 버전과 입력에 따라 Scale이 달라질 수 있습니다.

권장 흐름:

```text
Logit
→ Temperature Scaling
→ Calibrated Probability
→ Confidence Threshold
→ OOD Score
→ 자동 분류 또는 미분류
```

## 9.1 Temperature Scaling

검증 데이터로 Temperature 값을 학습합니다.

```python
calibrated_logits = (
    logits / temperature
)

probabilities = torch.softmax(
    calibrated_logits,
    dim=-1,
)
```

Temperature가 1보다 크면 지나치게 확신하는 Probability를 완화합니다.

## 9.2 Energy 기반 OOD Score

등록되지 않은 업종 또는 비정상 문자열을 탐지할 때 사용할 수 있습니다.

```python
energy = (
    -temperature
    * torch.logsumexp(
        logits / temperature,
        dim=-1,
    )
)
```

운영 정책 예:

```text
Confidence 높음
→ 자동 업종 확정

Confidence 중간
→ Top-3 후보 제공

Confidence 낮음
→ 미분류 처리

OOD Score 높음
→ 신규 업종 또는 이상 입력으로 검수
```

## 9.3 미분류 평가

미분류 기능은 다음 지표로 별도 평가합니다.

```text
미분류 Precision
미분류 Recall
Known Class Accuracy
OOD AUROC
False Acceptance Rate
False Rejection Rate
```

---

## 10. 데이터가 적은 경우의 대안

Class별 Label 데이터가 적다면 전체 Transformer Fine-tuning 외에 Sentence Encoder 기반 접근을 비교할 수 있습니다.

```text
한국어 Sentence Encoder
→ Contrastive Fine-tuning
→ Sentence Embedding
→ Linear Classifier 또는 Logistic Regression
```

장점:

- 적은 Label에서 빠르게 기준선 구축 가능
- 학습 비용이 비교적 낮음
- Embedding 기반 유사 가맹점 검색 가능
- 새로운 Class 추가 실험이 쉬움

적합한 상황:

```text
Class별 데이터가 매우 적음
GPU 자원이 제한적임
빠른 PoC가 필요함
유사 가맹점 검색 기능도 필요함
```

158개 Class에 충분한 학습 데이터가 있다면 일반 Encoder Fine-tuning을 우선 적용합니다.

---

## 11. 운영 경량화

권장 Hybrid 모델은 기존 CNN+MLP보다 무거울 수 있습니다.

먼저 정확도가 높은 Teacher 모델을 만든 후 작은 Student로 Distillation하는 방식을 권장합니다.

```text
Teacher
KLUE-RoBERTa
+ Byte Transformer
+ Hierarchical Heads

          ↓ Knowledge Distillation

Student
경량 Transformer
또는 작은 Hybrid Encoder
```

Distillation Loss 예:

```python
total_loss = (
    hard_label_loss
    + alpha * soft_target_loss
)
```

운영 최적화 후보:

```text
ONNX Runtime
INT8 Dynamic Quantization
Static Quantization
Torch Compile
Batch Inference
Tokenizer 병렬화
Embedding Cache
CPU Thread 조정
```

개발 순서:

```text
정확도가 높은 Teacher 확보
→ 오류 분석
→ Student Distillation
→ Quantization
→ 운영 Latency 검증
```

처음부터 지나치게 작은 모델만 개발하면 최종 성능 상한을 확인하기 어렵습니다.

---

## 12. 모델 후보별 추천

| 상황 | 추천 구조 |
|---|---|
| 가장 빠른 Transformer 기준선 | `KLUE-RoBERTa-base + Linear Head` |
| 최종 권장 구조 | `KLUE-RoBERTa + Byte Tiny Transformer + Hierarchical Heads` |
| 영문·다국어 비중이 높음 | `mDeBERTa-v3 + Byte Encoder` |
| 오타와 특수문자가 매우 많음 | Byte·Character 중심 Hybrid Encoder |
| Label 데이터가 적음 | Sentence Encoder + Contrastive Learning |
| CPU Latency가 중요함 | Hybrid Teacher → 경량 Student Distillation |
| 신규 업종 탐지가 필요함 | Calibrated Softmax + Energy OOD |
| 업종 계층 관계가 중요함 | Hierarchical Multi-task Classifier |
| 주소와 설명 데이터가 있음 | Text + Metadata Multimodal Fusion |

---

## 13. 단계별 개발 계획

## 13.1 1단계: Transformer 기준선 구축

같은 데이터 Split으로 다음 모델을 비교합니다.

```text
기존 SubwordCNN + WordNet
KLUE-RoBERTa-base
mDeBERTa-v3-base
Byte-only Tiny Transformer
```

첫 모델은 다음처럼 단순하게 구성합니다.

```text
Pretrained Encoder
→ Pooling
→ Dropout
→ Linear(업종 수)
```

이 단계의 목적:

- Transformer 단일 모델이 기존 Ensemble을 개선하는지 확인
- 학습 시간과 추론 속도 측정
- Fine-tuning Hyperparameter 기준 확보
- 데이터 Split 문제 확인

## 13.2 2단계: 도메인 추가 사전학습

```text
KLUE-RoBERTa
→ 비라벨 가맹점명 MLM
→ Merchant-RoBERTa
```

비교:

```text
원본 KLUE-RoBERTa
Merchant-RoBERTa
```

동일한 분류 데이터와 설정으로 비교합니다.

## 13.3 3단계: Byte Hybrid 추가

```text
Merchant-RoBERTa
+
Byte Tiny Transformer
+
Gated Fusion
```

별도의 Noise Test Set을 구성합니다.

```text
공백 제거
공백 추가
특수문자 삽입
대소문자 변경
한영 혼용
숫자 지점명
회사 표현
일부 문자 오타
```

## 13.4 4단계: 계층형 Head

```text
대분류 Head
중분류 Head
세분류 Head
```

계층 구조를 추가했을 때 세분류 Macro-F1과 희귀 Class Recall이 개선되는지 확인합니다.

## 13.5 5단계: Contrastive Loss

```text
Cross-Entropy
+
Supervised Contrastive Loss
```

유사 상호명과 신규 브랜드 Test Set을 중심으로 효과를 확인합니다.

## 13.6 6단계: Confidence와 OOD

```text
Temperature Scaling
Energy OOD
Threshold 결정
미분류 정책
```

## 13.7 7단계: 운영 최적화

```text
Student Distillation
ONNX 변환
INT8 Quantization
Latency Benchmark
```

---

## 14. Ablation Test

모든 기능을 한 번에 적용하면 어떤 요소가 성능을 개선했는지 판단하기 어렵습니다.

다음 순서로 하나씩 추가합니다.

| 실험 | 구성 |
|---|---|
| A | KLUE-RoBERTa 기준선 |
| B | A + 도메인 추가 사전학습 |
| C | B + Byte Encoder |
| D | C + Gated Fusion |
| E | D + 계층형 Loss |
| F | E + Contrastive Loss |
| G | F + Data Augmentation |
| H | G + Confidence Calibration |

각 실험에서 다음을 함께 기록합니다.

```text
Macro-F1
Weighted-F1
Top-1 Accuracy
희귀 Class F1
Noise Test 성능
Brand Holdout 성능
P95 Latency
모델 크기
```

---

## 15. 평가 방법

Accuracy만으로 모델을 선택하지 않습니다.

## 15.1 분류 성능

필수 지표:

```text
Macro-F1
Weighted-F1
Top-1 Accuracy
Top-3 Accuracy
Top-5 Accuracy
Class별 Precision
Class별 Recall
Class별 F1
Confusion Matrix
```

Class 불균형이 크다면 대표 지표로 `Macro-F1`을 권장합니다.

## 15.2 계층 성능

```text
대분류 Accuracy
중분류 Accuracy
세분류 Accuracy
계층 일관성 비율
대분류는 맞고 세분류는 틀린 비율
```

## 15.3 Confidence 성능

```text
ECE
Brier Score
Negative Log-Likelihood
미분류 Precision
미분류 Recall
OOD AUROC
```

## 15.4 운영 성능

```text
단건 P50 Latency
단건 P95 Latency
Batch 처리량
CPU 메모리
GPU 메모리
모델 파일 크기
초당 처리 건수
```

---

## 16. 데이터 Split 설계

모델 구조보다 데이터 Split이 평가 결과에 더 큰 영향을 줄 수 있습니다.

잘못된 예:

```text
Train: 스타벅스 강남역점
Test:  스타벅스 역삼점
```

모델이 업종을 이해한 것이 아니라 브랜드명을 외웠을 가능성이 큽니다.

권장 Split:

```text
브랜드 기준 Group Split
정규화된 대표 상호 기준 Group Split
시간 기준 Split
신규 브랜드 Holdout
희귀 업종 Holdout
Noise 전용 Test Set
OOD 전용 Test Set
```

최종 평가 세트 예:

| Test Set | 목적 |
|---|---|
| Random Test | 전체 평균 성능 |
| Brand Holdout | 처음 보는 브랜드 일반화 |
| Time Holdout | 미래 데이터 성능 |
| Noise Test | 오타·특수문자 강건성 |
| Rare Class Test | 소수 업종 성능 |
| OOD Test | 미등록 업종 탐지 |

---

## 17. 추천 초기 Hyperparameter

## 17.1 KLUE-RoBERTa 기준선

| 항목 | 초기값 |
|---|---:|
| Max Length | 32 또는 64 |
| Batch Size | 16~64 |
| Learning Rate | `2e-5` |
| Epoch | 3~10 |
| Weight Decay | `0.01` |
| Warmup Ratio | `0.05~0.1` |
| Dropout | `0.1~0.2` |
| Gradient Clipping | `1.0` |

가맹점명은 짧기 때문에 Max Length를 지나치게 크게 설정할 필요가 없습니다.

## 17.2 Byte Encoder

| 항목 | 초기값 |
|---|---:|
| Max Byte Length | 96 |
| Byte Embedding | 64 |
| Hidden Size | 128 |
| CNN Layer | 2 |
| Transformer Layer | 2 |
| Attention Head | 4 |
| Dropout | 0.1 |
| Pooling | Attention Pooling |

## 17.3 Fusion

| 항목 | 초기값 |
|---|---:|
| 공통 Projection 크기 | 256 |
| Fusion | Gated Sum |
| Fusion Dropout | 0.1 |
| Final Hidden | 256 |

## 17.4 Loss Weight

```python
fine_weight = 1.0
mid_weight = 0.3
top_weight = 0.2
contrastive_weight = 0.1
```

초기값일 뿐이며 검증 데이터로 조정해야 합니다.

---

## 18. 권장 PyTorch 모델 구조 예시

아래 코드는 전체 구현이 아니라 구조를 설명하기 위한 예시입니다.

```python
import torch
import torch.nn as nn
from transformers import AutoModel


class MerchantHybridEncoder(nn.Module):
    def __init__(
        self,
        pretrained_name,
        num_top_classes,
        num_mid_classes,
        num_fine_classes,
        byte_vocab_size=260,
        byte_hidden_size=128,
        fusion_size=256,
    ):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(
            pretrained_name
        )

        text_hidden_size = (
            self.text_encoder.config.hidden_size
        )

        self.byte_embedding = nn.Embedding(
            byte_vocab_size,
            byte_hidden_size,
            padding_idx=0,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=byte_hidden_size,
            nhead=4,
            dim_feedforward=byte_hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )

        self.byte_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.text_projection = nn.Linear(
            text_hidden_size,
            fusion_size,
        )

        self.byte_projection = nn.Linear(
            byte_hidden_size,
            fusion_size,
        )

        self.gate = nn.Sequential(
            nn.Linear(
                fusion_size * 2,
                fusion_size,
            ),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(0.1)

        self.top_classifier = nn.Linear(
            fusion_size,
            num_top_classes,
        )

        self.mid_classifier = nn.Linear(
            fusion_size,
            num_mid_classes,
        )

        self.fine_classifier = nn.Linear(
            fusion_size,
            num_fine_classes,
        )

    @staticmethod
    def masked_mean(
        hidden_state,
        attention_mask,
    ):
        mask = attention_mask.unsqueeze(-1)
        mask = mask.to(hidden_state.dtype)

        summed = (
            hidden_state * mask
        ).sum(dim=1)

        count = mask.sum(dim=1).clamp(min=1.0)

        return summed / count

    def forward(
        self,
        input_ids,
        attention_mask,
        byte_ids,
        byte_attention_mask,
    ):
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        text_feature = self.masked_mean(
            text_output.last_hidden_state,
            attention_mask,
        )

        byte_feature = self.byte_embedding(
            byte_ids
        )

        byte_padding_mask = (
            byte_attention_mask == 0
        )

        byte_feature = self.byte_encoder(
            byte_feature,
            src_key_padding_mask=byte_padding_mask,
        )

        byte_feature = self.masked_mean(
            byte_feature,
            byte_attention_mask,
        )

        text_feature = self.text_projection(
            text_feature
        )

        byte_feature = self.byte_projection(
            byte_feature
        )

        gate_input = torch.cat(
            [
                text_feature,
                byte_feature,
            ],
            dim=-1,
        )

        gate = self.gate(
            gate_input
        )

        fused_feature = (
            gate * text_feature
            + (1.0 - gate) * byte_feature
        )

        fused_feature = self.dropout(
            fused_feature
        )

        return {
            "embedding": fused_feature,
            "top_logits": self.top_classifier(
                fused_feature
            ),
            "mid_logits": self.mid_classifier(
                fused_feature
            ),
            "fine_logits": self.fine_classifier(
                fused_feature
            ),
        }
```

실제 구현에서는 Byte Sequence 길이 축소용 CNN, Attention Pooling, Positional Encoding, Contrastive Projection Head 등을 추가할 수 있습니다.

---

## 19. 개발 우선순위

개발 복잡도와 기대 효과를 고려한 우선순위는 다음과 같습니다.

### 우선순위 1

```text
KLUE-RoBERTa-base 단일 분류 모델
```

목적:

- 기존 모델 대비 Transformer 기준 성능 확인
- 데이터 Split 검증
- 빠른 PoC

### 우선순위 2

```text
도메인 추가 사전학습
```

목적:

- 가맹점 표현과 브랜드 패턴 학습
- 신규 브랜드 일반화 개선

### 우선순위 3

```text
Byte Encoder + Gated Fusion
```

목적:

- 오타와 비정형 입력 대응
- 영문·한글 혼합 처리
- Tokenizer 의존성 완화

### 우선순위 4

```text
계층형 Multi-task Head
```

목적:

- 대·중·세분류 관계 활용
- 희귀 세분류 학습 보완

### 우선순위 5

```text
Confidence Calibration + OOD
```

목적:

- 미분류 정책 안정화
- 신규 업종과 이상 입력 검출

### 우선순위 6

```text
Distillation + Quantization
```

목적:

- CPU 추론 속도 개선
- 운영 비용 절감

---

## 20. 최종 권장안

가장 현실적인 개발 순서는 다음과 같습니다.

```text
1. KLUE-RoBERTa-base 단일 모델로 기준선 구축
2. 비라벨 가맹점명으로 Domain-Adaptive Pretraining
3. UTF-8 Byte Tiny Transformer 추가
4. Feature 수준 Gated Fusion 적용
5. 대·중·세분류 Multi-task Head 적용
6. Class-Balanced Cross-Entropy 적용
7. Supervised Contrastive Loss 실험
8. Temperature Scaling과 Energy OOD 적용
9. 최종 Teacher를 경량 Student로 Distillation
10. ONNX 및 INT8 최적화
```

최종 목표 구조:

```text
Merchant-RoBERTa
       +
Byte Tiny Transformer
       +
Gated Feature Fusion
       +
Hierarchical Classification
       +
Calibrated OOD Detection
```

핵심 방향은 다음과 같습니다.

```text
기존 모델의 문자 강건성 유지
+
사전학습 Transformer의 문맥 이해 추가
+
업종 계층 구조 활용
+
신규 브랜드와 비정형 입력 대응
+
운영 가능한 Confidence와 미분류 체계 구축
```

단순히 `SubwordCNN`을 BERT로 교체하는 것보다, 한국어 Encoder와 Byte Encoder를 Feature 수준에서 결합하는 Hybrid 구조가 가맹점명 분류 업무에 더 적합한 신규 모델 후보입니다.

---

## 21. 참고할 연구 방향

신규 모델 설계 시 다음 연구 주제를 참고할 수 있습니다.

- KLUE 및 한국어 RoBERTa
- DeBERTa-v3
- CANINE
- ByT5
- Hierarchical Text Classification
- Domain-Adaptive Pretraining
- Task-Adaptive Pretraining
- Supervised Contrastive Learning
- Class-Balanced Loss
- Temperature Scaling
- Energy-based OOD Detection
- SetFit
- Knowledge Distillation
