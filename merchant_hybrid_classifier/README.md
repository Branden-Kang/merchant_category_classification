# MerchantHybridEncoder Colab Demo

이 프로젝트는 다음 신규 구조를 샘플 데이터로 테스트합니다.

```text
KLUE-RoBERTa
+
UTF-8 Byte Tiny Transformer
+
Gated Feature Fusion
+
대분류·중분류·세분류 Multi-task Head
```

## 샘플 업종

| Fine Label | 업종 | 중분류 | 대분류 |
|---:|---|---|---|
| 0 | 카페 | 카페 | 음식 |
| 1 | 한식 | 한식 | 음식 |
| 2 | 편의점 | 편의점 | 소매 |
| 3 | 약국 | 약국 | 의료 |
| 4 | 의류 | 의류 | 소매 |

## Colab 권장 설정

Colab 메뉴에서 다음을 선택합니다.

```text
런타임
→ 런타임 유형 변경
→ T4 GPU
```

## 설치

```bash
pip install -r requirements.txt
```

## 전체 데모

다음 명령은 샘플 데이터 생성, 학습, 테스트, 추론을 모두 실행합니다.

```bash
python main.py \
  --mode demo \
  --device auto \
  --epochs 3 \
  --batch-size 8
```

처음 실행할 때 `klue/roberta-base` Tokenizer와 모델이 다운로드됩니다.

## 빠른 연결 확인

텍스트 Encoder를 고정하면 학습이 더 가벼워집니다.

```bash
python main.py \
  --mode demo \
  --device auto \
  --epochs 2 \
  --batch-size 8 \
  --freeze-text-encoder
```

이 모드는 코드 연결 확인에는 적합하지만, 전체 Fine-tuning보다 성능이 낮을 수 있습니다.

## 단계별 실행

### 샘플 데이터 생성

```bash
python sample_data.py
```

### 학습

```bash
python main.py \
  --mode train \
  --device auto \
  --epochs 3
```

### 테스트

```bash
python main.py \
  --mode test \
  --device auto
```

### 추론

```bash
python main.py \
  --mode inference \
  --device auto
```

## 주요 결과

```text
model/best_model.pt
model/tokenizer/
result/test_result.csv
result/inference_result.csv
```

## Gate 값

결과의 `text_gate_mean`은 Gated Fusion에서 한국어 Text Encoder가 차지한 평균 비중입니다.

```text
1에 가까움: Text Encoder 비중이 큼
0에 가까움: Byte Encoder 비중이 큼
```

샘플 데이터가 매우 작기 때문에 Gate 값 자체를 운영 기준으로 사용하면 안 됩니다.

## 실제 데이터 적용

학습 CSV:

```csv
id,text,top_label,mid_label,fine_label
0,가맹점명,0,0,0
```

업종 메타 CSV:

```csv
fine_label,fine_name,mid_label,mid_name,top_label,top_name,store_category_cd
0,카페,0,카페,0,음식,C000
```

추론 CSV:

```csv
id,text
0,가맹점명
```

실제 158개 업종을 사용할 때는 메타 파일과 Label을 실제 체계에 맞게 변경하면 Class 수가 자동 계산됩니다.

## 주의사항

- 이 프로젝트는 구조 검증용 샘플입니다.
- 샘플 데이터는 업종 단어가 직접 포함되어 실제 데이터보다 쉽습니다.
- 운영 모델 평가는 브랜드 Group Split, 시간 Split, Noise Test가 필요합니다.
- 현재 Confidence는 Softmax 결과이며 별도의 Temperature Scaling은 포함하지 않았습니다.
