# 가맹점 업종 분류 프로젝트 코드 설명

이 프로젝트는 **가맹점명 또는 가맹점 관련 텍스트를 입력받아 업종을 분류하는 PyTorch 기반 앙상블 모델**입니다.

동일한 텍스트를 두 가지 방식으로 분석합니다.

```text
가맹점명
   │
   ├─ 문자 단위 입력 ── SubwordCNN ── 업종별 점수
   │
   └─ 단어 단위 입력 ── WordNet ───── 업종별 점수
                                      │
                               두 결과 결합
                                      │
                                  Ensemble
                                      │
                                  최종 업종
```

문자 단위 모델은 가맹점명의 철자와 부분 문자열 패턴을 분석하고, 단어 단위 모델은 단어 조합과 의미적 특징을 분석합니다. 마지막으로 두 모델의 출력을 결합해 최종 업종을 결정합니다.

샘플 프로젝트는 빠른 실행을 위해 다음 5개 업종을 사용합니다.

| Label | 업종 |
|---:|---|
| 0 | 카페 |
| 1 | 한식 |
| 2 | 편의점 |
| 3 | 약국 |
| 4 | 의류 |

---

## 1. 전체 파일 구조

```text
store_classifier_demo/
├── main.py
├── model.py
├── data_preprocess.py
├── utils.py
├── tools.py
├── logger.py
├── sample_data.py
├── requirements.txt
│
├── data/
│   ├── train_data.csv
│   ├── valid_data.csv
│   ├── test_data.csv
│   └── pred_data.csv
│
├── meta_data/
│   └── store_category_code_meta.csv
│
├── model/
├── result/
└── log/
```

각 파일의 역할은 다음과 같습니다.

| 파일 | 역할 |
|---|---|
| `main.py` | 전체 실행 흐름과 학습·테스트·추론 모드 제어 |
| `data_preprocess.py` | 텍스트 정제, 토큰화, Vocabulary 생성, DataLoader 생성 |
| `model.py` | `SubwordCNN`, `WordNet`, `Ensemble` 모델 정의 |
| `utils.py` | 학습, 검증, 정확도 계산, Checkpoint 저장 |
| `tools.py` | 추론용 데이터 생성, 결과 정리, 오분류 분석 |
| `sample_data.py` | 실행 확인용 샘플 CSV 및 업종 메타데이터 생성 |
| `logger.py` | 콘솔과 로그 파일에 실행 기록 저장 |
| `requirements.txt` | 프로젝트 실행에 필요한 라이브러리 목록 |

---

## 2. 전체 실행 흐름

기본 실행 명령은 다음과 같습니다.

```bash
python main.py
```

`main.py`의 기본 모드는 `demo`입니다.

```python
parser.add_argument(
    "--mode",
    default="demo",
    choices=["demo", "train", "test", "inference"],
)
```

`demo` 모드에서는 다음 과정이 순서대로 실행됩니다.

```text
1. 샘플 데이터 생성
2. Vocabulary 생성
3. 모델 학습
4. 검증
5. 가장 낮은 검증 Loss의 모델 저장
6. 테스트 데이터 평가
7. 예측 데이터 추론
8. 결과 CSV 생성
```

실제 코드는 다음과 같은 흐름을 가집니다.

```python
def run_demo(args, device, mylogger):
    create_sample_data(".")

    train_args = deepcopy(args)
    train_args.mode = "train"
    train_model(train_args, device, mylogger)

    test_args = deepcopy(args)
    test_args.mode = "test"
    evaluate_or_infer(test_args, device, mylogger)

    inference_args = deepcopy(args)
    inference_args.mode = "inference"
    inference_args.result = "inference"
    evaluate_or_infer(inference_args, device, mylogger)
```

`deepcopy()`는 기존 `args` 객체를 직접 변경하지 않고 학습, 테스트, 추론용 옵션을 각각 독립적으로 만들기 위해 사용합니다.

---

## 3. `sample_data.py`

`sample_data.py`는 전체 프로젝트가 정상적으로 연결되어 있는지 확인하기 위한 샘플 데이터를 생성합니다.

### 3.1 업종 메타정보

```python
CATEGORIES = [
    (0, "C000", "카페", "음식", "카페", "일반"),
    (1, "C001", "한식", "음식", "한식", "일반"),
    (2, "C002", "편의점", "소매", "편의점", "일반"),
    (3, "C003", "약국", "의료", "약국", "일반"),
    (4, "C004", "의류", "소매", "의류", "일반"),
]
```

각 값의 의미는 다음과 같습니다.

| 위치 | 컬럼 |
|---:|---|
| 1 | `store_category_index` |
| 2 | `store_category_cd` |
| 3 | `store_category_nm` |
| 4 | `store_category_top_nm` |
| 5 | `store_category_mid_nm` |
| 6 | `asset_management_nm` |

이 정보는 다음 파일로 저장됩니다.

```text
meta_data/store_category_code_meta.csv
```

### 3.2 학습·검증·테스트 데이터

학습 데이터는 업종마다 10개씩 생성됩니다.

```python
TRAIN_TEXTS = {
    0: [
        "봄날 카페 아메리카노",
        "커피하우스 라떼",
        ...
    ],
    1: [
        "한식당 고향집",
        "김치찌개 식당",
        ...
    ],
}
```

`_to_frame()` 함수는 업종별 문장들을 다음 구조의 DataFrame으로 바꿉니다.

```python
def _to_frame(grouped_texts):
    rows = []
    row_id = 0

    for label, texts in grouped_texts.items():
        for text in texts:
            rows.append(
                {
                    "id": row_id,
                    "text": text,
                    "label": label,
                }
            )
            row_id += 1

    return pd.DataFrame(rows)
```

생성되는 CSV 형식은 다음과 같습니다.

```csv
id,text,label
0,봄날 카페 아메리카노,0
1,커피하우스 라떼,0
2,한식당 고향집,1
```

기본 입력 컬럼의 의미는 다음과 같습니다.

| 컬럼 | 의미 |
|---|---|
| `id` | 데이터 식별자 |
| `text` | 가맹점명 또는 분류할 문장 |
| `label` | 정답 업종 Index |

### 3.3 추론 데이터

추론 데이터에는 정답 Label이 없으므로 `text` 컬럼만 생성합니다.

```python
PRED_TEXTS = [
    "아메리카노 커피 카페",
    "김치찌개 한식당",
    "야간 편의점 도시락",
    "건강 처방전 약국",
    "여성 패션 의류 매장",
]
```

```csv
text
아메리카노 커피 카페
김치찌개 한식당
야간 편의점 도시락
```

추론 단계에서 `tools.data_inferer_df()`가 `id`와 임시 `label`을 추가합니다.

---

## 4. `data_preprocess.py`

`data_preprocess.py`는 원문 텍스트를 모델이 처리할 수 있는 숫자 Tensor로 변환합니다.

전체 전처리 흐름은 다음과 같습니다.

```text
원문 텍스트
→ 문자열 정규화
→ 문자 토큰과 단어 토큰 생성
→ Vocabulary Index 변환
→ 고정 길이 Padding
→ PyTorch Tensor
```

### 4.1 회사 형태 표현 제거

```python
COMPANY_PATTERN = re.compile(
    r"\(주\)주식회사|"
    r"유한회사|"
    r"사단법인|"
    r"재단법인|"
    r"\(주\)|"
    r"\(유\)|"
    r"주식회사"
)
```

다음과 같은 회사 형태 표현을 제거합니다.

```text
(주)
(유)
주식회사
유한회사
사단법인
재단법인
```

### 4.2 텍스트 정규화

```python
def _normalize_text(x_raw) -> str:
    text = COMPANY_PATTERN.sub(" ", str(x_raw))
    text = re.sub(r"[^\w\s]+", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

처리 과정은 다음과 같습니다.

1. 입력값을 문자열로 변환합니다.
2. 회사 형태 표현을 제거합니다.
3. 영문을 소문자로 변환합니다.
4. 문자, 숫자, 밑줄, 공백을 제외한 특수문자를 제거합니다.
5. 여러 공백을 하나로 합칩니다.
6. 문자열 양끝 공백을 제거합니다.

예를 들어 다음 텍스트는

```text
(주)행복-커피!!!
```

대략 다음과 같이 정규화됩니다.

```text
행복 커피
```

### 4.3 문자 단위 토큰화

```python
def custom_tokenizer(
    x_raw,
    seq_max_len=SUBWORD_SEQ_MAX_LEN,
):
    text = _normalize_text(x_raw).replace(" ", "")
    return list(text)[:seq_max_len]
```

예를 들어

```text
봄날 카페
```

는 다음 문자 목록으로 변환됩니다.

```python
["봄", "날", "카", "페"]
```

문자 단위 분석은 다음과 같은 상황에서 유용합니다.

- 띄어쓰기 차이
- 붙여 쓴 가맹점명
- Vocabulary에 없는 신규 단어
- 비슷한 철자 또는 접미 표현
- 짧은 상호명

예를 들어 다음 표현은 문자 패턴이 상당 부분 겹칩니다.

```text
행복약국
행복 약국
행복약국점
```

문자 토큰은 `SubwordCNN`에 전달됩니다.

### 4.4 단어 단위 토큰화

```python
def custom_tokenizer_word(
    x_raw,
    seq_max_len=WORD_SEQ_MAX_LEN,
):
    text = _normalize_text(x_raw)

    if _mecab is not None:
        tokens = _mecab.morphs(text)
    else:
        tokens = text.split()
        if not tokens and text:
            tokens = [text]

    return tokens[:seq_max_len]
```

Mecab을 사용할 수 있으면 형태소 단위로 분석합니다.

```text
김치찌개 맛집
```

예시 결과:

```python
["김치찌개", "맛집"]
```

Mecab을 사용할 수 없는 환경에서는 공백 단위 토큰화를 사용합니다.

```python
tokens = text.split()
```

단어 토큰은 `WordNet`에 전달됩니다.

### 4.5 `SimpleVocab`

```python
class SimpleVocab:
```

`SimpleVocab`은 학습 데이터에 등장한 토큰을 정수 Index로 변환하는 Vocabulary입니다.

예를 들어 다음과 같은 문자 Vocabulary가 만들어졌다고 가정합니다.

```python
{
    "<unk>": 0,
    "<pad>": 1,
    "카": 2,
    "페": 3,
    "약": 4,
    "국": 5,
}
```

그러면

```text
카페
```

는 다음 숫자로 변환됩니다.

```python
[2, 3]
```

Vocabulary에 없는 토큰은 `<unk>` Index로 변환합니다.

```python
def __getitem__(self, token: str) -> int:
    return self.stoi.get(token, self.unk_index)
```

특수 토큰의 의미는 다음과 같습니다.

| 토큰 | Index | 의미 |
|---|---:|---|
| `<unk>` | 0 | Vocabulary에 없는 토큰 |
| `<pad>` | 1 | 입력 길이를 맞추기 위한 Padding |

`vocab.itos`는 Index에서 토큰으로 변환할 때 사용하고, `vocab.stoi`는 토큰에서 Index로 변환할 때 사용합니다.

### 4.6 Vocabulary 생성

```python
def make_vocab(train_data, args):
    counter_subword = Counter()
    counter_word = Counter()

    text_column = (
        "text"
        if "text" in train_data.columns
        else train_data.columns[1]
    )

    for text in train_data[text_column].astype(str):
        counter_subword.update(
            custom_tokenizer(text, args.seq_len1)
        )
        counter_word.update(
            custom_tokenizer_word(text, args.seq_len2)
        )

    vocab1 = SimpleVocab(
        counter_subword,
        min_freq=args.min_freq_subword,
    )
    vocab2 = SimpleVocab(
        counter_word,
        min_freq=args.min_freq_word,
    )

    return vocab1, vocab2
```

`vocab1`은 문자 단위 Vocabulary이고 `vocab2`는 단어 단위 Vocabulary입니다.

`min_freq`는 학습 데이터에서 최소 몇 번 이상 등장한 토큰을 Vocabulary에 넣을지 결정합니다.

```text
--min-freq-subword
--min-freq-word
```

샘플 프로젝트의 기본값은 모두 `1`입니다.

### 4.7 Padding

딥러닝 모델은 하나의 Batch 안에서 모든 입력 길이가 같아야 합니다.

```python
def padding_sentences(
    sentence,
    seq_max_len,
    pad_idx=1,
):
    sentence = list(sentence)[:seq_max_len]

    if len(sentence) < seq_max_len:
        sentence.extend(
            [pad_idx] * (seq_max_len - len(sentence))
        )

    return sentence
```

예를 들어 최대 길이가 8이고 입력 Index가 다음과 같다면

```python
[2, 5, 7]
```

Padding 후 결과는 다음과 같습니다.

```python
[2, 5, 7, 1, 1, 1, 1, 1]
```

입력이 최대 길이보다 길면 뒤쪽 토큰을 자릅니다.

문자 입력과 단어 입력은 서로 다른 길이를 사용합니다.

| 입력 | 옵션 | 기본값 |
|---|---|---:|
| 문자 입력 | `seq_len1` | 30 |
| 단어 입력 | `seq_len2` | 8 |

### 4.8 `DatasetMapper`

```python
class DatasetMapper(Dataset):
```

Pandas DataFrame을 PyTorch Dataset 형식으로 연결합니다.

`id`, `text`, `label` 컬럼이 있으면 컬럼명으로 접근합니다.

```python
if {"id", "text", "label"}.issubset(data_frame.columns):
    self.ids = data_frame["id"].reset_index(drop=True)
    self.x = data_frame["text"].reset_index(drop=True)
    self.y = data_frame["label"].reset_index(drop=True)
```

컬럼명이 다르면 앞의 세 컬럼을 순서대로 사용합니다.

```python
self.ids = data_frame.iloc[:, 0]
self.x = data_frame.iloc[:, 1]
self.y = data_frame.iloc[:, 2]
```

각 데이터는 다음 형식으로 반환됩니다.

```python
(
    id,
    text,
    label,
)
```

### 4.9 `DataGenerator`

```python
class DataGenerator:
```

`DataGenerator`는 다음 작업을 수행합니다.

```text
문자 토큰 생성
→ 문자 Vocabulary Index 변환
→ 문자 Padding
→ 단어 토큰 생성
→ 단어 Vocabulary Index 변환
→ 단어 Padding
→ Batch Tensor 생성
```

한 Batch의 반환값은 다음과 같습니다.

```python
(
    id_list,
    text_list1,
    text_list2,
    label_list,
)
```

Tensor 형태는 다음과 같습니다.

| Tensor | 형태 |
|---|---|
| `id_list` | `[batch_size]` |
| `text_list1` | `[batch_size, seq_len1]` |
| `text_list2` | `[batch_size, seq_len2]` |
| `label_list` | `[batch_size]` |

기본 Batch 크기가 8일 때는 다음과 같습니다.

```text
id_list     : [8]
text_list1  : [8, 30]
text_list2  : [8, 8]
label_list  : [8]
```

`collate_batch()`에서 만들어지는 Tensor는 `torch.long` 자료형을 사용합니다.

```python
torch.tensor(
    text_list1,
    dtype=torch.long,
)
```

Embedding Layer의 입력과 `CrossEntropyLoss`의 Label은 정수형 Index여야 하므로 `torch.long`이 필요합니다.

---

## 5. `model.py`

모델은 다음 세 부분으로 구성됩니다.

```text
SubwordCNN
WordNet
Ensemble
```

### 5.1 Embedding Layer 생성

```python
def build_embedding(
    vocab,
    embedding_size,
    padding_idx=PADDING_IDX,
):
```

Vocabulary에 사전 학습 벡터가 없으면 새로운 Embedding Layer를 만듭니다.

```python
return nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=embedding_size,
    padding_idx=padding_idx,
)
```

Vocabulary에 사전 학습 벡터가 있으면 해당 값을 초기값으로 사용합니다.

```python
return nn.Embedding.from_pretrained(
    pretrained_vectors,
    freeze=False,
    padding_idx=padding_idx,
)
```

`freeze=False`이므로 학습 과정에서 Embedding도 함께 업데이트됩니다.

Embedding은 토큰 Index를 실수 벡터로 변환합니다.

예를 들어 입력 형태가

```text
[8, 30]
```

이고 Embedding 크기가 32이면 출력은 다음 형태입니다.

```text
[8, 30, 32]
```

### 5.2 `SubwordCNN`

```python
class SubwordCNN(nn.Module):
```

문자 단위 입력에서 연속된 문자 패턴을 추출하는 1차원 CNN 모델입니다.

사용하는 Kernel 크기는 다음과 같습니다.

```python
self.kernels = (2, 3, 4, 5)
```

각 Kernel은 서로 다른 길이의 문자 패턴을 봅니다.

| Kernel | 분석 패턴 |
|---:|---|
| 2 | 2글자 패턴 |
| 3 | 3글자 패턴 |
| 4 | 4글자 패턴 |
| 5 | 5글자 패턴 |

예를 들어 `행복약국`이라는 문자열에서는 다음과 같은 패턴을 추출할 수 있습니다.

```text
행복
복약
약국
행복약
복약국
행복약국
```

CNN Layer는 다음 구조입니다.

```python
nn.Sequential(
    nn.Conv1d(
        in_channels=self.embedding_size,
        out_channels=self.out_size,
        kernel_size=kernel_size,
    ),
    nn.ReLU(),
    nn.MaxPool1d(
        kernel_size=pool_kernel_size,
        stride=self.stride,
    ),
)
```

#### 입력 차원 변경

Embedding 후 Tensor 형태는 다음과 같습니다.

```text
[batch, sequence, embedding]
```

`Conv1d`는 Channel 차원이 두 번째 위치에 있어야 합니다.

```python
x = x.permute(0, 2, 1)
```

예:

```text
변경 전: [8, 30, 32]
변경 후: [8, 32, 30]
```

여기에서 32는 Embedding Channel 수이고 30은 문자 Sequence 길이입니다.

#### 여러 Kernel의 결과 결합

```python
conv_outputs = [
    layer(x)
    for layer in self.layers
]
```

Kernel 2, 3, 4, 5의 결과를 각각 계산합니다.

각 결과를 Batch 단위로 펼칩니다.

```python
flattened = [
    output.flatten(start_dim=1)
    for output in conv_outputs
]
```

모든 CNN 특징을 하나로 합칩니다.

```python
merged = torch.cat(
    flattened,
    dim=1,
)
```

마지막으로 완전 연결 Layer를 통과합니다.

```python
out = F.relu(self.fc1(merged))
out = self.dropout(out)
return self.fc2(out)
```

최종 출력 형태는 다음과 같습니다.

```text
[batch_size, num_classes]
```

샘플 프로젝트에서는 다음과 같습니다.

```text
[8, 5]
```

### 5.3 CNN 출력 길이 계산

`Conv1d` 출력 길이는 다음 함수에서 계산합니다.

```python
@staticmethod
def conv_output_size(
    sequence_length,
    kernel_size,
):
    output_size = sequence_length - kernel_size + 1
```

`MaxPool1d`까지 통과한 길이는 다음과 같이 계산합니다.

```python
output_size = math.floor(
    (conv_size - pool_kernel)
    / self.stride
    + 1
)
```

이 계산 결과를 이용해 `fc1`의 입력 크기를 미리 결정합니다.

```python
fc_input_size = (
    sum(
        self.conv_pool_output_size(
            self.seq_len,
            kernel_size,
        )
        for kernel_size in self.kernels
    )
    * self.out_size
)
```

따라서 `seq_len1`, Kernel 크기, Stride가 바뀌더라도 완전 연결 Layer의 입력 크기가 자동으로 맞춰집니다.

### 5.4 `WordNet`

```python
class WordNet(nn.Module):
```

단어 단위 입력 전체를 펼친 후 MLP로 분류합니다.

입력 형태:

```text
[batch_size, seq_len2]
```

Embedding 후:

```text
[batch_size, seq_len2, embedding_size2]
```

다음 코드로 단어 길이와 Embedding 크기를 하나의 차원으로 합칩니다.

```python
x = x.flatten(start_dim=1)
```

기본값 기준:

```text
[8, 8, 32]
→ [8, 256]
```

MLP 구조는 다음과 같습니다.

```python
self.linear_stack = nn.Sequential(
    nn.Linear(
        self.seq_len * self.embedding_size,
        512,
    ),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(params.dropout),
    nn.Linear(256, self.num_classes),
)
```

구성 요소의 역할은 다음과 같습니다.

| Layer | 역할 |
|---|---|
| `Linear` | 입력 특징을 새로운 차원으로 변환 |
| `ReLU` | 비선형성 추가 |
| `BatchNorm1d` | Batch 단위 활성값 정규화 |
| `Dropout` | 일부 값을 무작위로 제거하여 과적합 완화 |
| 마지막 `Linear` | 업종별 Logit 출력 |

최종 출력 형태는 다음과 같습니다.

```text
[batch_size, num_classes]
```

### 5.5 `Ensemble`

```python
class Ensemble(nn.Module):
```

두 모델의 출력을 결합합니다.

```python
subword_logits = self.model_a(subword)
word_logits = self.model_b(word)
```

샘플의 Class가 5개라면 두 출력은 다음 형태입니다.

```text
SubwordCNN: [batch, 5]
WordNet:    [batch, 5]
```

두 결과를 이어 붙입니다.

```python
merged = torch.cat(
    (
        subword_logits,
        word_logits,
    ),
    dim=1,
)
```

결합 후 형태:

```text
[batch, 10]
```

마지막 분류 Layer가 이를 다시 5개 업종 점수로 변환합니다.

```python
self.classifier = nn.Linear(
    num_classes * 2,
    num_classes,
)
```

최종 출력:

```text
[batch, 5]
```

이 값은 확률이 아니라 **Logit**, 즉 업종별 원시 점수입니다.

`modelA`, `modelB` Property는 기존 코드와의 호환성을 위해 유지되어 있습니다.

```python
@property
def modelA(self):
    return self.model_a
```

---

## 6. `utils.py`

`utils.py`는 모델 학습, 검증, 정확도 계산, Checkpoint 저장을 담당합니다.

### 6.1 `AverageMeter`

```python
class AverageMeter:
```

현재 값과 누적 평균을 저장합니다.

```python
def update(self, val, n=1):
    self.val = float(val)
    self.sum += float(val) * n
    self.count += n
    self.avg = (
        self.sum / self.count
        if self.count
        else 0.0
    )
```

예를 들어

```text
Batch 1 Loss = 1.5, 샘플 8개
Batch 2 Loss = 1.0, 샘플 8개
```

라면 누적 평균은 다음과 같습니다.

```text
(1.5 × 8 + 1.0 × 8) / 16 = 1.25
```

### 6.2 `ProgressMeter`

```python
class ProgressMeter:
```

현재 Batch 진행률과 Loss, 정확도, 처리 시간을 출력합니다.

예:

```text
Epoch: [1][1/7] Time 0.032 Loss 1.5421 Acc@1 25.00 Acc@5 100.00
```

### 6.3 정확도 계산

```python
def accuracy(
    output,
    target,
    topk=(1,),
):
```

Top-k 정확도를 계산합니다.

모델 출력의 점수가 높은 Class Index를 가져옵니다.

```python
_, pred = output.topk(
    max_k,
    dim=1,
    largest=True,
    sorted=True,
)
```

정답 Label과 비교합니다.

```python
correct = pred.eq(
    target.view(1, -1).expand_as(pred)
)
```

#### Top-1 정확도

가장 높은 점수의 Class 하나가 정답인지 확인합니다.

```text
모델 Logit: [0.2, 1.7, 0.4, -0.1, 0.3]
예측 Class: 1
```

정답 Label이 1이면 Top-1 예측 성공입니다.

#### Top-5 정확도

점수가 높은 5개 Class 안에 정답이 포함되는지 확인합니다.

샘플 프로젝트는 전체 Class가 5개이므로 Top-5에는 모든 Class가 포함됩니다. 따라서 샘플 환경에서 Top-5는 모델 성능을 구분하는 지표로서 의미가 제한적입니다.

코드는 전체 Class 수가 5보다 적은 경우도 처리할 수 있도록 실제 Class 수와 요청한 Top-k 중 작은 값을 사용합니다.

```python
max_k = min(
    max(topk),
    output.size(1),
)
```

### 6.4 Learning Rate 조정

```python
def adjust_learning_rate(
    optimizer,
    epoch,
    args,
):
```

일정 Epoch마다 Learning Rate를 감소시킵니다.

일반적인 형태는 다음과 같습니다.

```text
초기 Learning Rate × 0.1^(epoch 구간)
```

Learning Rate가 너무 크면 학습이 불안정할 수 있고, 학습 후반에는 더 작은 값을 사용해 Parameter를 세밀하게 조정할 수 있습니다.

### 6.5 학습 함수

```python
def train(
    train_loader,
    model,
    loss_function,
    optimizer,
    epoch,
    device,
):
```

한 Epoch 동안 모델을 학습합니다.

학습 모드로 전환합니다.

```python
model.train()
```

Batch에서 데이터를 가져옵니다.

```python
_, x1_batch, x2_batch, y_batch = batch
```

Tensor를 실행 장치로 이동합니다.

```python
x1_batch = x1_batch.to(device)
x2_batch = x2_batch.to(device)
y_batch = y_batch.to(device)
```

모델 출력을 계산합니다.

```python
y_pred = model(
    x1_batch,
    x2_batch,
)
```

Loss를 계산합니다.

```python
loss = loss_function(
    y_pred,
    y_batch,
)
```

정확도를 계산합니다.

```python
acc1, acc5 = accuracy(
    y_pred,
    y_batch,
    topk=(1, 5),
)
```

Gradient를 초기화하고 역전파를 수행합니다.

```python
optimizer.zero_grad(
    set_to_none=True
)
loss.backward()
optimizer.step()
```

순서는 다음과 같습니다.

```text
기존 Gradient 초기화
→ Loss 역전파
→ 각 Parameter의 Gradient 계산
→ Optimizer가 Parameter 업데이트
```

한 Epoch가 끝나면 평균 Loss, Top-1, Top-5 정확도를 반환합니다.

### 6.6 `CrossEntropyLoss`

`main.py`에서는 다음 Loss를 사용합니다.

```python
loss_function = nn.CrossEntropyLoss(
    weight=class_weights
)
```

모델 출력에 별도의 `softmax()`를 적용하지 않고 Logit을 그대로 전달합니다.

```python
y_pred = model(...)
loss = loss_function(
    y_pred,
    label,
)
```

`CrossEntropyLoss` 내부에 Log-Softmax 연산이 포함되어 있기 때문입니다.

다음과 같이 Softmax를 먼저 적용하면 학습 계산이 중복될 수 있습니다.

```python
# 권장하지 않는 형태
probability = torch.softmax(
    y_pred,
    dim=1,
)
loss = loss_function(
    probability,
    label,
)
```

### 6.7 검증 함수

```python
def validate(...):
```

검증, 테스트, 추론에서 공통으로 사용됩니다.

평가 모드로 전환합니다.

```python
model.eval()
```

Gradient 계산을 비활성화합니다.

```python
with torch.no_grad():
```

평가 과정에서는 역전파가 필요하지 않기 때문에 메모리 사용량과 계산량을 줄일 수 있습니다.

Top-k 결과도 함께 저장합니다.

```python
score, output = y_pred.topk(
    top_k,
    dim=1,
    largest=True,
    sorted=True,
)
```

여기에서

| 변수 | 의미 |
|---|---|
| `output` | 점수가 높은 Class Index |
| `score` | 해당 Class의 Logit 점수 |

예:

```text
output: [3, 0, 4, 2, 1]
score:  [4.5, 2.1, 1.3, 0.7, -0.2]
```

평가 결과는 Batch별 List에 저장한 후 마지막에 한 번만 결합합니다.

```python
ids = torch.cat(
    id_batches,
    dim=0,
)
```

반복문 안에서 `torch.cat()`을 계속 호출하는 것보다 효율적입니다.

### 6.8 Checkpoint 저장

검증 Loss가 기존 최저값보다 낮으면 모델을 저장합니다.

```python
if (
    not no_update
    and losses.avg < best_valid_losses
):
```

저장되는 주요 값은 다음과 같습니다.

```python
{
    "modelA": model.model_a.state_dict(),
    "modelB": model.model_b.state_dict(),
    "model": model.state_dict(),
    "model_config": model.model_config,
    "best_valid_loss": losses.avg,
}
```

각 항목의 의미는 다음과 같습니다.

| Key | 의미 |
|---|---|
| `modelA` | 문자 CNN 모델 Parameter |
| `modelB` | 단어 MLP 모델 Parameter |
| `model` | 전체 앙상블 모델 Parameter |
| `model_config` | 학습 시 사용한 모델 구조 설정 |
| `best_valid_loss` | 저장 시점의 검증 Loss |

실제 추론에서는 전체 앙상블 상태인 `checkpoint["model"]`을 불러옵니다.

---

## 7. `tools.py`

`tools.py`는 추론 입력을 만들고 모델 결과를 DataFrame으로 정리합니다.

### 7.1 `data_inferer`

```python
def data_inferer(x):
```

문자열 또는 문자열 목록을 추론용 DataFrame으로 바꿉니다.

예:

```python
data_inferer(
    [
        "아메리카노 카페",
        "행복 약국",
    ]
)
```

결과:

```text
id  text              label
0   아메리카노 카페    0
1   행복 약국          0
```

추론 데이터에는 정답 Label이 없으므로 모델 입력 구조를 맞추기 위한 임시 Label을 넣습니다.

### 7.2 `data_inferer_df`

```python
def data_inferer_df(x):
```

`text` 컬럼 하나만 있는 DataFrame에 `id`와 임시 `label`을 추가합니다.

입력:

```text
text
아메리카노 카페
행복 약국
```

출력:

```text
id  text              label
0   아메리카노 카페    0
1   행복 약국          0
```

### 7.3 Tensor를 NumPy로 변환

```python
def tensor_to_numpy(tensor):
    return (
        tensor
        .detach()
        .cpu()
        .numpy()
    )
```

처리 과정은 다음과 같습니다.

```text
계산 Graph에서 분리
→ CPU로 이동
→ NumPy Array로 변환
```

GPU Tensor는 직접 NumPy로 변환할 수 없기 때문에 먼저 CPU로 이동해야 합니다.

### 7.4 테스트 결과 정리

```python
def test_result_parser(
    id_tensor,
    result_tensor,
    score_tensor,
    args,
):
```

모델의 Top-k 결과를 원본 테스트 데이터와 결합합니다.

생성되는 주요 컬럼은 다음과 같습니다.

```text
id
text
label
pred1
pred2
pred3
pred4
pred5
scores
correct1
correct5
```

| 컬럼 | 의미 |
|---|---|
| `pred1` | 가장 높은 점수의 예측 Class |
| `pred2` | 두 번째 예측 Class |
| `pred3` | 세 번째 예측 Class |
| `pred4` | 네 번째 예측 Class |
| `pred5` | 다섯 번째 예측 Class |
| `scores` | `pred1`에 해당하는 Logit |
| `correct1` | Top-1 예측의 정답 여부 |
| `correct5` | Top-k 목록에 정답이 포함되는지 여부 |

`correct1`은 다음 방식으로 계산합니다.

```python
result["correct1"] = (
    labels == result["pred1"]
).astype(int)
```

`correct5`는 Top-k 예측 중 하나라도 정답과 같으면 1이 됩니다.

```python
result["correct5"] = (
    result[correct_columns]
    .sum(axis=1)
    > 0
).astype(int)
```

### 7.5 추론 결과 정리

```python
def inference_result_parser(
    id_tensor,
    result_tensor,
    score_tensor,
    args,
):
```

Top-1 예측 Class를 업종 메타정보와 결합합니다.

기본 출력 컬럼은 다음과 같습니다.

```text
id
store_category_index
scores
store_category_cd
store_category_nm
store_category_top_nm
store_category_mid_nm
asset_management_nm
```

예:

```text
0,0,3.51,C000,카페,음식,카페,일반
```

`store_category_index`를 기준으로 다음 파일과 결합합니다.

```text
meta_data/store_category_code_meta.csv
```

### 7.6 Score Threshold

```python
def score_threshold(
    score,
    index,
    threshold=-1.0,
    unknown_index=157,
):
```

예측 점수가 기준보다 낮으면 미분류 Class로 변경할 수 있습니다.

```text
score < threshold
→ unknown_index
```

기본 Threshold는 `-1.0`이므로 Threshold 기능은 비활성화되어 있습니다.

명령행에서 다음처럼 활성화할 수 있습니다.

```bash
python main.py \
  --mode inference \
  --score-threshold 2.0
```

현재 `scores`는 확률이 아니라 Raw Logit입니다. 운영 환경에서는 검증 데이터로 적절한 기준값을 찾거나 Softmax 확률을 기준으로 사용하는 것이 더 이해하기 쉬울 수 있습니다.

### 7.7 오분류 분석

```python
def make_incorrect_table(df):
```

Label별 테스트 성능과 자주 예측된 업종을 집계합니다.

주요 결과:

```text
label별 전체 샘플 수
Top-1 정답 수
Top-k 정답 수
Top-1 정확도
Top-k 정확도
Top-1에서 자주 예측된 Class
Top-k 전체에서 자주 등장한 Class
각 예측의 등장 비율
```

다음과 같은 컬럼이 생성됩니다.

```text
all
correct1
correct5
acc1
acc5
pred_1_1 ~ pred_1_5
rate_1_1 ~ rate_1_5
pred_5_1 ~ pred_5_5
rate_5_1 ~ rate_5_5
```

`pred_1_*`는 Top-1 예측에서 자주 나온 Class이고, `pred_5_*`는 Top-k 전체에서 자주 나온 Class입니다.

---

## 8. `logger.py`

```python
def make_logger(
    date_string,
    name,
    log_dir="./log",
):
```

실행 로그를 콘솔과 파일에 동시에 기록합니다.

```python
file_handler = logging.FileHandler(
    log_file,
    encoding="utf-8",
)

stream_handler = logging.StreamHandler()
```

로그 형식은 다음과 같습니다.

```text
2026-07-21 14:20:00 | INFO | mode=train, device=cpu
```

같은 이름의 Logger가 이미 만들어져 있으면 Handler를 중복으로 추가하지 않습니다.

```python
if log.handlers:
    return log
```

이를 통해 Colab이나 Notebook에서 셀을 여러 번 실행했을 때 동일 로그가 중복 출력되는 현상을 줄입니다.

---

## 9. `main.py`

`main.py`는 모든 파일을 연결하고 실행 모드를 결정하는 중심 코드입니다.

### 9.1 실행 옵션

```python
args = parse_args()
```

주요 실행 옵션은 다음과 같습니다.

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--mode` | 실행 모드 | `demo` |
| `--device` | CPU 또는 GPU | `cpu` |
| `--seed` | 난수 Seed | `42` |
| `--num-classes` | 업종 Class 개수 | `5` |
| `--epochs` | 학습 Epoch 수 | `3` |
| `--batch-size` | Batch 크기 | `8` |
| `--lr` | Learning Rate | `0.001` |
| `--optimizer` | Optimizer | `adam` |

모델 관련 옵션은 다음과 같습니다.

| 옵션 | 설명 | 기본값 |
|---|---|---:|
| `--seq-len1` | 문자 입력 최대 길이 | 30 |
| `--seq-len2` | 단어 입력 최대 길이 | 8 |
| `--embedding-size1` | 문자 Embedding 크기 | 32 |
| `--embedding-size2` | 단어 Embedding 크기 | 32 |
| `--out-size` | CNN Output Channel 수 | 8 |
| `--stride` | Pooling Stride | 1 |
| `--dropout` | Dropout 비율 | 0.2 |

실행 모드는 다음 네 가지입니다.

```text
demo
train
test
inference
```

### 9.2 난수 Seed

```python
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
```

실행마다 결과가 지나치게 달라지는 것을 줄이고 재현성을 높입니다.

완전히 동일한 결과까지 보장하는 설정은 아니지만, 모델 초기값과 일부 Random 연산을 일정하게 유지하는 데 도움이 됩니다.

### 9.3 Thread 수 설정

```python
torch.set_num_threads(
    max(1, args.num_threads)
)
```

CPU 실행 시 PyTorch가 사용할 Thread 수를 제어합니다.

샘플처럼 데이터가 매우 작을 때는 Thread 수가 지나치게 많으면 오히려 실행 준비 비용이 커질 수 있습니다.

### 9.4 실행 장치 선택

```python
def get_device(device_name):
    if (
        device_name == "gpu"
        and torch.cuda.is_available()
    ):
        return torch.device("cuda")

    return torch.device("cpu")
```

`--device gpu`를 지정했더라도 CUDA를 사용할 수 없으면 자동으로 CPU를 선택합니다.

### 9.5 폴더 생성

```python
def ensure_directories(args):
```

모델, 결과, 로그 폴더가 없으면 자동으로 만듭니다.

```python
os.makedirs(
    path,
    exist_ok=True,
)
```

Vocabulary 경로의 상위 폴더도 생성합니다.

### 9.6 모델 생성

```python
def create_model(
    args,
    vocab1,
    vocab2,
    device,
):
```

문자 모델과 단어 모델을 만든 후 앙상블로 결합합니다.

```python
model_a = models.SubwordCNN(
    args,
    vocab1,
    args.num_classes,
)

model_b = models.WordNet(
    args,
    vocab2,
    args.num_classes,
)

model = models.Ensemble(
    model_a,
    model_b,
    args.num_classes,
).to(device)
```

학습에 사용한 구조 설정도 저장합니다.

```python
model.model_config = {
    "num_classes": args.num_classes,
    "seq_len1": args.seq_len1,
    "seq_len2": args.seq_len2,
    "embedding_size1": args.embedding_size1,
    "embedding_size2": args.embedding_size2,
    "out_size": args.out_size,
    "stride": args.stride,
    "dropout": args.dropout,
}
```

Checkpoint를 불러올 때 동일한 구조를 재현하기 위해 사용합니다.

### 9.7 Class Weight

```python
def create_class_weights(
    labels,
    num_classes,
    device,
):
```

학습 데이터의 Class 불균형을 보정합니다.

```python
counts = np.bincount(
    labels,
    minlength=num_classes,
)
```

각 Class의 데이터 수를 계산합니다.

```python
weights[nonzero] = (
    len(labels)
    / (
        num_classes
        * counts[nonzero]
    )
)
```

데이터가 적은 Class는 더 큰 Weight를 받아 Loss에 더 크게 반영됩니다.

예:

```text
카페: 1,000개
약국:   100개
```

약국 Class에 상대적으로 큰 Weight가 적용됩니다.

샘플 데이터는 Class별 개수가 같으므로 Weight 차이가 거의 없습니다.

### 9.8 Optimizer 선택

```python
def create_optimizer(
    model,
    args,
):
```

다음 Optimizer를 지원합니다.

```text
Adam
RMSprop
SGD
```

예:

```bash
python main.py \
  --mode train \
  --optimizer adam
```

기본값은 Adam입니다.

### 9.9 최신 Checkpoint 찾기

```python
def find_latest_checkpoint(
    model_path,
    model_select,
):
```

`model_`로 시작하고 `_checkpoint.pt`로 끝나는 파일을 정렬한 뒤 마지막 파일을 선택합니다.

```python
files = sorted(
    file_name
    for file_name in os.listdir(model_path)
    if (
        file_name.startswith("model_")
        and file_name.endswith(model_select)
    )
)
```

학습된 모델이 없으면 명확한 오류를 발생시킵니다.

```python
raise FileNotFoundError(
    "checkpoint가 없습니다. "
    "먼저 train 또는 demo를 실행하세요."
)
```

### 9.10 Vocabulary 저장과 불러오기

```python
def save_vocab(
    vocab1,
    vocab2,
    path,
):
    torch.save(
        {
            "vocab1": vocab1,
            "vocab2": vocab2,
        },
        path,
    )
```

추론에서는 학습 때 사용한 Vocabulary를 그대로 사용해야 합니다.

같은 토큰이라도 새 Vocabulary를 만들면 Index가 달라질 수 있습니다.

```text
학습 Vocabulary: "카" → 7
새 Vocabulary:   "카" → 12
```

모델은 학습 시 Index 7에 대응하는 Embedding을 학습했기 때문에 추론 시에도 동일한 Mapping이 필요합니다.

### 9.11 모델 학습

```python
def train_model(
    args,
    device,
    mylogger,
):
```

수행 순서는 다음과 같습니다.

```text
학습 CSV 읽기
→ 검증 CSV 읽기
→ Label 정수 변환
→ Vocabulary 생성
→ DataLoader 생성
→ 모델 생성
→ Class Weight 생성
→ Loss 함수 생성
→ Optimizer 생성
→ Epoch 반복 학습
→ 검증
→ Best Checkpoint 저장
→ Vocabulary 저장
```

학습 데이터와 검증 데이터를 읽습니다.

```python
train_data = pd.read_csv(
    args.train_data_path
)

valid_data = pd.read_csv(
    args.valid_data_path
)
```

Label을 정수형으로 변환합니다.

```python
train_data["label"] = (
    train_data["label"].astype(int)
)
```

DataLoader 생성 시 학습 데이터만 섞습니다.

```python
train_loader = data_generator.load(
    train_data,
    shuffle=True,
)

valid_loader = data_generator.load(
    valid_data,
    shuffle=False,
)
```

Epoch마다 학습과 검증을 수행합니다.

```python
for epoch in range(
    1,
    args.epochs + 1,
):
    train_result = tr.train(...)
    valid_result = tr.validate(...)
```

Early Stopping 옵션이 켜져 있으면 최근 검증 Loss가 개선되지 않을 때 학습을 종료합니다.

```python
if (
    args.early_stop
    and len(val_losses) > 5
):
```

### 9.12 저장된 모델 설정 적용

```python
def _apply_saved_config(
    args,
    config,
):
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
```

추론 실행 옵션과 학습 시 모델 구조가 다른 경우 Checkpoint에 저장된 설정으로 맞춥니다.

예를 들어 학습 시

```text
embedding_size1 = 32
seq_len1 = 30
```

이었는데 추론 명령에서 다른 값을 지정하더라도 저장된 모델 구조를 우선 적용합니다.

### 9.13 모델 불러오기

```python
def load_model(
    args,
    vocab1,
    vocab2,
    device,
):
```

처리 순서는 다음과 같습니다.

```text
최신 Checkpoint 찾기
→ Checkpoint 불러오기
→ 저장된 모델 설정 적용
→ 동일한 구조의 모델 생성
→ State Dictionary 적용
→ 평가 모드 전환
```

```python
model.load_state_dict(
    checkpoint["model"]
)

model.eval()
```

### 9.14 테스트·추론 데이터 준비

```python
def prepare_test_data(args):
```

`test` 모드에서는 정답 Label이 있는 테스트 CSV를 읽습니다.

```python
if args.mode == "test":
    data = pd.read_csv(
        args.test_data_path
    )
```

`inference --adhoc` 모드에서는 사용자가 입력한 가맹점명을 `|` 기준으로 분리합니다.

```python
texts = [
    value.strip()
    for value in input(
        "가맹점명을 | 로 구분해 입력하세요: "
    ).split("|")
    if value.strip()
]
```

일반 추론에서는 `pred_data.csv`를 읽습니다.

`text` 컬럼 하나만 있으면 `tools.data_inferer_df()`를 이용해 모델 입력 형식으로 변환합니다.

### 9.15 테스트와 추론 실행

```python
def evaluate_or_infer(
    args,
    device,
    mylogger,
):
```

공통 수행 단계는 다음과 같습니다.

```text
입력 데이터 준비
→ Vocabulary 불러오기
→ DataLoader 생성
→ 모델 불러오기
→ 검증 함수로 모델 실행
→ 결과 Parser 적용
→ CSV 저장
```

테스트에서는 정답이 있으므로 다음 정보를 계산할 수 있습니다.

```text
Loss
Top-1 정확도
Top-k 정확도
오분류표
```

추론에서는 다음 정보를 생성합니다.

```text
예측 Class
예측 점수
업종 코드
업종명
업종 대분류
업종 중분류
```

### 9.16 `main()`

```python
def main():
```

실행 옵션을 읽고 Seed, 폴더, 장치, Logger를 준비합니다.

```python
args = parse_args()
device = get_device(args.device)
```

실행 모드에 따라 함수를 선택합니다.

```python
if args.mode == "demo":
    run_demo(
        args,
        device,
        mylogger,
    )

elif args.mode == "train":
    train_model(
        args,
        device,
        mylogger,
    )

else:
    evaluate_or_infer(
        args,
        device,
        mylogger,
    )
```

다음 조건문은 파일이 직접 실행될 때만 `main()`을 호출합니다.

```python
if __name__ == "__main__":
    main()
```

다른 Python 파일에서 `main.py`를 Import할 때는 자동으로 학습이나 추론이 시작되지 않습니다.

---

## 10. Tensor 전체 흐름

기본 옵션과 Batch 크기 8을 기준으로 Tensor 형태를 정리하면 다음과 같습니다.

### 10.1 문자 입력

```text
원문
→ 문자 토큰
→ 문자 Index
→ [8, 30]
→ Embedding
→ [8, 30, 32]
→ permute
→ [8, 32, 30]
→ CNN
→ MaxPool
→ flatten
→ Linear
→ [8, 5]
```

### 10.2 단어 입력

```text
원문
→ 단어 토큰
→ 단어 Index
→ [8, 8]
→ Embedding
→ [8, 8, 32]
→ flatten
→ [8, 256]
→ MLP
→ [8, 5]
```

### 10.3 앙상블

```text
SubwordCNN 출력 [8, 5]
WordNet 출력    [8, 5]
→ concatenate
→ [8, 10]
→ 최종 Linear
→ [8, 5]
```

최종 Tensor의 각 값은 해당 업종의 Logit입니다.

```text
[카페 점수, 한식 점수, 편의점 점수, 약국 점수, 의류 점수]
```

가장 높은 Logit의 Index가 Top-1 예측 업종입니다.

---

## 11. 실제 데이터 적용 시 수정할 부분

현재 샘플은 5개 Class입니다. 기존 프로젝트처럼 158개 Class를 사용하려면 다음과 같이 실행할 수 있습니다.

```bash
python main.py \
  --mode train \
  --num-classes 158 \
  --epochs 30 \
  --batch-size 128 \
  --device gpu
```

실제 데이터에는 다음 조건이 필요합니다.

```text
Label 값이 0부터 157 사이
업종 메타 파일에 각 Label의 정보 존재
학습·검증·테스트 CSV가 id, text, label 구조
추론 CSV에는 최소 text 컬럼 존재
```

### 11.1 Class 수

```bash
--num-classes 158
```

모델의 마지막 출력 크기와 Ensemble 출력 크기가 158로 변경됩니다.

### 11.2 업종 메타데이터

다음 컬럼이 필요합니다.

```text
store_category_index
store_category_cd
store_category_nm
store_category_top_nm
store_category_mid_nm
asset_management_nm
```

`store_category_index`는 학습 Label과 동일해야 합니다.

### 11.3 입력 길이

실제 가맹점명의 길이 분포에 따라 다음 옵션을 조정할 수 있습니다.

```bash
--seq-len1 50
--seq-len2 10
```

문자 입력 길이가 너무 짧으면 가맹점명 뒷부분이 잘릴 수 있고, 지나치게 길면 계산량과 Parameter 수가 증가합니다.

### 11.4 Vocabulary 최소 빈도

대규모 데이터에서는 한 번만 등장한 오타나 희귀 토큰을 제거하기 위해 다음 값을 높일 수 있습니다.

```bash
--min-freq-subword 5
--min-freq-word 3
```

### 11.5 Class 불균형

업종별 데이터 수 차이가 크면 현재 구현된 Class Weight가 적용됩니다.

데이터가 극단적으로 불균형하면 다음 방법도 검토할 수 있습니다.

- Oversampling
- WeightedRandomSampler
- Focal Loss
- 업종별 최소 데이터 수 확보
- 유사 업종 통합
- Label 품질 점검

---

## 12. 현재 데모에서 알아둘 점

### 12.1 Top-5 정확도

샘플은 Class가 5개뿐이므로 Top-5에는 전체 Class가 포함됩니다.

따라서 샘플 환경에서 Top-5 정확도는 대부분 100%가 되며 모델 품질을 구분하는 지표로는 의미가 제한적입니다.

158개 Class를 사용하는 실제 환경에서는 Top-5 정확도가 유용할 수 있습니다.

### 12.2 추론 Loss와 정확도

추론 데이터에는 정답이 없기 때문에 코드가 모델 입력 형식을 맞추기 위해 임시 Label을 추가합니다.

따라서 추론 실행 중 표시되는 Loss와 정확도는 실제 성능 지표로 해석하면 안 됩니다.

추론에서는 다음 정보를 확인해야 합니다.

```text
store_category_index
store_category_nm
scores
```

### 12.3 Score는 확률이 아님

`tools.py`의 `scores`는 Softmax 확률이 아니라 Raw Logit입니다.

```text
3.5가 35%라는 의미가 아님
-1보다 작다고 반드시 매우 낮은 확률이라는 의미가 아님
서로 다른 샘플의 Logit을 단순 비교하기 어려울 수 있음
```

확률이 필요하면 모델 출력에 Softmax를 적용할 수 있습니다.

```python
probabilities = torch.softmax(
    y_pred,
    dim=1,
)
```

Top-1 확률과 Index는 다음과 같이 구할 수 있습니다.

```python
top_probability, top_index = (
    probabilities.max(dim=1)
)
```

Confidence Threshold를 운영에 적용할 때는 검증 데이터에서 적절한 값을 찾는 것이 좋습니다.

### 12.4 BatchNorm과 Batch 크기

`WordNet`에는 다음 Layer가 있습니다.

```python
nn.BatchNorm1d(256)
```

학습 모드에서 Batch 크기가 1인 경우 BatchNorm이 통계값을 계산하지 못해 오류가 발생할 수 있습니다.

마지막 Batch가 1개가 될 가능성이 있다면 다음 방법을 검토할 수 있습니다.

```python
DataLoader(
    ...,
    drop_last=True,
)
```

또는 `BatchNorm1d`를 `LayerNorm`으로 변경할 수도 있습니다.

### 12.5 Vocabulary와 모델은 함께 관리

다음 두 파일은 서로 연결된 학습 결과입니다.

```text
Vocabulary
Checkpoint
```

모델만 교체하거나 Vocabulary만 새로 만들면 토큰 Index가 달라져 결과가 비정상적일 수 있습니다.

학습 모델을 배포할 때는 다음 항목을 하나의 버전으로 관리하는 것이 좋습니다.

```text
모델 Checkpoint
Vocabulary
업종 메타데이터
모델 설정
전처리 코드
```

### 12.6 샘플 데이터의 한계

샘플 데이터는 코드 연결 확인을 위한 매우 작은 데이터입니다.

문장 안에 `카페`, `약국`, `의류`처럼 업종을 직접 나타내는 단어가 포함되어 있어 실제 업무 데이터보다 분류가 쉽습니다.

실제 성능을 평가하려면 다음이 필요합니다.

- 실제 가맹점명 데이터
- 충분한 업종별 샘플 수
- 중복 제거
- 학습·검증·테스트 데이터 분리
- Label 오류 점검
- 유사 상호의 데이터 누수 방지
- 업종별 Precision, Recall, F1-score 분석

---

## 13. 프로젝트 핵심 요약

이 프로젝트의 핵심 구조는 다음과 같습니다.

```text
문자 패턴에 강한 SubwordCNN
+
단어 조합에 강한 WordNet
+
두 모델의 판단을 결합하는 Ensemble
=
가맹점명 업종 분류
```

각 파일은 다음처럼 연결됩니다.

```text
sample_data.py
   └─ 샘플 CSV 생성

data_preprocess.py
   └─ 텍스트를 문자·단어 Tensor로 변환

model.py
   └─ 문자 CNN, 단어 MLP, Ensemble 정의

utils.py
   └─ 학습, 검증, 정확도, Checkpoint 저장

tools.py
   └─ 테스트·추론 결과와 업종 메타정보 결합

logger.py
   └─ 실행 로그 기록

main.py
   └─ 모든 모듈을 연결하고 실행 모드 제어
```
