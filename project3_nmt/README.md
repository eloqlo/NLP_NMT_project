## Baseline model of Goorm Project-3 NMT

---

공부는 당연히 어려워야하고, 많이 한 사람이 잘하는거다. 쉬운거만 뒤적거리려하지 절대 말아.<br>
어렵게, 많이 공부하자.<br>
한번 결단을 내렸으면 끝을 봐보자. 중간에 흐지부지 포기하는 사람으로 크지 말자.<br>
딴생각이 드는대로 행동하는 도파민의 노예처럼 행동말고, 아무리 모르겠어도 정면으로 받아들여라. 알고나면 쉬울거다.<br>

---

test1 : 19%<br>
test2 : 11.75%<br>
최종결과 = 두 점수의 조화평균<br>
Harmonic Mean = 14.52<br>

<학습속도 올리는 방향>
1. Bucketing(len 별로 sorting해서 효율 높이기)<br>
2. Generation 되는 strategy 가 비효율적인 상황<br>
   * decoder에서 이전 time step의 key, value 값 가져와서 저장해놓으면 계산 효율성 생김<br>
   * decoder에서, 512번째 seq input에 대한 output 계산할 때, 이전에 계산한 key, value만을 불러와서
     self-attention 계산하면 된다.

3. Post-process(Grid, Beam Search) 에서 비효율적으로 짜여져있는 상황<br>
4. Auxiliary Loss<br>
5. Batch Accumulation<br>

---

영-> 한 번역을 위한 Encoder-Decoder 모델

- 한국어 번역을 위해 Decoder에서 `monologg/koelectra-base-v3-discriminator`의 `token_embeddings`을 한국어 pretrained Subword Embedding으로 사용합니다.
- 영어 데이터를 위해 Encoder에서 `bert-base-uncased`의 pretrained model로 사용합니다.



### 1. 필요한 라이브러리 설치

`pip install -r requirements.txt`  <<  돼있음

### 2. 모델 학습

`script/train.sh`를 실행합니다

학습된 모델은 epoch 별로 `CHECKPOINT/epoch-{number}.bin` 으로 저장됩니다.<br>
Best Checkpoint가 `CHECKPOINT/best_model`에 저장됩니다.<br>

### 3. 추론하기

`script/test.sh`를 실행합니다

### 4. 제출하기

3번 스텝 `inference.py`에서 `RESULTDIR`에 저장된 `result.test.csv`와 `result.test2.csv`을 제출합니다.

### 5. Metric : util/metrics.py

- BLEU - 4 gram (upto)
- Sentence-level BLEU
- Pos tagger : Mecab


