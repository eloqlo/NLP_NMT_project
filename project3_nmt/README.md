## Baseline model of Goorm Project-3 NMT

---

���δ� �翬�� ��������ϰ�, ���� �� ����� ���ϴ°Ŵ�. ����Ÿ� �����Ÿ������� ���� ����.<br>
��ư�, ���� ��������.<br>
�ѹ� ����� �������� ���� ������. �߰��� �������� �����ϴ� ������� ũ�� ����.<br>
�������� ��´�� �ൿ�ϴ� ���Ĺ��� �뿹ó�� �ൿ����, �ƹ��� �𸣰ھ �������� �޾Ƶ鿩��. �˰��� ����Ŵ�.<br>

---

test1 : 19%<br>
test2 : 11.75%<br>
������� = �� ������ ��ȭ���<br>
Harmonic Mean = 14.52<br>

<�н��ӵ� �ø��� ����>
1. Bucketing(len ���� sorting�ؼ� ȿ�� ���̱�)<br>
2. Generation �Ǵ� strategy �� ��ȿ������ ��Ȳ<br>
   * decoder���� ���� time step�� key, value �� �����ͼ� �����س����� ��� ȿ���� ����<br>
   * decoder����, 512��° seq input�� ���� output ����� ��, ������ ����� key, value���� �ҷ��ͼ�
     self-attention ����ϸ� �ȴ�.

3. Post-process(Grid, Beam Search) ���� ��ȿ�������� ¥�����ִ� ��Ȳ<br>
4. Auxiliary Loss<br>
5. Batch Accumulation<br>

---

��-> �� ������ ���� Encoder-Decoder ��

- �ѱ��� ������ ���� Decoder���� `monologg/koelectra-base-v3-discriminator`�� `token_embeddings`�� �ѱ��� pretrained Subword Embedding���� ����մϴ�.
- ���� �����͸� ���� Encoder���� `bert-base-uncased`�� pretrained model�� ����մϴ�.



### 1. �ʿ��� ���̺귯�� ��ġ

`pip install -r requirements.txt`  <<  ������

### 2. �� �н�

`script/train.sh`�� �����մϴ�

�н��� ���� epoch ���� `CHECKPOINT/epoch-{number}.bin` ���� ����˴ϴ�.<br>
Best Checkpoint�� `CHECKPOINT/best_model`�� ����˴ϴ�.<br>

### 3. �߷��ϱ�

`script/test.sh`�� �����մϴ�

### 4. �����ϱ�

3�� ���� `inference.py`���� `RESULTDIR`�� ����� `result.test.csv`�� `result.test2.csv`�� �����մϴ�.

### 5. Metric : util/metrics.py

- BLEU - 4 gram (upto)
- Sentence-level BLEU
- Pos tagger : Mecab


