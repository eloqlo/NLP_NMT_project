from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, ngrams, brevity_penalty
from eunjeon import Mecab

def bleu_upto(reference, hypothesis, n_gram):
    res = []
    for i in range(1, n_gram + 1):
        res.append(calc_bleu_ngram(reference, hypothesis, i))
    return res


def corpuswise_bleu(predicts, gts, n_gram=4):
    res_predict = []
    res_gt = []

    mecab = Mecab()

    for predict in predicts:
        res_predict.append([i[0] for i in mecab.pos(predict)])

    for gt in gts:
        res_gt.append([i[0] for i in mecab.pos(gt)])

    return bleu_upto(res_gt, res_predict, n_gram)


def calc_bleu_ngram(reference, hypothesis, n_gram):
    score = 0.0
    ratio = 1 / n_gram

    cc = SmoothingFunction()

    for refer, hypo in zip(reference, hypothesis):
        # refer.index()
        score += sentence_bleu([refer], hypo, (ratio,) * n_gram, cc.method1)

    return score / len(reference)
