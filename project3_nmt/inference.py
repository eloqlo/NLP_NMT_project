from util.data_utils import load_dataset
from util.args import NMTArgument
import os
from transformers import AdamW, BertTokenizer, ElectraTokenizer
from util.batch_generator import NMTBatchfier
from model.seq2seq import Seq2Seq
import numpy as np

import random
from torch.utils.data import IterableDataset, DataLoader
import torch

DEVICE = "cuda"
MODELPATH = "checkpoint/vanila"  # plz specify directory
RESULTDIR="results"

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_batchfier(args, source_tokenizer, target_tokenizer):
    n_gpu = torch.cuda.device_count()
    train, dev, test, test2 = load_dataset(args, source_tokenizer, target_tokenizer)
    padding_idx = args.pad_id

    train_batch = NMTBatchfier(args, train, batch_size=args.per_gpu_train_batch_size * n_gpu, maxlen=args.seq_len,
                               padding_index=padding_idx, device=DEVICE)
    dev_batch = NMTBatchfier(args, dev, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                             padding_index=padding_idx, device=DEVICE)
    test_batch = NMTBatchfier(args, test, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                              padding_index=padding_idx, device=DEVICE,test=True)

    test2_batch = NMTBatchfier(args, test2, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                               padding_index=padding_idx, device=DEVICE,test=True)



    return train_batch, dev_batch, test_batch, test2_batch


# 영어 -> 한국어  Generation 하는 과정
def run(args):
    # args = ExperimentArgument()

    args.aug_ratio = 0.0
    set_seed(args.seed)

    if not os.path.isdir(RESULTDIR):
        os.makedirs(RESULTDIR)

    print(args.__dict__)

    source_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    target_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")

    args.pad_id =source_tokenizer.pad_token_id
    tgt_vocab_size = target_tokenizer.vocab_size
    model = Seq2Seq(tgt_vocab_size)

    args.extended_vocab_size = 0
    train_gen, dev_gen, test_gen, test2_gen = get_batchfier(args, source_tokenizer, target_tokenizer)

    model.to(DEVICE)

    if MODELPATH == "":
        raise EnvironmentError("require to clarify the argment of model_path")


    # best 모델의 checkpoint 를 가져와서 model 에 주입
    state_dict = torch.load(os.path.join(MODELPATH, "best_model", "best_model.bin"))
    model.load_state_dict(state_dict)


    if isinstance(test_gen, IterableDataset):
        test_gen = DataLoader(dataset=test_gen,
                               batch_size=test_gen.size,
                               shuffle=False,
                               collate_fn=test_gen.collate_test)


    if isinstance(test2_gen, IterableDataset):
        test2_gen = DataLoader(dataset=test2_gen,
                               batch_size=test2_gen.size,
                               shuffle=False,
                               collate_fn=test2_gen.collate_test)

    model.eval()

    from util.evaluation import NMTEvaluator

    # generation/mini-batch
    evaluator = NMTEvaluator(args,model,target_tokenizer=target_tokenizer, device=DEVICE)

    # save the results
    test_results=evaluator.generate_epoch(test_gen)
    test_results.to_csv(os.path.join(RESULTDIR,"result.test.csv"),header=True,index=False)

    test2_results=evaluator.generate_epoch(test2_gen)
    test2_results.to_csv(os.path.join(RESULTDIR, "result.test2.csv"),header=True,index=False)


if __name__ == "__main__":
    args = NMTArgument()
    run(args)
