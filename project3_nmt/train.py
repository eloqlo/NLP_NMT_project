from util.trainer import NMTTrainer
from util.data_utils import load_dataset
from util.args import NMTArgument
import os
from transformers import AdamW, BertTokenizer, AutoTokenizer
import tqdm
from util.batch_generator import NMTBatchfier
from model.seq2seq import Seq2Seq
import numpy as np

from transformers import get_scheduler

import torch.nn as nn
import torch
import random
import logging

logger = logging.getLogger(__name__)


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# optimizer, loss_function 선언부
# 왜 CrossEntropyLoss 를 사용하는게 맞는걸까?
def get_trainer(args, model, train_batchfier, test_batchfier):
    optimizer = AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    criteria = nn.CrossEntropyLoss(ignore_index=args.pad_id)

    if args.max_train_steps is None:
        args.num_update_steps_per_epoch = train_batchfier.num_buckets
        args.max_train_steps = args.n_epoch * args.num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # AdamW, Loss, lr_scheduler, batch, grad_accumulation, clip_norm 설정한 trainer
    trainer = NMTTrainer(args, model, train_batchfier, test_batchfier, optimizer, args.gradient_accumulation_step,
                         criteria, args.clip_norm, lr_scheduler)

    return trainer

# batch DATALOADER : 배치별로 data 주네
def get_batchfier(args, source_tokenizer, target_tokenizer):
    n_gpu = torch.cuda.device_count()
    train, dev, test, test2 = load_dataset(args, source_tokenizer, target_tokenizer)
    padding_idx = args.pad_id

    train_batch = NMTBatchfier(args, train, batch_size=args.per_gpu_train_batch_size * n_gpu, maxlen=args.seq_len,
                               padding_index=padding_idx, device="cuda")
    dev_batch = NMTBatchfier(args, dev, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                             padding_index=padding_idx, device="cuda")
    test_batch = NMTBatchfier(args, test, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                              padding_index=padding_idx, device="cuda")

    test2_batch = NMTBatchfier(args, test2, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                               padding_index=padding_idx, device="cuda")

    return train_batch, dev_batch, test_batch, test2_batch


def run(gpu, args):
    # args = ExperimentArgument()

    args.gpu = gpu
    args.device = gpu       # 몇번째 GPU 인지 지정
    args.aug_ratio = 0.0
    set_seed(args.seed)     # seed 지정

    print(args.__dict__)

    source_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")   # 영어표현위해 BERT tokenizer 사용 <tokenizer-BERT>
    target_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-discriminator")    # decoder 은 KoELECTRA tokenizer 사용

    args.pad_id = source_tokenizer.pad_token_id     # padding index 0 들어감.
    tgt_vocab_size = target_tokenizer.vocab_size
    model = Seq2Seq(tgt_vocab_size)  # model/Seq2Seq.py

    args.extended_vocab_size = 0
    train_gen, dev_gen, test_gen, test2_gen = get_batchfier(args, source_tokenizer, target_tokenizer)

    model.to("cuda")

    trainer = get_trainer(args, model, train_gen, dev_gen)  # trainer
    best_dir = os.path.join(args.savename, "best_model")

    if not os.path.isdir(best_dir):
        os.makedirs(best_dir)

    results = []

    optimal_perplexity = 1000.0  # perplexity : model 의 시점당 평균 선택지
    not_improved = 0

    if args.do_train:
        for e in tqdm(range(0, args.n_epoch)):
            # train
            print("Epoch : {0}".format(e))
            trainer.train_epoch()
            save_path = os.path.join(args.savename, "epoch_{0}".format(e))
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            # eval 진행
            if args.evaluate_during_training:
                loss, step_perplexity = trainer.test_epoch()
                results.append({"eval_loss": loss, "eval_ppl": step_perplexity})
                torch.save(model.state_dict(), save_path)   # epoch 별로 checkpoint 저장한다.
                print("Update Model checkpoints per epoch compeleted! ")

                # epoch 별로 가장 낮은 perplexity 를 저장하도록 만들었다.
                if optimal_perplexity > step_perplexity:
                    optimal_perplexity = step_perplexity
                    torch.save(model.state_dict(), os.path.join(best_dir, "best_model.bin"))
                    print("Update Model checkpoints at {0}!! ".format(best_dir))
                    not_improved = 0
                else:
                    not_improved += 1

            if not_improved >= 5:
                break

    if args.do_test:
        if args.model_path_list == "":
            raise EnvironmentError("require to clarify the argment of model_path")

        model_path = [model_path for model_path in args.model_path_list][0]
        state_dict = torch.load(os.path.join(model_path, "best_model", "best_model.bin"))
        model.load_state_dict(state_dict)

        if args.model_path_list == "":
            raise EnvironmentError("require to clarify the argment of model_path")


if __name__ == "__main__":
    args = NMTArgument()    # argument 파일이 먼저 실행이 된다. 미리 지정한 arguments가 args에 들어간다.
    run("cuda", args)
