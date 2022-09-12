from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
import torch
from model.seq2seq import Seq2Seq
import math


class TrainerBase:
    def __init__(self, args, model: Seq2Seq, train_batchfier, test_batchfier, optimizers,
                 update_step, criteria, clip_norm, scheduler):
        self.args = args
        self.model = model
        self.train_batchfier = train_batchfier
        self.test_batchfier = test_batchfier
        self.optimizers = optimizers
        self.criteria = criteria
        self.step = 0
        self.update_step = update_step
        self.clip_norm = clip_norm
        self.scheduler = scheduler

    def reformat_inp(self, inp):
        raise NotImplementedError

    def train_epoch(self):
        return NotImplementedError

    def test_epoch(self):
        return NotImplementedError


class NMTTrainer(TrainerBase):
    def __init__(self, args, model: Seq2Seq, train_batchfier, test_batchfier, optimizers,
                 update_step, criteria, clip_norm, scheduler):
        super(NMTTrainer, self).__init__(args, model, train_batchfier, test_batchfier, optimizers,
                                         update_step, criteria, clip_norm, scheduler)

    def reformat_inp(self, inp):
        inp_tensor = tuple(i.to("cuda") for i in inp)
        return inp_tensor

    def train_epoch(self):
        model = self.model
        batchfier = self.train_batchfier
        optimizer = self.optimizers

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate)

        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0

        pbar_cnt = 0
        model.zero_grad()
        pbar = tqdm(batchfier, total=batchfier.dataset.num_buckets)

        for inputs in pbar:
            input_ids = inputs["input_ids"]
            attn_mask = inputs["attention_mask"]
            tgt_ids = inputs["decoder_input_ids"]
            labels = inputs["labels"]

            logits = model(input_ids,attn_mask,tgt_ids)
            # logits = outputs["logits"]

            loss = self.criteria(logits.view(-1, logits.size(-1)), labels.view(-1))
            step_loss += loss.item()
            tot_cnt += 1

            loss.backward()

            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                self.scheduler.step(self.step)
                model.zero_grad()
                ppl = math.exp(step_loss / (self.update_step * pbar_cnt))
                pbar.set_description(
                    "training loss : %f, ppl: %f , iter : %d" % (
                        step_loss / (self.update_step * pbar_cnt), ppl,
                        n_bar), )
                pbar.update()

        pbar.close()

    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier

        criteria = self.criteria

        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate)

        model.eval()
        # cached_data_loader=get_cached_data_loader(batchfier,batchfier.size,custom_collate=batchfier.collate,shuffle=False)
        model.zero_grad()
        pbar = tqdm(batchfier)
        pbar_cnt = 0
        step_loss = 0

        for inputs in pbar:
            with torch.no_grad():
                input_ids = inputs["input_ids"]
                attn_mask = inputs["attention_mask"]
                tgt_ids = inputs["decoder_input_ids"]
                labels = inputs["labels"]

                logits = model(input_ids, attn_mask, tgt_ids)

                loss = self.criteria(logits.view(-1, logits.size(-1)), labels.view(-1))
                step_loss += loss.item()
                pbar_cnt += 1

            pbar.set_description(
                "test loss : %f  test perplexity : %f" % (
                    step_loss / pbar_cnt, math.exp(step_loss / pbar_cnt)), )
            pbar.update()
        pbar.close()


        return step_loss / pbar_cnt,  math.exp(step_loss / pbar_cnt)