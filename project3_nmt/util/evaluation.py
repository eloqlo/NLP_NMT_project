from model.seq2seq import Seq2Seq
import pandas as pd
from tqdm import tqdm
from .sampler import GreedySampler
import torch


class EvaluatorBase:
    def __init__(self, args, model: Seq2Seq):
        self.args = args
        self.model = model
        self.step = 0

    def reformat_inp(self, inp):
        raise NotImplementedError

    def generate_epoch(self, dataset):
        raise NotImplementedError

# 전체 datasets mini_batch 단위로 돌면서 generation하도록 만든 클래스
class NMTEvaluator(EvaluatorBase):
    def __init__(self, args, model: Seq2Seq, target_tokenizer,device):
        super(NMTEvaluator, self).__init__(args, model)
        self.tokenizer = target_tokenizer
        self.device= device
        self.start_id= 2
        self.eos_id = 3
        self.sampler = GreedySampler(model, 256,self.start_id,self.eos_id)
        self.model.eval()

    def generate_epoch(self, batchfier):
        generated = []

        batchfier = [batch for batch in batchfier]
        sid_list = []
        pbar = tqdm(batchfier)

        for inputs in pbar:
            sid = inputs["sid"]
            input_ids = inputs["input_ids"]
            bsz = input_ids.size(0)

            attn_mask = inputs["attention_mask"]
            tgt_ids = (torch.ones([bsz, 1])*self.start_id).long().to(self.device)

            with torch.no_grad():
                # encoder_output = self.model.model.encoder(input_ids=input_ids, attention_mask=attn_mask)
                outputs = self.sampler.sample(input_ids, attn_mask, tgt_ids)

            tokenized_sequences = self.remove_after_eos(outputs.cpu().tolist())
            detokenized = [self.tokenizer.decode(sequence) for sequence in tokenized_sequences]
            generated.extend(detokenized)
            sid_list.extend(sid)

        return pd.DataFrame({'sid': sid_list, 'predicts': generated})

    def remove_after_eos(self, sequences):
        res=[]
        for sequence in sequences:
            if self.eos_id in sequence:
                sequence = sequence[:sequence.index(self.eos_id)]

            res.append(sequence[1:]) # eliminate start idx

        return res




