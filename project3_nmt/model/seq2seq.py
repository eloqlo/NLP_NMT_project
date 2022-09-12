import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, DistilBertModel, AutoModel

# from transformers.configuration_bert import BertConfig
import numpy as np
from .transformer import Decoder, DecoderLayer      # 하버드 nlp 에서 transformer 구현한거 바탕으로 직접 작성한 transformer.py 파일


def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss


class Seq2Seq(nn.Module):
    def __init__(self, tgt_vocab_size, n_layer=6, pad_idx=0, device="cuda"):
        super(Seq2Seq, self).__init__()
        self.config = BertConfig.from_pretrained("bert-base-uncased")

        # ENCODER
        self.encoder = BertModel.from_pretrained("bert-base-uncased")    # BERT

        # decoder configuration
        d_model = self.config.hidden_size
        num_head = self.config.num_attention_heads
        d_ff = self.config.intermediate_size
        dropout_rate = self.config.hidden_dropout_prob

        self.device = device

        # DECODER
        self.decoder = Decoder(tgt_vocab_size, num_head, d_model, d_ff, n_layer=n_layer, dropout=dropout_rate)  # 하버드 nlp 에서 transformers 구현해놓은 코드 기반으로 작성
        self.lm_head = nn.Linear(d_model, tgt_vocab_size)
        self.pad_idx = pad_idx

        # DECODER - embedding
        electra_model = AutoModel.from_pretrained("monologg/koelectra-base-discriminator")
        target_embedding_matrix = electra_model.embeddings.word_embeddings.weight.data
        self.set_target_embedding(target_embedding_matrix)

    def set_target_embedding(self, trg_embedding_matrix):
        self.decoder.embeddings.word_embedding.weight.data = trg_embedding_matrix   # 불러온 weight를 내꺼에 넣는다.
        self.lm_head.weight.data = self.decoder.embeddings.word_embedding.weight.data

    def forward(self, input_ids, attention_mask, tgt_ids):
        src_hidden = self.encoder(input_ids)["last_hidden_state"]   # input의 encoder 통과결과인 key, value(encoder의 memory)를 받고

        tgt_size = tgt_ids.size(1)
        tgt_mask = self.subsequent_mask(tgt_size).to(self.device)

        out = self.decoder(tgt_ids, src_hidden, attention_mask, tgt_mask)   # memory,  encoder의 att mask, target att mask 를 dcoder에 넣는다.

        return self.lm_head(out)    # dense layer

    def subsequent_mask(self, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

        return torch.from_numpy(subsequent_mask) == 0

    def generate(self, memory, attention_mask, tgt_ids):
        tgt_size = tgt_ids.size(1)
        tgt_mask = self.subsequent_mask(tgt_size).to(self.device)
        out = self.decoder(tgt_ids, memory, attention_mask, tgt_mask)

        return self.lm_head(out)[:, -1]
