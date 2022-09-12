import torch


from model.seq2seq import Seq2Seq


class SamplerBase:
    def __init__(self, model, seq_length):
        self.model = model
        self.seq_length = seq_length

    def sample(self, inps, past):
        return NotImplementedError

# [#2, 13:40] input 이 들어왔을 때, 한국어문장 생성하도록하는 클래스?
class GreedySampler(SamplerBase):
    def __init__(self, model: Seq2Seq, seq_length, start_index=2,eos_index=3):
        """
        :param model:
        :param seq_length:
        :param stochastic: choice [top_k,top_p] if True
        """
        super(GreedySampler, self).__init__(model, seq_length)
        self.start_index = start_index
        self.eos_index=eos_index

    @torch.no_grad()
    def sample(self, input_ids, attention_mask, generated):
        source_hidden = self.model.encoder(input_ids, attention_mask)["last_hidden_state"]

        for t in range(0, self.seq_length):
            lm_logits = self.model.generate(source_hidden,attention_mask,generated)
            predict = self.sampling(lm_logits)
            generated = torch.cat([generated, predict], dim=-1)
            if self.is_terminated(generated):
                break

        return generated

    def sampling(self, logits):
        return torch.argmax(logits, -1, keepdim=True)

    def is_terminated(self,mini_batch):

        num_of_eos=torch.sum(mini_batch==self.eos_index,-1)

        return torch.prod(num_of_eos) > 0
