import torch
import torch.mm


class Transformer(mm.Model):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init()

        self.d_model = d_model
        self.vocab_size = vocab_size