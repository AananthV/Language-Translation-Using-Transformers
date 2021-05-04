import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

source_vocab_length=100
target_vocab_length=100