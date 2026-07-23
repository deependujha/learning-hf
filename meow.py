import torch
from torch import nn
from safetensors.torch import load_model, save_model


a = torch.zeros((100, 100))
b = a[:1, :]
# torch.save({"b": b}, "model.bin")
# File is 41k instead of the expected 400 bytes
# In practice it could happen that you save several 10GB instead of 1GB.
save_model({"b":b}, "model.safetensors")

