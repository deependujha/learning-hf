import torch
from torch import nn
from safetensors.torch import load_model, save_model


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(100, 100)
        self.b = self.a

    def forward(self, x):
        return self.b(self.a(x))


model = Model()
print(model.state_dict())
# odict_keys(['a.weight', 'a.bias', 'b.weight', 'b.bias'])
torch.save(model.state_dict(), "model.bin")
# This file is now 41k instead of ~80k, because A and B are the same weight hence only 1 is saved on disk with both `a` and `b` pointing to the same buffer

save_model(model, "model.safetensors")

