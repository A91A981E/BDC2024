import torch.nn as nn


class Gate(nn.Module):
    def __init__(self, feature_dim, n_expert):
        super(Gate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, n_expert),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.gate(x)
