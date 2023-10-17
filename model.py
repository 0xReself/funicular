import torch
from torch import nn

#funicular
class Swipo(torch.nn.Module):
    def __init__(self):
        super(Swipo, self).__init__()

        self.layers = torch.nn.Sequential(
            nn.Linear(128*128, 2048),
            nn.LeakyReLU(0.1),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)