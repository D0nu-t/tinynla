import torch
import torch.nn as nn


class DiffusionDenoiser(nn.Module):

    def __init__(
        self,
        hidden_dim
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                hidden_dim,
                2048
            ),

            nn.GELU(),

            nn.Linear(
                2048,
                hidden_dim
            )
        )

    def forward(
        self,
        noisy_x
    ):

        return self.net(noisy_x)