import torch.nn as nn


class ActivationEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim,
        latent_dim=512
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
                latent_dim
            )
        )

    def forward(
        self,
        x
    ):

        return self.net(x.mean(dim=1))