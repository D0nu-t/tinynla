import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):

    def __init__(
        self,
        n_codes=1024,
        code_dim=512
    ):

        super().__init__()

        self.codebook = nn.Embedding(
            n_codes,
            code_dim
        )

    def forward(
        self,
        z
    ):

        distances = torch.cdist(
            z,
            self.codebook.weight
        )

        indices = distances.argmin(dim=-1)

        z_q = self.codebook(indices)

        return z_q, indices