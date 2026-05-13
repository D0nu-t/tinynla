import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModel
)


class TokenLevelReconstructor(nn.Module):

    def __init__(
        self,
        hidden_dim,
        n_layers=4,
        n_heads=8,
        max_len=64,
        encoder_name="BAAI/bge-base-en-v1.5"
    ):

        super().__init__()

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                encoder_name
            )
        )

        self.encoder = (
            AutoModel.from_pretrained(
                encoder_name
            )
        )

        embed_dim = (
            self.encoder.config.hidden_size
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers
        )

        self.positional = nn.Parameter(
            torch.randn(
                1,
                max_len,
                embed_dim
            )
        )

        self.output_proj = nn.Linear(
            embed_dim,
            hidden_dim
        )

        self.max_len = max_len

    def encode_text(
        self,
        texts,
        device
    ):

        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        ).to(device)

        outputs = self.encoder(**toks)

        return outputs.last_hidden_state

    def forward(
        self,
        texts,
        seq_len,
        device
    ):

        semantic_tokens = self.encode_text(
            texts,
            device
        )

        batch_size = semantic_tokens.shape[0]

        queries = (
            self.positional[:, :seq_len]
            .repeat(batch_size, 1, 1)
        )

        decoded = self.decoder(
            tgt=queries,
            memory=semantic_tokens
        )

        activations = self.output_proj(
            decoded
        )

        return activations