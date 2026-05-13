import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


class ActivationExtractor:

    def __init__(
        self,
        model_name,
        layer_idx,
        device="cuda",
        max_length=64,
        normalize=True
    ):

        self.device = device

        self.layer_idx = layer_idx

        self.max_length = max_length

        self.normalize = normalize

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(model_name)
            .to(device)
        )

        self.model.eval()

        self.hidden_states = None

        #
        # GPT-style transformer block
        #

        target_layer = (
            self.model.transformer.h[layer_idx]
        )

        target_layer.register_forward_hook(
            self.hook_fn
        )

    def hook_fn(
        self,
        module,
        inputs,
        outputs
    ):

        #
        # GPT2 blocks may return tuple
        #

        if isinstance(outputs, tuple):
            hidden = outputs[0]
        else:
            hidden = outputs

        self.hidden_states = (
            hidden.detach()
        )

    @torch.no_grad()
    def extract_sequence(
        self,
        text
    ):

        toks = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        _ = self.model(**toks)

        hidden = self.hidden_states.squeeze(0)

        if self.normalize:

            hidden = F.normalize(
                hidden,
                dim=-1
            )

        return hidden.cpu()

    @torch.no_grad()
    def extract_pooled(
        self,
        text,
        pooling="mean"
    ):

        seq = self.extract_sequence(text)

        if pooling == "mean":
            return seq.mean(dim=0)

        elif pooling == "last":
            return seq[-1]

        elif pooling == "max":
            return seq.max(dim=0).values

        else:
            raise ValueError(
                f"Unknown pooling: {pooling}"
            )