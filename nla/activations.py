import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.hooks import ActivationHook


class ActivationExtractor:
    def __init__(self, model_name, layer_idx, device="cuda"):
        self.device = device
        self.layer_idx = layer_idx

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        ).to(device)

        self.model.eval()

        self.hook = ActivationHook()

        target_layer = self.model.transformer.h[layer_idx]

        target_layer.register_forward_hook(self.hook.hook_fn)

    @torch.no_grad()
    def extract(self, text):

        self.hook.clear()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64
        ).to(self.device)

        _ = self.model(**inputs)

        hidden = self.hook.activations[0]
        print(hidden.shape)
        #
        # Ensure shape:
        # [batch, seq, hidden]
        #

        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        vec = hidden[:, -1, :]

        vec = F.normalize(vec, dim=-1)

        return vec.squeeze(0).cpu()