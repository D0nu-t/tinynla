import torch


class ActivationHook:
    def __init__(self):
        self.activations = []

    def hook_fn(self, module, inputs, outputs):

        # GPT2 block hooks may return:
        #
        # tensor
        # tuple(hidden_states, ...)
        #

        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        self.activations.append(
            hidden_states.detach()
        )

    def clear(self):
        self.activations = []