import torch


class ActivationPatcher:

    def __init__(self, replacement_vector):

        self.replacement_vector = replacement_vector

    def hook_fn(self, module, inputs, outputs):

        #
        # GPT2 block output handling
        #

        if isinstance(outputs, tuple):

            hidden_states = outputs[0]

            hidden_states = hidden_states.clone()

            #
            # Replace LAST TOKEN activation
            #

            hidden_states[:, -1, :] = (
                self.replacement_vector
            )

            #
            # Rebuild tuple
            #

            outputs = (hidden_states,) + outputs[1:]

            return outputs

        else:

            outputs = outputs.clone()

            outputs[:, -1, :] = (
                self.replacement_vector
            )

            return outputs