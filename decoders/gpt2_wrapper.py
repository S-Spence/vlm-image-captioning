import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

class GPT2Decoder(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

        # freeze all model parameters until unfrozen for training
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        if self.debug:
            print(f"GPT-2 initialized with trainable parameters {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def generate(self, *args, **kwargs):
        """Expose the generate method from GPT-2."""
        return self.model.generate(*args, **kwargs)
    
    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        """
        Forward pass for the GPT-2 model.
        
        Parameters:
            input_ids: Tensor of input IDs.
            inputs_embeds: Tensor of input embeddings.
            **kwargs: Additional arguments for the forward method.

        Returns:
            logits: Tensor of shape (batch_size, sequence_length, vocab_size) representing raw logits.
        """
        return self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
