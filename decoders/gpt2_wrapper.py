import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention



# hugging face discussion on patching the model with cross attention:
# https://github.com/huggingface/transformers/issues/4483?utm_source=chatgpt.com
# Also referenced is_cross_attention implementation in: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
# referenced GPT2 cross attention implementation without llava: https://www.kaggle.com/code/shreydan/visiongpt2-image-captioning-pytorch
class GPT2Decoder(nn.Module):
    def __init__(self, cross_attention=True, gated_cross_attention=False, debug=False):
        super().__init__()
        self.debug = debug
        self.gated = gated_cross_attention
        self.cross_attention = cross_attention

        if self.cross_attention:
            config = GPT2Config.from_pretrained("gpt2", add_cross_attention=True)
            self.model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
            self._init_cross_attention_blocks()
            if self.debug:
                print("Added cross-attention to GPT-2 transformer blocks.")
        else:
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")

        # freeze all model parameters until unfrozen for training
        # leave new cross attention layers unfrozen if adding
        for name, param in self.model.named_parameters():
            if self.cross_attention and ('cross_attn' in name or 'ln_cross' in name or 'cross_gate' in name or 'lm_head' in name or 'ln_f' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

        if self.debug:
            attention_type = ("gated cross attention" if gated_cross_attention else "cross attention") if cross_attention else "no cross attention"
            print(f"GPT-2 initialized with {attention_type}, and trainable parameters {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def generate(self, *args, **kwargs):
        """Expose the generate method from GPT-2."""
        return self.model.generate(*args, **kwargs)
    
    def forward(self, input_ids=None, inputs_embeds=None, encoder_hidden_states=None, **kwargs):
        """
        Forward pass with optional cross-attention within transformer blocks.
        
        Parameters:
            input_ids: Tensor of input IDs.
            inputs_embeds: Tensor of input embeddings.
            encoder_hidden_states: Tensor of encoder hidden states for cross-attention.
            **kwargs: Additional arguments for the forward method.

        Returns:
            logits: Tensor of shape (batch_size, sequence_length, vocab_size) representing raw logits.
        """
        if self.cross_attention and encoder_hidden_states is not None:
            # patch forward method for cross-attention
            orig_forwards = []
            for block in self.model.transformer.h:
                orig_forwards.append(block.forward)
                block.forward = lambda hidden_states, block=block, **block_kwargs: self._cross_attention_forward(
                    block, hidden_states, **block_kwargs)
            
            outputs = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
            
            # restore original blocks
            for block, orig_forward in zip(self.model.transformer.h, orig_forwards):
                block.forward = orig_forward
            
            return outputs.logits
        else:
            # no changes needed to the forward method when cross attention is not applied
            return self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        
    def generate_with_cross_attention(self, **kwargs):

        original_model_forward = self.model.forward
        do_sample = kwargs.get("do_sample", False)
        num_beams = kwargs.get("num_beams", 1)
        input_ids = kwargs.get("input_ids")
        encoder_hidden_states = kwargs.get("encoder_hidden_states")

        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be provided in kwargs")

        if input_ids is None and "inputs_embeds" not in kwargs:
            raise ValueError("Must provide either input_ids or inputs_embeds for generation")

        # expand encoder_hidden_states to match beam search's expanded batch size
        if not do_sample and num_beams > 1:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_beams, dim=0)

        # patch the model's forward method to inject encoder_hidden_states
        self.model.forward = lambda input_ids=None, inputs_embeds=None, **inner_kwargs: original_model_forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            **inner_kwargs
        )

        try:
            outputs = self.model.generate(**kwargs)
        finally:
            # always restore the original forward method
            self.model.forward = original_model_forward

        return outputs

    def _init_cross_attention_blocks(self):
        """
        define additional layers for each transformer block
        """
        for block in self.model.transformer.h:
            block.cross_attn = GPT2Attention(self.model.config, is_cross_attention=True)
            block.ln_cross = nn.LayerNorm(self.model.config.n_embd)
            if self.gated:
                block.cross_gate = nn.Parameter(torch.ones(self.model.config.n_embd))
    
    def _cross_attention_forward(self, block, hidden_states, **kwargs):
        """
        Custom forward for transformer blocks with cross-attention.
        
        Parameters:
            block: Transformer block to modify.
            hidden_states: Tensor of shape (batch_size, seq_len, n_embd).
            **kwargs: Additional arguments for the forward method.

        Returns:
            Tuple of (hidden_states, attn_outputs).
        """
        # original self-attention block
        attn_outputs = block.attn(block.ln_1(hidden_states), **kwargs)
        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output

        # cross-attention added
        encoder_hidden_states = kwargs.get('encoder_hidden_states', None)
        if self.cross_attention and encoder_hidden_states is not None:
            cross_input = block.ln_cross(hidden_states)
            cross_output, _ = block.cross_attn(
                cross_input,
                encoder_hidden_states=encoder_hidden_states
            )
            if self.gated:
                gate = torch.tanh(block.cross_gate).reshape(1, 1, -1)
                cross_output = gate * cross_output
            # scale cross_output
            cross_output *= 1.5
            hidden_states = hidden_states + cross_output

        # MLP block
        feed_forward_hidden_states = block.mlp(block.ln_2(hidden_states))
        hidden_states = hidden_states + feed_forward_hidden_states

        return hidden_states, attn_outputs[1:]
