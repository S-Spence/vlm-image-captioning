from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch

class VicunaDecoder(torch.nn.Module):
    def __init__(self, debug=False, model="lmsys/vicuna-7b-v1.5"):
        super().__init__()
        self.debug = debug
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.is_lora_applied = False

        # freeze all model parameters until unfrozen for training
        for param in self.model.parameters():
            param.requires_grad = False

        if self.debug:
            print(f"Vicuna model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters.")

    def add_lora(self):
        """
        Add LoRA adapters—called when training_type='lora'.
        """
        if not self.is_lora_applied:
            config = LoraConfig(
                r=64, 
                lora_alpha=16, 
                target_modules=["q_proj", "v_proj"], 
                lora_dropout=0.05
            )
            self.model = get_peft_model(self.model, config)
            self.is_lora_applied = True
            if self.debug:
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"LoRA adapters added—trainable params: {trainable_params}")

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        return self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
