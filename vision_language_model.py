import torch
import torch.nn as nn

from encoders.image_encoder import ImageEncoder
from encoders.image_encoder_with_projections import LLavaProjectionEncoder
from decoders.gpt2_wrapper import GPT2Decoder
from decoders.vicuna_wrapper import VicunaDecoder

class VisionLanguageModel(nn.Module):
    def __init__(self, tokenizer, image_encoder_type="vit", decoder_type="gpt2", llava_projections=False,
                 cross_attention=True, gated_cross_attention=True, debug=False, d_model=768):
        """
        Params:
            tokenizer: HuggingFace tokenizer.
            image_encoder_type (str): type of image encoder to use ('vit' or 'clip')
            decoder_type (str): type of decoder to use ('gpt2', 'vicuna')
            llava_projections (bool): use LLaVA-style projections
            cross_attention (bool): use cross attention in the decoder
            gated_cross_attention (bool): use gated cross attention in the decoder
            debug (bool): print debug information
            d_model (int): target embedding dimension for the language model
        """
        super().__init__()
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        
        self.tokenizer = tokenizer
        self.llava_projections = llava_projections
        self.cross_attention = cross_attention
        self.gated_cross_attention = gated_cross_attention
        self.decoder_type = decoder_type

        # initialize the image encoder
        self.image_encoder = self._init_image_encoder(image_encoder_type, d_model)

        # initialize the decoder
        self.decoder = self._init_model(decoder_type)

    def _init_image_encoder(self, image_encoder_type: str, d_model: int) -> nn.Module:
        """
        Initialize the image encoder based on type and projection config.
        """
        if self.llava_projections:
            return LLavaProjectionEncoder(
                d_model=d_model,
                model_type=image_encoder_type,
                debug=self.debug
            )
        else:
            return ImageEncoder(
                model_type=image_encoder_type,
                use_cls_token=False,
                debug=self.debug
            )
        
    def _init_model(self, model_type):
        """
        Initialize the model based on the specified configuration.
        Optionally load from pretrained weights if path is provided.

        Parameters:
            model_type (str): the decoder type

        Returns:
            The initialized GPT model (GPT2Decoder or VicunaDecoder).
        """
        if model_type == "gpt2":
            print("Initializing GPT-2 model...")
            gpt = GPT2Decoder(
                cross_attention=self.cross_attention,
                gated_cross_attention=self.gated_cross_attention
            ).to(self.device)
            return gpt
        if model_type == "vicuna":
            print("Initializing Vicuna model...")
            return VicunaDecoder(debug=self.debug).to(self.device)
        else:
            raise ValueError("Unsupported model type. Choose 'gpt2' or 'vicuna'.")
    
    def forward(self, images, tokens):
        """
        Forward pass for the VisionGPTModel.
        
        Parameters:
            images (long tensor): A batch of image tensors (B, 3, H, W)
            tokens (long tensor): Token IDs of shape (B, sequence_length)
        
        Returns:
          logits: A tensor of shape (B, sequence_length, vocab_size) representing raw logits.
        """
        tokens = tokens.to(self.device)
        # get image embeddings from the image encoder.
        image_embeddings = self.image_encoder(images).to(self.device)
        if self.decoder_type == "vicuna":
            # cast to float16 for vicuna
            image_embeddings = image_embeddings.to(torch.float16)
        
        if self.debug:
            print("Image embeddings shape:", image_embeddings.shape)
        
        if not self.cross_attention:
            # prepend text tokens to inputs for regular llava approach
            input_embeds = self.decoder.get_input_embeddings()(tokens)

            prepended = torch.cat([image_embeddings, input_embeds], dim=1)

            output = self.decoder(input_ids=None, inputs_embeds=prepended)
        else:
            # use cross attention
            output = self.decoder(tokens, encoder_hidden_states=image_embeddings)

        # different models have different output shapes, check for logits or get them
        logits = output if isinstance(output, torch.Tensor) else output.logits
        return logits

    def predict(self, batch_images, max_new_tokens=15, do_sample=False, top_p=0.9, temperature=1.0, n_beams=1):
        """
        Predict captions for a list of images, optionally using top-p sampling.
        
        Parameters:
            batch_images (List[Image or Tensor]): List of image tensors or PIL images
            max_new_tokens (int): Maximum number of tokens to generate (default: 15)
            do_sample (bool): Whether to use sampling or beam search (default: False)
            top_p (float): Top-p sampling parameter (default: 0.9)
            temperature (float): Sampling temperature (default: 1.0)
            n_beams (int): Number of beams for beam search (default: 1 for greedy sampling)
            
        Returns:
            List[str]: List of predicted captions.
        """
        self.eval()
        
        if self.cross_attention:
            # Use a simple prompt: a tensor of shape [batch_size, 1] with BOS token.
            input_ids = torch.full((len(batch_images), 1), self.tokenizer.bos_token_id, 
                                    dtype=torch.long, device=self.device)
        else:
            # prepend "Caption:" as the prompt.
            input_ids = self.tokenizer(["Caption:"] * len(batch_images), 
                                    return_tensors="pt", 
                                    padding=True, truncation=True).input_ids.to(self.device)
        
        # setup common params
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "no_repeat_ngram_size": 2,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            kwargs["do_sample"] = do_sample
            kwargs["top_p"] = top_p
            kwargs["temperature"] = temperature
        else:
            kwargs["num_beams"] = n_beams
            kwargs["early_stopping"] = True
            kwargs["length_penalty"] = 0.6

        # generate captions with optional cross attention
        if self.cross_attention:
            attention_mask = torch.ones(input_ids.shape, device=self.device)
            kwargs["input_ids"] = input_ids
            kwargs["encoder_hidden_states"] = self.image_encoder(batch_images)
            kwargs["attention_mask"] = attention_mask

            outputs = self.decoder.generate_with_cross_attention(**kwargs)
        else:
            image_embeddings = self.image_encoder(batch_images)
            token_embeddings = self.decoder.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([image_embeddings, token_embeddings], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device)
            kwargs["inputs_embeds"] = inputs_embeds
            kwargs["attention_mask"] = attention_mask
            
            outputs = self.decoder.generate(**kwargs)
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
