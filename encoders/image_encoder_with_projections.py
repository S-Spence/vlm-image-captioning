import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ViTModel, CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO

# feature extraction based on llava impementation: 
# https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_encoder/clip_encoder.py
class LLavaProjectionEncoder(nn.Module):
    """
    Image encoder with MLP projection for LLaVA-style image-to-text embedding alignment.
    
    Parameters:
        d_model (int): Target embedding dimension for the language model.
        model_name (str): Hugging Face model name.
        model_type (str): 'vit' or 'clip'.
        use_cls_token (bool): Whether to keep the [CLS] token.
        debug (bool): Print shape information for debugging.
    """
    def __init__(self, d_model=512, debug=False, model_name=None, model_type="vit", use_cls_token=False):
        super().__init__()
        self.d_model = d_model
        self.debug = debug
        self.encoder_type = model_type
        self.use_cls_token = use_cls_token
        self.device = self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")


        if self.encoder_type == "vit":
            model_name = model_name or "google/vit-base-patch16-224-in21k"
            self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
            self.model = ViTModel.from_pretrained(model_name).to(self.device)
        elif self.encoder_type == "clip":
            model_name = model_name or "openai/clip-vit-base-patch16"
            self.feature_extractor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).vision_model.to(self.device)
        else:
            raise ValueError("model_type must be either 'vit' or 'clip'")
        
        self.num_image_tokens = self._get_num_image_tokens()

        # freeze the pretrained model
        for param in self.model.parameters():
            param.requires_grad = False

        # MLP for image projections
        # used GeLU activation as in the original LLaVA paper
        hidden_dim = self.model.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        ).to(self.device)

    def _get_num_image_tokens(self):
        """
        Calculate the number of image tokens based on model config
        
        Returns:
            int: Number of image tokens (including CLS token if applicable)
        """
        
        image_size = self.model.config.image_size  
        patch_size = self.model.config.patch_size  
        
        num_patches = (image_size // patch_size) ** 2  
        return num_patches + 1 if self.use_cls_token else num_patches

    def forward(self, images):
        if not isinstance(images, list):
            images = [images]

        # convert images to tensors and move to device
        img_tensor = self.feature_extractor(images=images, return_tensors="pt").pixel_values.to(self.device)

        if self.encoder_type == "clip":
            outputs = self.model(pixel_values=img_tensor, output_hidden_states=True)
        else:
            outputs = self.model(img_tensor)

        # maybe drop the [CLS] token
        if self.use_cls_token:
            features = outputs.last_hidden_state
        else:
            features = outputs.last_hidden_state[:, 1:]

        if self.debug:
            print("Image features shape:", features.shape)

        # pass the image features through the MLP projection
        projected = self.projection(features)

        if self.debug:
            print("Projected image embeddings shape:", projected.shape)

        return projected

if __name__ == '__main__':
    url = "https://upload.wikimedia.org/wikipedia/commons/9/94/White_horse_in_field.jpg"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    encoder = LLavaProjectionEncoder(d_model=512, debug=True, model_type="vit", use_cls_token=False)
    output = encoder(img)
    print("Output shape vit:", output.shape)

    encoder = LLavaProjectionEncoder(d_model=512, debug=True, model_type="vit", use_cls_token=True)
    output = encoder(img)
    print("Output shape vit with CLS:", output.shape)

    encoder = LLavaProjectionEncoder(d_model=512, debug=True, model_type="clip", use_cls_token=False)
    output = encoder(img)
    print("Output shape clip:", output.shape)

    encoder = LLavaProjectionEncoder(d_model=512, debug=True, model_type="clip", use_cls_token=True)
    output = encoder(img)
    print("Output shape clip with CLS:", output.shape)
