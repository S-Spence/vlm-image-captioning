from PIL import Image
import requests
from io import BytesIO
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, ViTModel, AutoImageProcessor
from PIL import Image
from typing import List, Union

class ImageEncoder(nn.Module):
    """
    Generic image encoder that supports ViT and CLIP encoders.
    Returns raw patch embeddings (without LLaVA-style projections).

    Parameters:
        model_type (str): 'vit' or 'clip'
        model_name (str): HuggingFace model name
        use_cls_token (bool): Whether to retain the [CLS] token (default: False)
        debug (bool): Print shapes for debugging
    """
    def __init__(self, model_type="vit", model_name=None, use_cls_token=False, debug=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")

        self.model_type = model_type
        self.use_cls_token = use_cls_token
        self.debug = debug

        if model_type == "vit":
            model_name = model_name or "google/vit-base-patch16-224-in21k"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = ViTModel.from_pretrained(model_name).to(self.device)
        elif model_type == "clip":
            model_name = model_name or "openai/clip-vit-base-patch16"
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).vision_model.to(self.device)

        else:
            raise ValueError("model_type must be either 'vit' or 'clip'")

        # freeze params for vision encoder
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images: Union[List[Image.Image], Image.Image]):
        if not isinstance(images, list):
            images = [images]

        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        outputs = self.model(pixel_values)

        last_hidden_state = outputs.last_hidden_state

        if not self.use_cls_token:
            # drop CLS token
            last_hidden_state = last_hidden_state[:, 1:]

        if self.debug:
            print(f"Patch embeddings shape: {last_hidden_state.shape}")

        return last_hidden_state

if __name__ == '__main__':
    url = "https://upload.wikimedia.org/wikipedia/commons/9/94/White_horse_in_field.jpg"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    encoder = ImageEncoder(debug=True, model_type="vit", use_cls_token=False)
    output = encoder(img)
    print("Output shape vit:", output.shape)

    encoder = ImageEncoder(debug=True, model_type="vit", use_cls_token=True)
    output = encoder(img)
    print("Output shape vit with CLS:", output.shape)

    encoder = ImageEncoder(debug=True, model_type="clip", use_cls_token=False)
    output = encoder(img)
    print("Output shape clip:", output.shape)

    encoder = ImageEncoder(debug=True, model_type="clip", use_cls_token=True)
    output = encoder(img)
    print("Output shape clip with CLS:", output.shape)
