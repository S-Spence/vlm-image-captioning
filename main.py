# main.py
from flask import Flask, request, jsonify
import torch
from PIL import Image
from transformers import AutoTokenizer
import os
from vison_language_model import VisionLanguageModel

# Define model-related paths and parameters.
MODEL_PATH = "path/to/your/saved_model.pt"      # path to the whole model file saved from training
TOKENIZER_NAME = "gpt2"  # or the name of your tokenizer (could be a local path)

# Load tokenizer and model.
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
# If needed, set special tokens; for GPT-2, you might set pad_token to eos_token
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Instantiate the VisionLanguageModel with the same parameters as during training.
model = VisionLanguageModel(
    tokenizer=tokenizer,
    model_config=None,
    image_encoder_type="vit",
    decoder_type="gpt2",
    llava_projections=True,
    cross_attention=True,
    gated_cross_attention=True,
    debug=False,
    d_model=768,
    model_path=None
)

# Load the saved model weights into the model.
state_dict = torch.load(MODEL_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.load_state_dict(state_dict)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

# Create the Flask app.
app = Flask(__name__)

@app.route("/caption", methods=["POST"])
def predict():
    # Check if image is provided in the request.
    if "image" not in request.files:
        return jsonify({"error": "No image file provided in the request."}), 400

    file = request.files["image"]
    
    try:
        # Open the image and convert to RGB.
        image = Image.open(file).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Unable to open image: {str(e)}"}), 400

    # Run prediction. The model.predict method expects a list of images.
    try:
        caption_list = model.predict([image], max_new_tokens=15)
        caption = caption_list[0]
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    return jsonify({"caption": caption})

if __name__ == "__main__":
    # Run the Flask app; adjust host and port as needed.
    app.run(host="0.0.0.0", port=5000)
