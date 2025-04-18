import argparse
from flask import Flask, request, jsonify
import torch
from PIL import Image
from transformers import AutoTokenizer
from vision_language_model import VisionLanguageModel
import os
from peft import set_peft_model_state_dict

app = Flask(__name__)

def load_model(args):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # initialize model
    model = VisionLanguageModel(
        tokenizer=tokenizer,
        image_encoder_type=args.image_encoder_type,
        decoder_type=args.decoder_type,
        llava_projections=args.llava_projections,
        cross_attention=args.cross_attention,
        gated_cross_attention=args.gated_cross_attention,
        debug=False,
        d_model=args.d_model
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if args.train_type == "lora":
        model.decoder.add_lora()
        state_dict = torch.load(args.model_path, map_location=device)
        # load LoRA weights into the decoder
        set_peft_model_state_dict(model.decoder.model, state_dict["lora"])
        # load MLP weights into the image encoder
        model.image_encoder.projection.load_state_dict(state_dict["mlp"])

    elif args.train_type == "mlp-pretrain":
        model.image_encoder.projection.load_state_dict(torch.load(args.model_path, map_location=device))

    else:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model

@app.route("/caption", methods=["POST"])
def caption():
    data = request.get_json()
    image_path = data.get("image_path")

    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Valid image_path must be provided."}), 400

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Unable to open image: {str(e)}"}), 400

    try:
        caption_list = app.model.predict([image], max_new_tokens=15, n_beams=2)
        caption = caption_list[0]
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    return jsonify({"caption": caption})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to saved model weights (state_dict)")
    parser.add_argument("--tokenizer_name", default="gpt2", help="Tokenizer name or path")
    parser.add_argument("--image_encoder_type", default="vit")
    parser.add_argument("--decoder_type", default="gpt2")
    parser.add_argument("--llava_projections", action="store_true")
    parser.add_argument("--cross_attention", action="store_true")
    parser.add_argument("--gated_cross_attention", action="store_true")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--train_type", choices=["finetune", "mlp-pretrain", "lora"], default="finetune")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app.model = load_model(args)
    app.run(host="0.0.0.0", port=5000)
