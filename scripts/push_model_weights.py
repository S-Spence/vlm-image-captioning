from huggingface_hub import HfApi
import os

try:
    from dotenv import load_dotenv
except ImportError:
    import subprocess
    import sys
    print("python-dotenv not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

load_dotenv()
repo_id = os.getenv("HF_REPO")
weights_dir = "./model_weights"

if not repo_id:
    raise ValueError("HF_REPO not set in .env")

api = HfApi()

for filename in os.listdir(weights_dir):
    if filename.endswith(".pt"):
        local_path = os.path.join(weights_dir, filename)

        print(f"Uploading {filename} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model"
        )
