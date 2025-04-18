import os
import argparse
import shutil
from huggingface_hub import hf_hub_download, HfApi

try:
    from dotenv import load_dotenv
except ImportError:
    import subprocess
    import sys
    print("python-dotenv not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

load_dotenv()
REPO_ID = os.getenv("HF_REPO")

def get_timestamp(file):
    name_part = os.path.splitext(file)[0]
    *_, timestamp = name_part.rsplit('_', 1)
    return timestamp

def get_latest_file(prefix):
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")

    matching = [f for f in files if f.startswith(prefix) and f.endswith(".pt")]
    if not matching:
        raise ValueError(f"No files found for prefix: {prefix}")

    latest = sorted(matching, key=get_timestamp)[-1]
    return latest

def download_model(prefix, out_dir):
    print(f"Using repo: {REPO_ID}")
    print(f"Searching for latest weights with prefix '{prefix}'...")
    filename = get_latest_file(prefix)
    print(f"Downloading: {filename}")

    path = hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir_use_symlinks=False)
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, filename)
    shutil.copy2(path, target)
    print(f"Saved to: {target}")
    return target

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="example 'experiment_1'")
    parser.add_argument("--output_dir", default="model_weights", help="Directory to save the model weights")
    args = parser.parse_args()

    download_model(args.prefix, args.output_dir)
