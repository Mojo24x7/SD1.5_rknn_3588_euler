#!/usr/bin/env python3
from huggingface_hub import snapshot_download
from pathlib import Path

HF_REPO_ID = "Mojo24x7/SD1.5_rknn_3588_euler"
HF_REPO_TYPE = "dataset"

BASE_DIR = Path(__file__).resolve().parents[1]

def main():
    print(f"[fetch_assets] Downloading assets from {HF_REPO_ID}")

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        local_dir=BASE_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print("[fetch_assets] Done.")
    print("Assets are now available under:")
    print(f"  {BASE_DIR}/models")
    print(f"  {BASE_DIR}/model")
    print(f"  {BASE_DIR}/gfpgan/weights")

if __name__ == "__main__":
    main()
