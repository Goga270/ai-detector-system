"""Скачать приватный датасет с Hugging Face Hub в data/token_detector_bio/hf_dataset.

Нужен токен с правом Read: export HF_TOKEN=hf_... перед запуском."""
from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import login

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_ID = "mandal437/bert-token-detector-bio"
HF_DATASET_DIR = REPO_ROOT / "data" / "token_detector_bio" / "hf_dataset"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "Задай HF_TOKEN (Read достаточно для скачивания приватного датасета)."
        )

    os.environ["HF_TOKEN"] = token
    login(token=token, add_to_git_credential=False)

    print(f"[HF] Загрузка {REPO_ID} ...")
    ds = load_dataset(REPO_ID, token=True)
    print(ds)

    HF_DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    if HF_DATASET_DIR.exists():
        print(f"[HF] Перезапись {HF_DATASET_DIR}")
    ds.save_to_disk(str(HF_DATASET_DIR))
    print(f"[HF] OK: {HF_DATASET_DIR}")


if __name__ == "__main__":
    main()
