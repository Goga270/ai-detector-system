from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[3]
HF_DATASET_DIR = REPO_ROOT / "data" / "token_detector_bio" / "hf_dataset"
REPO_ID = "mandal437/bert-token-detector-bio"


def _require_hf_token() -> None:
    try:
        HfApi().whoami()
    except Exception as e:
        raise SystemExit(
            "Hugging Face: нет авторизации (ожидается токен с правом записи → 401).\n\n"
            "Варианты:\n"
            "  1) В терминале: huggingface-cli login\n"
            "  2) Переменная окружения: export HF_TOKEN=hf_...\n"
            "     (создать: https://huggingface.co/settings/tokens — тип с Write)\n\n"
            f"Детали: {e}"
        ) from e


def main() -> None:
    _require_hf_token()
    if not HF_DATASET_DIR.is_dir():
        raise SystemExit(
            f"Нет локального датасета: {HF_DATASET_DIR}\n"
            "Сначала запусти prepare_bio_dataset.py с --save-hf-dataset."
        )
    ds = load_from_disk(str(HF_DATASET_DIR))
    ds.push_to_hub(REPO_ID, private=True)
    print(f"OK: загружено в https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()