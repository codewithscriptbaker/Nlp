"""
setup_model.py — Download, convert, and prepare NLLB-200 for CTranslate2 inference.

Usage:
    python setup_model.py

Steps performed:
    1. Downloads the HuggingFace NLLB-200-distilled-600M model weights.
    2. Converts them to CTranslate2 format with the chosen quantization.
    3. Saves the tokenizer alongside the model for offline use.

After running this script, use translate.py to perform translations.

Quantization options (set QUANTIZATION below):
    - int8          Best for CPU — real speed gains and smallest model size.
    - float16       Best for GPU (CUDA) — fast half-precision inference.
    - int8_float16  Hybrid — int8 weights with float16 compute (GPU only).
    - float32       Full precision — highest quality, largest/slowest.
"""

import os
import ctranslate2
import transformers


# ──────────────────────────────────────────────
#  Configuration — edit these values as needed
# ──────────────────────────────────────────────
MODEL_NAME = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR = "nllb-200-ct2"
QUANTIZATION = "int8"  # Recommended: "int8" for CPU, "float16" for GPU


def convert_model(model_name: str, output_dir: str, quantization: str) -> str:
    """Convert a HuggingFace NLLB model to CTranslate2 format.

    Returns the absolute path to the converted model directory.
    """
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)

    if os.path.exists(output_path) and os.listdir(output_path):
        print(f"[skip] '{output_path}' already exists. Delete it to re-convert.")
        return output_path

    print(f"  Model   : {model_name}")
    print(f"  Quant   : {quantization}")
    print(f"  Output  : {output_path}")
    print("  This may take several minutes (download + conversion)...\n")

    converter = ctranslate2.converters.TransformersConverter(
        model_name_or_path=model_name,
    )
    converter.convert(
        output_dir=output_path,
        quantization=quantization,
        force=True,
    )
    print(f"\n[done] Model saved to: {output_path}")
    return output_path


def save_tokenizer(model_name: str, output_dir: str) -> str:
    """Save the tokenizer locally so translate.py works fully offline."""
    tokenizer_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), output_dir, "tokenizer"
    )

    if os.path.exists(tokenizer_dir) and os.listdir(tokenizer_dir):
        print(f"[skip] Tokenizer already saved at '{tokenizer_dir}'.")
        return tokenizer_dir

    print(f"  Saving tokenizer to: {tokenizer_dir}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tokenizer_dir)
    print("[done] Tokenizer saved.\n")
    return tokenizer_dir


if __name__ == "__main__":
    print("=" * 55)
    print("  NLLB-200 — CTranslate2 Model Setup")
    print("=" * 55 + "\n")

    print("[Step 1/2] Converting model...\n")
    convert_model(MODEL_NAME, OUTPUT_DIR, QUANTIZATION)

    print("\n[Step 2/2] Saving tokenizer...\n")
    save_tokenizer(MODEL_NAME, OUTPUT_DIR)

    print("=" * 55)
    print("  Setup complete. Run `python translate.py` to translate.")
    print("=" * 55)
