"""
translate.py — Translate text using the NLLB-200 CTranslate2 model.

Usage:
    python translate.py

Prerequisites:
    Run `python setup_model.py` first to download and convert the model.

Steps performed:
    1. Loads the CTranslate2 translator from the converted model directory.
    2. Loads the tokenizer from the locally saved tokenizer directory.
    3. Tokenizes the source text and translates it to the target language.
    4. Decodes and prints the translated output.

To change source/target language, edit src_lang and tgt_lang below.
Common NLLB language codes:
    eng_Latn  English          fra_Latn  French
    spa_Latn  Spanish          deu_Latn  German
    zho_Hans  Chinese (Simpl)  arb_Arab  Arabic
    hin_Deva  Hindi            jpn_Jpan  Japanese
    rus_Cyrl  Russian          por_Latn  Portuguese
    urd_Arab  Urdu             ben_Beng  Bengali

Full list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
"""

import ctranslate2
import transformers

# ──────────────────────────────────────────────
#  Configuration — edit these values as needed
# ──────────────────────────────────────────────
MODEL_DIR = "nllb-200-ct2"
TOKENIZER_DIR = "nllb-200-ct2/tokenizer"
DEVICE = "cpu"          # "cpu" or "cuda"

SRC_LANG = "eng_Latn"   # source language code
TGT_LANG = "fra_Latn"   # target language code

TEXT = "Hello world!"    # text to translate


def translate(
    text: str,
    src_lang: str,
    tgt_lang: str,
    translator: ctranslate2.Translator,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> str:
    """Translate a single string from src_lang to tgt_lang."""
    tokenizer.src_lang = src_lang
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    results = translator.translate_batch(
        [tokens],
        target_prefix=[[tgt_lang]],
    )
    output_tokens = results[0].hypotheses[0][1:]
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))


if __name__ == "__main__":
    print("Loading model and tokenizer...")
    translator = ctranslate2.Translator(MODEL_DIR, device=DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        src_lang=SRC_LANG,
        local_files_only=True,
    )

    print(f"\n  Source ({SRC_LANG}): {TEXT}")
    result = translate(TEXT, SRC_LANG, TGT_LANG, translator, tokenizer)
    print(f"  Target ({TGT_LANG}): {result}\n")
