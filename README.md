# NLLB-200 CTranslate2 Translator

Offline machine translation using Meta's [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) model, accelerated by [CTranslate2](https://github.com/OpenNMT/CTranslate2).

Supports **200 languages** out of the box.

---

## Prerequisites

- Python 3.9+
- A conda or virtual environment (recommended)

Install dependencies:

```bash
pip install ctranslate2 transformers sentencepiece
```

> **GPU users:** also install CUDA-compatible PyTorch (`pip install torch` with CUDA support).

---

## Quick Start

### Step 1 — Convert the model

```bash
python setup_model.py
```

This will:
1. Download the NLLB-200-distilled-600M weights from HuggingFace.
2. Convert them to CTranslate2 format (default: `int8` quantization).
3. Save the tokenizer locally for offline use.

Output is written to `nllb-200-ct2/`.

### Step 2 — Translate

```bash
python translate.py
```

This will:
1. Load the converted model and tokenizer from disk.
2. Translate the configured text and print the result.

Edit the configuration section at the top of `translate.py` to change the source text, source/target languages, or device.

---

## Configuration

### Quantization (in `setup_model.py`)

| Option          | Best for | Speed  | Size    | Quality   |
|-----------------|----------|--------|---------|-----------|
| `int8`          | CPU      | Fast   | Smallest | Very good |
| `float16`       | GPU      | Fast   | Small    | Excellent |
| `int8_float16`  | GPU      | Fast   | Smallest | Very good |
| `float32`       | Any      | Slow   | Largest  | Best      |

### Language codes (in `translate.py`)

| Code        | Language              |
|-------------|-----------------------|
| `eng_Latn`  | English               |
| `fra_Latn`  | French                |
| `spa_Latn`  | Spanish               |
| `deu_Latn`  | German                |
| `zho_Hans`  | Chinese (Simplified)  |
| `arb_Arab`  | Arabic                |
| `hin_Deva`  | Hindi                 |
| `jpn_Jpan`  | Japanese              |
| `rus_Cyrl`  | Russian               |
| `por_Latn`  | Portuguese            |
| `urd_Arab`  | Urdu                  |
| `ben_Beng`  | Bengali               |

Full list of 200 language codes: [FLORES-200 README](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)

---

## Project Structure

```
ctranslate/
├── setup_model.py      # Step 1: Download, convert, and save the model
├── translate.py         # Step 2: Load the model and translate text
├── README.md            # This file
└── nllb-200-ct2/        # Generated after setup (not committed)
    ├── model.bin
    ├── vocabulary.json
    └── tokenizer/
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `float16` warning on CPU | Use `int8` quantization instead — float16 has no CPU benefit |
| `TokenizersBackend` error | Run `setup_model.py` and `translate.py` in the **same** conda environment |
| Slow first run | Initial download is ~1.2 GB; subsequent runs load from disk |
| Out of memory | Use `int8` quantization for the smallest memory footprint |

---

## License

The NLLB-200 model is released by Meta under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
CTranslate2 is licensed under [MIT](https://github.com/OpenNMT/CTranslate2/blob/master/LICENSE).
