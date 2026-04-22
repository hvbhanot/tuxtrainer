# tuxtrainer

Fine-tune small LLMs on your PDFs and push to **Ollama** — optimised for Google Colab with **Unsloth** (~2× faster training).

```
PDFs → extract & chunk → master model picks hyperparams → Unsloth QLoRA → GGUF → Ollama registry
```

The GGUF conversion is done by Unsloth's native `save_pretrained_gguf`, which merges the LoRA adapter, dequantizes the 4-bit base, runs llama.cpp internally, and quantizes — all in one call. The pipeline produces a **single `.gguf` file** in `finetune_output/gguf/`; no intermediate fp16 checkpoint is saved.

Pull your model on any device:

```bash
ollama pull yourname/my-model
ollama run yourname/my-model
```

---

## Quick Start (Google Colab)

### 1. Install

```python
!pip install git+https://github.com/hvbhanot/tuxtrainer.git
```

**Restart the runtime** after install (`Runtime → Restart session` or `Ctrl+M .`).

### 2. Setup environment

```python
from tuxtrainer.colab import setup_colab
setup_colab(pull_master_model="llama3.1")
```

`setup_colab()` installs and starts Ollama for the push step and keeps the Python dependency set on the Unsloth-compatible versions pinned by `tuxtrainer`. It does not manually run any `llama.cpp` conversion scripts; Unsloth handles that during GGUF export.

### 3. Configure & run

```python
import os
from pathlib import Path
from tuxtrainer import FinetuneConfig, FinetunePipeline

# Optional: set these in Colab secrets instead
os.environ["OLLAMA_NAMESPACE"] = "your-ollama-username"

config = FinetuneConfig(
    model_id="unsloth/Llama-3.2-1B-Instruct",
    pdf_paths=[Path("my_document.pdf")],
    master_backend="ollama",   # uses local Ollama, no API key needed
    master_model="llama3.1",
    use_unsloth=True,          # default; keeps the direct in-memory GGUF path
)

pipeline = FinetunePipeline(config)
pipeline.run()
# → exactly one .gguf file in finetune_output/gguf/
# → ollama pull your-ollama-username/<your-model>-finetuned
```

### Gated models

If you use a gated model (e.g. `meta-llama/...`), you need a HuggingFace token:

1. Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Accept the model's license on its HuggingFace page
3. Set it in Colab secrets or via `os.environ["HF_TOKEN"] = "..."`

---

## Master Model Backends

The master model automatically picks hyperparameters. Choose whichever backend you have access to:

| Backend | Config | Needs API Key |
|---------|--------|:-------------:|
| **Local Ollama** | `master_backend="ollama"` | None |
| **OpenAI** | `master_backend="openai"` | `OPENAI_API_KEY` |
| **Ollama Cloud** | `master_backend="ollama_cloud"` | `OLLAMA_API_KEY` |
| **HuggingFace** | `master_backend="hf_api"` | `HF_TOKEN` |

**Local Ollama** is the easiest on Colab — it's already installed by `setup_colab()`.

---

## CLI

```bash
# Full pipeline
$ tuxtrainer run --model unsloth/Llama-3.2-1B-Instruct --pdf doc.pdf --ollama-namespace myuser

# Local only (no registry push)
$ tuxtrainer run --model unsloth/Llama-3.2-1B-Instruct --pdf doc.pdf --no-ollama-push

# Skip Ollama entirely — just get the GGUF file
$ tuxtrainer run --model unsloth/Llama-3.2-1B-Instruct --pdf doc.pdf --skip-ollama

# Export an existing adapter to GGUF (Unsloth does merge + convert + quantize)
$ tuxtrainer export --adapter-path ./finetune_output/final_adapter \
    --model unsloth/Llama-3.2-1B-Instruct --quant q4_k_m

# Push an existing GGUF (accepts a file OR a directory containing one)
$ tuxtrainer push --gguf ./finetune_output/gguf --name my-expert --namespace myuser

# Prep dataset from PDFs only
$ tuxtrainer prep --pdf-dir ./documents/ --output dataset.jsonl
```

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace token (for gated models) |
| `OLLAMA_NAMESPACE` | Your Ollama registry username |
| `OLLAMA_API_KEY` | Ollama Cloud API key |
| `OPENAI_API_KEY` | OpenAI API key |

---

## Quantization

Quantization values are passed straight to Unsloth's `save_pretrained_gguf`.

| Level | Size | Quality |
|-------|------|---------|
| `q4_k_m` | Smallest | Good (default) |
| `q5_k_m` | Small | Better |
| `q6_k` | Medium | Very good |
| `q8_0` | Large | Excellent |
| `f16` | Largest | Perfect |

Legacy uppercase values (`Q4_K_M`, ...) are still accepted — they are normalized to lowercase at config-parse time. The old `quantisation=` config kwarg is also accepted for one release cycle and maps to `quantization=` with a `DeprecationWarning`.

---

## Why Unsloth for the GGUF step?

transformers 5.x introduced a `ConversionOps` system in `core_model_loading.py` whose bitsandbytes dequantize op does not implement `reverse_op`. That breaks `save_pretrained` on merged 4-bit models with `NotImplementedError`. Unsloth's `save_pretrained_gguf` runs its own merge + dequant + llama.cpp pipeline inside `unsloth_zoo.saving_utils` and bypasses the broken path entirely. See [unslothai/unsloth#4832](https://github.com/unslothai/unsloth/issues/4832) for background.

tuxtrainer pins `transformers>=4.56,<5.0` until the upstream fix ships in a released version.

---

## License

MIT
