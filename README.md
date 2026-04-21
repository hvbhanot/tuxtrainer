# tuxtrainer

Fine-tune small LLMs on your PDFs and push to **Ollama** — optimised for Google Colab with **Unsloth** (~2× faster training).

```
PDFs → extract & chunk → master model picks hyperparams → Unsloth QLoRA → GGUF → Ollama registry
```

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
)

pipeline = FinetunePipeline(config)
pipeline.run()
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

# Push an existing GGUF
$ tuxtrainer push --gguf model.gguf --name my-expert --namespace myuser

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

## Quantisation

| Level | Size | Quality |
|-------|------|---------|
| `Q4_K_M` | Smallest | Good |
| `Q5_K_M` | Small | Better |
| `Q6_K` | Medium | Very good |
| `Q8_0` | Large | Excellent |
| `F16` | Largest | Perfect |

---

## License

MIT
