# tuxtrainer

Fine-tune HuggingFace models with PDFs and **push to Ollama** — with **LLM-driven automatic hyperparameter selection** via the **Ollama Web API**.

## How It Works

```
PDFs → extract & chunk → master model picks hyperparams → LoRA/QLoRA fine-tune → merge → GGUF → push to Ollama registry
                            ↑                                                              ↓
                     Ollama Web API                                           pull on any device!
                     (POST /api/chat)                                    ollama pull yourname/model
```

**By default, the model is pushed to the Ollama registry** so you can use it on any device — your laptop, phone, another server — just `ollama pull yourname/model-name`.

## Quick Start on Colab

### 1. Install & setup

```python
!pip install -e /content/tuxtrainer

from tuxtrainer.colab import setup_colab
setup_colab()  # Installs Ollama + llama.cpp (Ollama needed for push step)
```

### 2. Set your API keys

```python
import os

# For the master model (picks hyperparameters via Ollama Cloud)
os.environ["OLLAMA_API_KEY"] = "your-ollama-api-key"

# For pushing to the Ollama registry (your username)
os.environ["OLLAMA_NAMESPACE"] = "your-ollama-username"
```

### 3. Fine-tune & push

```python
from tuxtrainer import FinetuneConfig, FinetunePipeline
from tuxtrainer.config import FinetuneMethod, Quantisation
from pathlib import Path

config = FinetuneConfig(
    model_id="meta-llama/Llama-3.1-8B",
    method=FinetuneMethod.QLORA,
    pdf_paths=[Path("my_document.pdf")],
    # ollama_push=True is the default — model is pushed to registry
    # ollama_namespace is read from OLLAMA_NAMESPACE env var
)

pipeline = FinetunePipeline(config)
result = pipeline.run()
# → Model pushed! Pull on any device: ollama pull yourname/llama-3.1-8b-finetuned
```

### 4. Use it anywhere

```bash
# On your laptop, phone, any device with Ollama:
ollama pull yourname/llama-3.1-8b-finetuned
ollama run yourname/llama-3.1-8b-finetuned
```

## Ollama Registry Push

The default behaviour is to **push the model to the Ollama registry** so it's available on any device. This requires:

1. **Ollama namespace** — your Ollama username (set via `OLLAMA_NAMESPACE` env var or `--ollama-namespace`)
2. **Ollama running locally** — needed to create the model before pushing (auto-installed on Colab)

| Scenario | Config | Result |
|----------|--------|--------|
| Push to registry (default) | `ollama_namespace="myuser"`, `ollama_push=True` | `ollama pull myuser/model` on any device |
| Local only | `ollama_namespace=None` | Model only on this machine: `ollama run model` |
| Skip Ollama entirely | `skip_ollama=True` | Just get the GGUF file |

## Master Model Backends

| Backend | Config Value | Needs local install? | API key env var |
|---------|-------------|:---:|---|
| **Ollama Web API** | `ollama_cloud` | ❌ | `OLLAMA_API_KEY` |
| **OpenAI** | `openai` | ❌ | `OPENAI_API_KEY` |
| **HuggingFace** | `hf_api` | ❌ | `HF_TOKEN` |
| **ZAI SDK** | `zai_sdk` | ❌ | `ZAI_API_KEY` |
| **Local Ollama** | `ollama` | ✅ | None |

**`ollama_cloud` is the default** for the master model — it uses the native Ollama Web API over HTTPS with Bearer token authentication. No extra packages needed.

## CLI Usage

```bash
# Full pipeline with registry push (default)
tuxtrainer run \
  --model meta-llama/Llama-3.1-8B \
  --pdf doc.pdf \
  --ollama-namespace myuser
# → ollama pull myuser/llama-3.1-8b-finetuned

# With explicit API keys
tuxtrainer run \
  --model meta-llama/Llama-3.1-8B \
  --pdf doc.pdf \
  --ollama-namespace myuser \
  --master-api-key sk-xxxxx

# Local only (no registry push)
tuxtrainer run \
  --model meta-llama/Llama-3.1-8B \
  --pdf doc.pdf \
  --no-ollama-push
# → ollama run llama-3.1-8b-finetuned

# Skip Ollama entirely (just get GGUF)
tuxtrainer run \
  --model meta-llama/Llama-3.1-8B \
  --pdf doc.pdf \
  --skip-ollama

# Use a different master model backend
tuxtrainer run \
  --model meta-llama/Llama-3.1-8B \
  --pdf doc.pdf \
  --ollama-namespace myuser \
  --master-backend openai

# Push an existing GGUF to Ollama
tuxtrainer push --gguf model.gguf --name my-expert --namespace myuser

# Just process PDFs
tuxtrainer prep --pdf-dir ./documents/ --output dataset.jsonl

# Fine-tune from a dataset
tuxtrainer train --model meta-llama/Llama-3.1-8B --dataset dataset.jsonl

# Export to GGUF
tuxtrainer export --adapter-path ./finetune_output/final_adapter --model meta-llama/Llama-3.1-8B

# Check system info
tuxtrainer info
```

## Environment Variables

| Variable | Purpose | Required? |
|----------|---------|:---------:|
| `OLLAMA_API_KEY` | Master model API key (Ollama Cloud) | For `ollama_cloud` backend |
| `OLLAMA_NAMESPACE` | Your Ollama registry username | For registry push |
| `OLLAMA_CLOUD_URL` | Override Ollama Cloud API URL | No (default: `https://api.ollama.ai`) |
| `OPENAI_API_KEY` | OpenAI API key | For `openai` backend |
| `HF_TOKEN` | HuggingFace token | For `hf_api` backend |

## Quantisation Options

| Level | Size | Quality | Speed |
|-------|------|---------|-------|
| `Q4_K_M` | Smallest | Good | Fastest |
| `Q5_K_M` | Small | Better | Fast |
| `Q6_K` | Medium | Very good | Medium |
| `Q8_0` | Large | Excellent | Slower |
| `F16` | Largest | Perfect | Slowest |

## What Needs to Be Installed Where

| Step | What you need | Install on Colab? |
|------|--------------|:---:|
| **PDF extraction** | PyMuPDF | `pip install pymupdf` ✅ |
| **HP selection** | Ollama Cloud API key | Just set `OLLAMA_API_KEY` ✅ |
| **Fine-tuning** | PyTorch + Transformers | Pre-installed on Colab GPU ✅ |
| **GGUF conversion** | llama.cpp | `setup_colab()` installs it ✅ |
| **Push to registry** | Ollama (local) + namespace | `setup_colab()` installs Ollama ✅ |

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  PDF Files   │───→│  PDF Processor │───→│   Dataset   │
└─────────────┘    └──────────────┘    └──────┬──────┘
                                               │
                    ┌──────────────┐            │
                    │ Master Model │◄───────────┘
                    │ (Ollama Web  │
                    │  API picks   │
                    │  HPs)        │
                    └──────┬───────┘
                           │
┌─────────────┐    ┌──────┴───────┐    ┌─────────────┐
│  HF Model   │───→│  Fine-tuner   │───→│  Merged Model│
│  (LoRA/QLoRA)│    │  (SFTTrainer) │    │             │
└─────────────┘    └──────────────┘    └──────┬──────┘
                                               │
                    ┌──────────────┐    ┌──────┴──────┐
                    │  Any device  │    │ GGUF Convert │
                    │  ollama pull │◄───│  + Quantise  │
                    │  yourname/m  │    │  + Push to   │
                    └──────────────┘    │  registry    │
                                        └─────────────┘
```

## License

MIT
