#!/usr/bin/env python3
"""
Example: Fine-tune a HuggingFace model with PDFs and push to Ollama — Colab-ready.

The master model uses the **Ollama Web API** by default, so no
local Ollama installation is needed for hyperparameter selection.
Just set your OLLAMA_API_KEY and go.

By default, the model is **pushed to the Ollama registry** so you can
pull it on any device with: ``ollama pull yourname/model-name``

Set your Ollama namespace via the OLLAMA_NAMESPACE env var or
the ``ollama_namespace`` config option.
"""

from pathlib import Path

from tuxtrainer.config import FinetuneConfig, FinetuneMethod, Quantisation
from tuxtrainer.pipeline import FinetunePipeline


def main():
    # ── Step 0 (Colab only): Install Ollama + llama.cpp ────────────────
    # Run this once at the top of your notebook:
    #
    #   from tuxtrainer.colab import setup_colab
    #   setup_colab()  # installs Ollama + llama.cpp (Ollama needed for push)

    # ── Set your API keys ──────────────────────────────────────────────
    # For the Ollama Web API master model (picks hyperparameters):
    #   import os
    #   os.environ["OLLAMA_API_KEY"] = "your-ollama-api-key"
    #
    # For pushing to the Ollama registry (your username):
    #   os.environ["OLLAMA_NAMESPACE"] = "your-ollama-username"
    #
    # Or use OpenAI instead for the master model:
    #   os.environ["OPENAI_API_KEY"] = "your-openai-key"

    # ── Configuration ──────────────────────────────────────────────────
    config = FinetuneConfig(
        # The HuggingFace model to fine-tune
        model_id="unsloth/Llama-3.2-1B-Instruct",

        # Fine-tuning method (qlora = 4-bit quantised, most VRAM efficient)
        method=FinetuneMethod.QLORA,

        # PDF files to use as training data
        pdf_paths=[
            Path("my_document.pdf"),
        ],

        # How to format the data
        data_format="instruction",

        # ── Master model: uses Ollama Web API by default ────────────────
        # No local Ollama needed for HP selection! Just set OLLAMA_API_KEY.
        auto_hyperparams=True,
        master_backend="ollama_cloud",    # Default — calls Ollama Web API
        master_model="llama3.1",          # Model name on Ollama cloud

        # ── Ollama registry push (default: True) ───────────────────────
        # The model is pushed to the Ollama registry so you can pull it
        # on any device. Set your namespace below or via OLLAMA_NAMESPACE.
        ollama_push=True,                 # Default — push to registry
        ollama_namespace="your-username",  # Or set OLLAMA_NAMESPACE env var

        # ── Export settings ──────────────────────────────────────────
        quantisation=Quantisation.Q4_K_M,
        output_dir=Path("./finetune_output"),

        # ── Skip Ollama entirely (just get the GGUF file) ──────────────
        # skip_ollama=True,  # Uncomment if you don't want Ollama at all
    )

    # ── Run the pipeline ───────────────────────────────────────────────
    pipeline = FinetunePipeline(config)
    result = pipeline.run()

    # After this completes, pull the model on any device:
    #   ollama pull your-username/llama-3.1-8b-finetuned

    print(f"\nDone! Output: {result}")


if __name__ == "__main__":
    main()
