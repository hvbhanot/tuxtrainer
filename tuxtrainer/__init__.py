"""
tuxtrainer: Fine-tune HuggingFace models with PDFs and push to Ollama.

Uses a master LLM (via the Ollama Web API by default) to automatically
select optimal hyperparameters. By default, the model is pushed to the
Ollama registry so you can use it on any device::

    ollama pull yourname/model-name

Works on Google Colab and local machines — the master model calls
``POST /api/chat`` over HTTPS, so no local Ollama installation is needed
for hyperparameter selection (only for the push step).

The GGUF export is delegated to Unsloth's native
``save_pretrained_gguf``: it merges the LoRA adapter, dequantizes the
4-bit base, runs llama.cpp, and quantizes — all in one call — avoiding
the transformers 5.x ``ConversionOps`` regression that breaks the
standard HuggingFace save path on merged 4-bit models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "1.1.0"

if TYPE_CHECKING:
    from tuxtrainer.colab import setup_colab
    from tuxtrainer.config import FinetuneConfig, HyperParams
    from tuxtrainer.pipeline import FinetunePipeline

__all__ = [
    "FinetuneConfig",
    "HyperParams",
    "FinetunePipeline",
    "setup_colab",
]


def __getattr__(name: str):
    """Lazily expose the public API without importing heavy dependencies."""
    if name in {"FinetuneConfig", "HyperParams"}:
        from tuxtrainer.config import FinetuneConfig, HyperParams

        exports = {
            "FinetuneConfig": FinetuneConfig,
            "HyperParams": HyperParams,
        }
        return exports[name]

    if name == "FinetunePipeline":
        from tuxtrainer.pipeline import FinetunePipeline

        return FinetunePipeline

    if name == "setup_colab":
        from tuxtrainer.colab import setup_colab

        return setup_colab

    raise AttributeError(f"module 'tuxtrainer' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy public attributes to interactive tools."""
    return sorted(set(globals()) | set(__all__))
