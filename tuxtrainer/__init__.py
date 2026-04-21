"""
tuxtrainer: Fine-tune HuggingFace models with PDFs and push to Ollama.

Uses a master LLM (via the Ollama Web API by default) to automatically
select optimal hyperparameters for fine-tuning.  By default, the model
is pushed to the Ollama registry so you can use it on any device:

    ollama pull yourname/model-name

Works on Google Colab and local machines — the master model calls
``POST /api/chat`` over HTTPS, so no local Ollama installation is needed
for hyperparameter selection (only for the push step).
"""

__version__ = "1.0.0"

from tuxtrainer.config import FinetuneConfig, HyperParams
from tuxtrainer.pipeline import FinetunePipeline

__all__ = [
    "FinetuneConfig",
    "HyperParams",
    "FinetunePipeline",
]


def __getattr__(name: str):
    """Lazy import for the colab setup helper."""
    if name == "setup_colab":
        from tuxtrainer.colab import setup_colab
        return setup_colab
    raise AttributeError(f"module 'tuxtrainer' has no attribute {name!r}")
