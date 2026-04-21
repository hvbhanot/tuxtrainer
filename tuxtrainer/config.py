"""
Configuration models for the tuxtrainer pipeline.

All tuneable knobs live here as Pydantic models so they are validated,
serialisable, and easy to override from the CLI or a YAML file.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Quantisation(str, Enum):
    """Quantisation levels for the final GGUF export."""
    Q4_K_M = "Q4_K_M"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    F16 = "F16"


class MasterModelBackend(str, Enum):
    """Where the master model (hyperparameter selector) runs."""
    OLLAMA_CLOUD = "ollama_cloud"  # Ollama Web API (default, no local install needed)
    OLLAMA = "ollama"              # Local Ollama server
    OPENAI = "openai"              # OpenAI-compatible API (GPT-4, etc.)
    HF_API = "hf_api"              # HuggingFace Inference API
    ZAI_SDK = "zai_sdk"            # z-ai-web-dev-sdk


class FinetuneMethod(str, Enum):
    """Fine-tuning strategy."""
    LORA = "lora"
    QLORA = "qlora"


# ---------------------------------------------------------------------------
# Hyperparameter model (what the master model decides)
# ---------------------------------------------------------------------------

class HyperParams(BaseModel):
    """Hyperparameters for LoRA / QLoRA fine-tuning.

    The master model fills these in; the user can also override any field
    explicitly from the CLI or config file.
    """
    # --- LoRA rank & alpha ---
    lora_r: int = Field(
        default=16,
        ge=1,
        le=256,
        description="LoRA rank (higher = more capacity, more VRAM).",
    )
    lora_alpha: int = Field(
        default=32,
        ge=1,
        description="LoRA alpha (scaling factor). Rule of thumb: 2× lora_r.",
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Dropout probability on LoRA layers.",
    )

    # --- Target modules (default for LLaMA-style models) ---
    lora_target_modules: list[str] = Field(
        default=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        description="Which linear layers to apply LoRA adapters to.",
    )

    # --- Training ---
    num_train_epochs: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Number of training epochs.",
    )
    per_device_train_batch_size: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Batch size per GPU.",
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Gradient accumulation steps (effective batch = batch × accum).",
    )
    learning_rate: float = Field(
        default=2e-4,
        gt=0.0,
        le=1e-2,
        description="Peak learning rate.",
    )
    lr_scheduler_type: str = Field(
        default="cosine",
        description="LR scheduler: cosine, linear, constant, etc.",
    )
    warmup_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Fraction of steps used for LR warm-up.",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description="AdamW weight decay.",
    )
    max_seq_length: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description="Maximum sequence length for training.",
    )

    # --- Optimisation ---
    optim: str = Field(
        default="adamw_8bit",
        description="Optimiser. 'adamw_8bit' saves VRAM; 'adamw_torch' for full precision.",
    )
    fp16: bool = Field(default=False, description="Use FP16 mixed precision.")
    bf16: bool = Field(default=True, description="Use BF16 mixed precision (Ampere+).")
    gradient_checkpointing: bool = Field(
        default=True,
        description="Trade compute for memory via gradient checkpointing.",
    )

    # --- Logging / Saving ---
    logging_steps: int = Field(default=10, ge=1)
    save_strategy: str = Field(default="epoch")
    eval_strategy: str = Field(default="epoch")

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------

class FinetuneConfig(BaseModel):
    """Full pipeline configuration.

    Every parameter that the CLI, YAML, or master model can set lives here.
    """

    # --- Model ---
    model_id: str = Field(
        ...,
        description="HuggingFace model ID, e.g. 'meta-llama/Llama-3.1-8B'.",
    )
    method: FinetuneMethod = Field(
        default=FinetuneMethod.QLORA,
        description="Fine-tuning method: lora or qlora.",
    )

    # --- Data ---
    pdf_paths: list[Path] = Field(
        default_factory=list,
        description="Paths to PDF files to use as training data.",
    )
    pdf_dirs: list[Path] = Field(
        default_factory=list,
        description="Directories containing PDF files.",
    )
    data_format: str = Field(
        default="instruction",
        description="How to format PDF content: 'instruction' (Q&A pairs) or 'completion'.",
    )
    train_test_split: float = Field(
        default=0.9,
        ge=0.5,
        le=0.99,
        description="Fraction of data used for training (rest = eval).",
    )

    # --- Hyperparameters (can be auto-filled by master model) ---
    hyperparams: HyperParams = Field(default_factory=HyperParams)

    # --- Master model ---
    master_backend: MasterModelBackend = Field(
        default=MasterModelBackend.OLLAMA_CLOUD,
        description=(
            "Backend for the master model that selects hyperparameters. "
            "'ollama_cloud' (default) calls the Ollama Web API via HTTPS — "
            "no local install needed. Set OLLAMA_API_KEY env var for authentication."
        ),
    )
    master_model: str = Field(
        default="llama3.1",
        description=(
            "Name of the master model. For ollama_cloud this is the Ollama Web API "
            "model name (e.g. 'llama3.1', 'llama3.1:70b'). For 'ollama' this is a "
            "local tag."
        ),
    )
    master_api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key for the master model backend. For ollama_cloud, set OLLAMA_API_KEY "
            "env var or pass it here. For openai, set OPENAI_API_KEY."
        ),
    )
    master_base_url: Optional[str] = Field(
        default=None,
        description=(
            "Custom base URL for the master model API. Defaults:\n"
            "  ollama_cloud → https://ollama.com\n"
            "  openai → https://api.openai.com/v1\n"
            "  zai_sdk → https://api.z.ai/v1"
        ),
    )
    auto_hyperparams: bool = Field(
        default=True,
        description="Whether to ask the master model to pick hyperparameters.",
    )

    # --- Export ---
    output_dir: Path = Field(
        default=Path("./finetune_output"),
        description="Working directory for checkpoints and exports.",
    )
    quantisation: Quantisation = Field(
        default=Quantisation.Q4_K_M,
        description="Quantisation level for the GGUF export.",
    )

    # --- Ollama (push to registry by default so you can use it on any device) ---
    skip_ollama: bool = Field(
        default=False,
        description=(
            "Skip the Ollama push step entirely. "
            "By default the model is pushed to the Ollama registry so you can "
            "pull it on any device with: ollama pull <namespace>/<model-name>"
        ),
    )
    ollama_namespace: Optional[str] = Field(
        default=None,
        description=(
            "Your Ollama registry namespace (username). Required to push to the "
            "registry so the model is available on other devices. "
"Set OLLAMA_NAMESPACE env var or pass it here. "
"Example: 'myuser' → model becomes 'myuser/llama-3.1-8b-finetuned'"
        ),
    )
    ollama_model_name: Optional[str] = Field(
        default=None,
        description="Name for the model in Ollama (default: derived from model_id).",
    )
    ollama_push: bool = Field(
        default=True,
        description=(
            "Push the model to the Ollama registry after creation (default: True). "
            "This makes the model available on any device via 'ollama pull'."
        ),
    )
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama API host.",
    )

    # --- Misc ---
    seed: int = Field(default=42)
    resume_from_checkpoint: Optional[Path] = Field(default=None)
    use_unsloth: bool = Field(
        default=False,
        description="Use Unsloth for 2× faster fine-tuning (requires unsloth package).",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("pdf_paths", mode="before")
    @classmethod
    def _resolve_pdf_paths(cls, v: list) -> list[Path]:
        return [Path(p).expanduser().resolve() for p in v]

    @field_validator("pdf_dirs", mode="before")
    @classmethod
    def _resolve_pdf_dirs(cls, v: list) -> list[Path]:
        return [Path(p).expanduser().resolve() for p in v]

    def get_ollama_namespace(self) -> str:
        """Return the Ollama registry namespace (username)."""
        ns = self.ollama_namespace or os.environ.get("OLLAMA_NAMESPACE", "")
        return ns

    def get_ollama_model_name(self) -> str:
        """Return the model name (without namespace prefix)."""
        if self.ollama_model_name:
            return self.ollama_model_name
        # Derive from HuggingFace ID: "meta-llama/Llama-3.1-8B" → "llama-3.1-8b-finetuned"
        name = self.model_id.split("/")[-1].lower()
        return f"{name}-finetuned"

    def get_ollama_full_name(self) -> str:
        """Return the full Ollama model name including namespace for registry push.

        Examples:
            namespace='myuser', model='llama-3.1-8b-finetuned'
            → 'myuser/llama-3.1-8b-finetuned'

            namespace='' (not set), model='llama-3.1-8b-finetuned'
            → 'llama-3.1-8b-finetuned' (local only, cannot push to registry)
        """
        ns = self.get_ollama_namespace()
        name = self.get_ollama_model_name()
        if ns:
            return f"{ns}/{name}"
        return name

    def get_all_pdf_paths(self) -> list[Path]:
        """Merge explicit PDF paths with PDFs found in pdf_dirs."""
        all_paths = list(self.pdf_paths)
        for d in self.pdf_dirs:
            if d.is_dir():
                all_paths.extend(sorted(d.glob("*.pdf")))
            else:
                raise FileNotFoundError(f"PDF directory does not exist: {d}")
        # Deduplicate
        return sorted(set(all_paths))

    class Config:
        use_enum_values = True
