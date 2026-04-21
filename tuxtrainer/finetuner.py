"""
Fine-tuning module — LoRA / QLoRA fine-tuning of HuggingFace models.

Supports:
  * QLoRA (4-bit quantised base model + LoRA adapters) — default, most VRAM efficient
  * LoRA (full-precision base + LoRA adapters) — for when you have plenty of VRAM
  * Optional Unsloth integration for ~2× faster training

The module loads the base model, applies LoRA adapters, formats the dataset
into the appropriate chat template, and runs SFTTrainer.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from datasets import Dataset
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from tuxtrainer.config import FinetuneConfig, FinetuneMethod, HyperParams

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _hf_token() -> Optional[str]:
    """Return the HuggingFace token from the environment, if any."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _handle_model_load_error(e: Exception, model_id: str) -> None:
    """Raise a user-friendly error for common model loading failures."""
    from huggingface_hub.utils import GatedRepoError

    error_str = str(e).lower()

    # Gated model
    if isinstance(e, GatedRepoError) or "gated" in error_str or "401" in error_str:
        console.print(Panel(
            f"Model '{model_id}' is gated on HuggingFace.\n\n"
            "1. Accept the license: https://huggingface.co/" + model_id + "\n"
            "2. Create a token: https://huggingface.co/settings/tokens\n"
            "3. Set HF_TOKEN and restart the runtime\n\n"
            "Non-gated alternatives:\n"
            "  • unsloth/Llama-3.2-1B-Instruct\n"
            "  • HuggingFaceTB/SmolLM2-1.7B-Instruct",
            title="Gated Model",
            border_style="red",
        ))
        raise RuntimeError(
            f"Model '{model_id}' is gated. Set HF_TOKEN or use a non-gated model."
        ) from e

    # Unsupported architecture
    if "does not recognize this architecture" in error_str or "model_type" in error_str:
        console.print(Panel(
            f"Model '{model_id}' uses an architecture not supported by your transformers version.\n\n"
            "Fixes:\n"
            "  1. Upgrade transformers:\n"
            "     [dim]pip install --upgrade transformers accelerate[/dim]\n"
            "  2. Or use a model with a known architecture:\n"
            "     [dim]unsloth/Llama-3.2-1B-Instruct[/dim]",
            title="Unsupported Model Architecture",
            border_style="red",
        ))
        raise RuntimeError(
            f"Model '{model_id}' architecture not supported. Upgrade transformers or use a different model."
        ) from e


def load_model_and_tokenizer(
    model_id: str,
    max_seq_length: int = 2048,
    method: FinetuneMethod = FinetuneMethod.QLORA,
    use_unsloth: bool = True,
):
    """Load the base model and tokenizer.

    For QLoRA the model is loaded in 4-bit NF4 quantisation.
    For LoRA it is loaded in full bf16/fp16 precision.
    If use_unsloth is True, the Unsloth-optimised FastLanguageModel is used.
    """
    if use_unsloth:
        return _load_with_unsloth(model_id, max_seq_length, method)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    console.print(f"[cyan]Loading model {model_id}...[/cyan]")

    token = _hf_token()

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
            token=token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load model
        if method == FinetuneMethod.QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                token=token,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                token=token,
            )
    except Exception as e:
        _handle_model_load_error(e, model_id)
        raise

    # Enable gradient checkpointing on the base model (before LoRA)
    model.enable_input_require_grads()

    console.print("[green]  Model loaded[/green]")
    return model, tokenizer


def _load_with_unsloth(
    model_id: str,
    max_seq_length: int = 2048,
    method: FinetuneMethod = FinetuneMethod.QLORA,
):
    """Load model using Unsloth's FastLanguageModel for faster training."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        console.print("[yellow]  Unsloth not found — falling back to standard HF loading[/yellow]")
        return load_model_and_tokenizer(model_id, max_seq_length, method, use_unsloth=False)

    console.print(f"[cyan]Loading model with Unsloth: {model_id}...[/cyan]")

    load_in_4bit = method == FinetuneMethod.QLORA
    token = _hf_token()
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,  # Auto-detect
            token=token,
        )
    except Exception as e:
        _handle_model_load_error(e, model_id)
        raise

    console.print("[green]  Model loaded[/green]")
    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA adapter application
# ---------------------------------------------------------------------------

def _default_target_modules_for_model(model) -> list[str]:
    """Return the recommended LoRA target modules for the given model architecture.

    Different model families use different names for their linear projections.
    This function inspects the model's module tree and returns the names that
    correspond to linear-like modules (or their wrappers) inside the
    transformer blocks.  Wrapped modules like ``Gemma4ClippableLinear`` are
    detected via their inner linear children.

    Falls back to the classic LLaMA-style defaults if the architecture is
    unrecognised.
    """
    import torch.nn as nn

    _SUPPORTED_LINEARS = [nn.Linear]
    try:
        from bitsandbytes.nn import Linear4bit as _L4, Linear8bit as _L8
        _SUPPORTED_LINEARS.extend([_L4, _L8])
    except ImportError:
        pass
    try:
        from transformers.pytorch_utils import Conv1D as _C1D
        _SUPPORTED_LINEARS.append(_C1D)
    except ImportError:
        pass
    _SUPPORTED_LINEARS = tuple(_SUPPORTED_LINEARS)

    _LAYER_PATTERNS = (".layers.", ".blocks.", ".h.", "layers.", "blocks.", "h.")

    # Walk all modules inside transformer layers and collect local names
    # of: (a) supported linear modules, and (b) modules that wrap supported
    # linears (their local name is what users expect, e.g. "q_proj").
    layer_targets: dict[str, int] = {}

    for name, mod in model.named_modules():
        if not any(pat in name for pat in _LAYER_PATTERNS):
            continue

        local = name.split(".")[-1]

        # Direct match: the module itself is a supported linear type
        if isinstance(mod, _SUPPORTED_LINEARS):
            layer_targets[local] = layer_targets.get(local, 0) + 1
            continue

        # Wrapper detection: the module contains supported linear children
        # but is not itself a supported type (e.g. Gemma4ClippableLinear).
        # We want the wrapper's name (e.g. "q_proj"), NOT the inner name.
        has_linear_child = any(
            isinstance(child, _SUPPORTED_LINEARS)
            for _cn, child in mod.named_children()
        )
        if has_linear_child:
            layer_targets[local] = layer_targets.get(local, 0) + 1

    # Remove overly-generic names (like "linear") if more specific names
    # exist, since "linear" usually refers to an inner sub-module of a
    # wrapper rather than a standalone projection.
    _GENERIC_NAMES = {"linear", "weight", "bias"}
    if layer_targets and any(n in _GENERIC_NAMES for n in layer_targets):
        specific_names = {n for n in layer_targets if n not in _GENERIC_NAMES}
        if specific_names:
            for gn in _GENERIC_NAMES:
                layer_targets.pop(gn, None)

    # Remove "container" names — modules like "self_attn" and "mlp" that
    # contain sub-modules which are already in the target set.  We only
    # want leaf-like projection names, not parent containers.
    _CONTAINER_NAMES = {"self_attn", "self_attention", "attention", "mlp",
                        "feed_forward", "ffn", "block", "layer"}
    for cn in _CONTAINER_NAMES:
        layer_targets.pop(cn, None)

    # Further filter: remove any name whose module has children that also
    # appear in the target set (it's a container, not a leaf).
    all_names_in_model = {}
    for name, mod in model.named_modules():
        if any(pat in name for pat in _LAYER_PATTERNS):
            all_names_in_model[name.split(".")[-1]] = mod

    leaf_targets = dict(layer_targets)
    for name in list(leaf_targets):
        mod = all_names_in_model.get(name)
        if mod is not None:
            child_locals = {cn.split(".")[-1] for cn, _ in mod.named_children()}
            if child_locals & set(leaf_targets) and name not in child_locals:
                # This module's children include other targets → it's a container
                del leaf_targets[name]

    if layer_targets:
        candidates = sorted(
            name for name, cnt in layer_targets.items() if cnt >= 2
        )
        if candidates:
            return candidates

    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


# ---------------------------------------------------------------------------
# Well-known defaults per model family (used before the model is loaded)
# ---------------------------------------------------------------------------

_MODEL_FAMILY_TARGETS: list[tuple[str, list[str]]] = [
    ("gemma2", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]),
    ("gemma", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]),
    ("phi3", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]),
    ("phi", [
        "q_proj", "k_proj", "v_proj", "dense",
        "fc1", "fc2",
    ]),
    ("qwen2", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]),
    ("llama", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]),
    ("mistral", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]),
    ("falcon", [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]),
    ("gpt_neox", [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]),
    ("gpt2", [
        "c_attn",
        "c_proj",
        "c_fc",
    ]),
    ("pythia", [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]),
    ("mpt", [
        "Wqkv",
        "out_proj",
        "up_proj",
        "down_proj",
    ]),
    ("opt", [
        "q_proj", "k_proj", "v_proj", "out_proj",
        "fc1", "fc2",
    ]),
]


def guess_target_modules_from_model_id(model_id: str) -> list[str]:
    """Guess reasonable default LoRA target modules from a HuggingFace model ID.

    Uses the model family name (e.g. ``"llama"``, ``"gemma"``, ``"phi"``)
    extracted from the model ID string to look up well-known projection names.
    More-specific family names are matched first (e.g. ``"gemma2"`` before
    ``"gemma"``, ``"gpt_neox"`` before ``"gpt2"``).

    This is a *fallback* — the real resolution happens at apply-time via
    ``resolve_target_modules_for_model`` which inspects the loaded model.
    """
    model_lower = model_id.lower().replace("-", "_")

    for family, targets in _MODEL_FAMILY_TARGETS:
        if family in model_lower:
            return list(targets)

    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


def resolve_target_modules_for_model(model, model_id: str, requested: list[str]) -> list[str]:
    """Auto-detect and resolve the best LoRA target modules for a model.

    This is the main entry-point called by the pipeline.  It:

    1. Inspects the loaded model to discover which linear-like projections
       exist inside the transformer layers.
    2. If the discovered names differ from ``requested``, logs the change.
    3. Resolves any wrapper modules (e.g. ``Gemma4ClippableLinear``) to
       their inner supported linear layers.

    Args:
        model: The loaded HuggingFace model.
        model_id: The HuggingFace model ID (used for logging).
        requested: The user-specified or default target module names.

    Returns:
        A list of target module names suitable for PEFT.
    """
    auto_detected = _default_target_modules_for_model(model)

    # If the auto-detected set differs significantly from what was requested,
    # prefer the auto-detected names (they are guaranteed to exist in the model).
    auto_set = set(auto_detected)
    req_set = set(requested)

    final_requested = requested
    if auto_set != req_set:
        # Check if any of the requested names actually exist as supported
        # modules in the model.  If fewer than half match, swap to auto-detected.
        import torch.nn as nn
        _SUPPORTED_LINEARS = [nn.Linear]
        try:
            from bitsandbytes.nn import Linear4bit as _L4, Linear8bit as _L8
            _SUPPORTED_LINEARS.extend([_L4, _L8])
        except ImportError:
            pass
        try:
            from transformers.pytorch_utils import Conv1D as _C1D
            _SUPPORTED_LINEARS.append(_C1D)
        except ImportError:
            pass
        _SUPPORTED_LINEARS = tuple(_SUPPORTED_LINEARS)

        match_count = 0
        for name, mod in model.named_modules():
            local = name.split(".")[-1]
            if local in req_set and isinstance(mod, (nn.Module,) + _SUPPORTED_LINEARS):
                if isinstance(mod, _SUPPORTED_LINEARS):
                    match_count += 1

        # If fewer than half the requested modules are supported linears,
        # swap to auto-detected names.
        if match_count < len(requested) // 2 + 1:
            console.print(
                f"[yellow]⚠ Default target modules {requested} don't fully match "
                f"the {model_id} architecture.[/yellow]"
            )
            console.print(
                f"[cyan]  Auto-detected target modules for this model: "
                f"{auto_detected}[/cyan]"
            )
            final_requested = auto_detected

    return _resolve_target_modules(model, final_requested)


def _resolve_target_modules(model, requested_modules: list[str]) -> list[str]:
    """Resolve target module names against the actual model layers.

    Some architectures (e.g. Gemma 4) wrap ``nn.Linear`` inside custom
    container modules like ``Gemma4ClippableLinear``.  PEFT can only attach
    LoRA adapters to leaf modules it recognises (Linear, Conv1d, …), so we
    need to point at the *inner* linear layer rather than the wrapper.

    Strategy: walk the model's named modules.  For every module whose
    *local name* (last part of its dotted path) appears in
    ``requested_modules``:

    - If it is a supported type (nn.Linear, etc.), keep the local name as-is.
    - If it is an unsupported wrapper, look for the first supported inner
      sub-module and record its **relative path** from the wrapper
      (e.g. ``"q_proj"`` wrapper with ``"linear"`` sub-module →
      ``"q_proj.linear"``).

    The returned list contains unique resolved names that will match across
    all transformer layers.
    """
    import torch.nn as nn

    _SUPPORTED_BASES = [nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.Conv3d]

    try:
        from transformers.pytorch_utils import Conv1D as _Conv1D
        _SUPPORTED_BASES.append(_Conv1D)
    except ImportError:
        pass

    try:
        from bitsandbytes.nn import Linear4bit as _Linear4bit, Linear8bit as _Linear8bit
        _SUPPORTED_BASES.extend([_Linear4bit, _Linear8bit])
    except ImportError:
        pass

    _SUPPORTED_BASES = tuple(_SUPPORTED_BASES)

    requested_set = set(requested_modules)
    resolved_names: dict[str, str] = {}  # local_name → resolved target name

    for full_name, module in model.named_modules():
        parts = full_name.split(".")
        local_name = parts[-1]

        if local_name not in requested_set:
            continue

        # Already resolved this name with a supported module.
        if local_name in resolved_names:
            continue

        # If the module itself is a supported type, the short name works fine.
        if isinstance(module, _SUPPORTED_BASES):
            resolved_names[local_name] = local_name
            continue

        # Unsupported wrapper — look for supported inner modules.
        found_inner = False
        for inner_name, inner_mod in module.named_modules():
            if inner_name and isinstance(inner_mod, _SUPPORTED_BASES):
                # inner_name is relative to wrapper, e.g. "linear"
                resolved_names[local_name] = f"{local_name}.{inner_name}"
                found_inner = True
                break

        # If no supported inner module found, fall back to the short name.
        # PEFT itself will report a clearer error.
        if not found_inner:
            resolved_names[local_name] = local_name

    if not resolved_names:
        return requested_modules

    result = [resolved_names.get(name, name) for name in requested_modules]

    if result != requested_modules:
        console.print(
            f"[dim]  Resolved LoRA target modules: "
            f"{requested_modules} → {result}[/dim]"
        )

    return result


def apply_lora_adapters(
    model,
    hyperparams: HyperParams,
    use_unsloth: bool = False,
    model_id: str = "",
):
    """Apply LoRA adapters to the base model.

    Returns the model with adapters attached.
    """
    target_modules = resolve_target_modules_for_model(
        model, model_id, hyperparams.lora_target_modules,
    )

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model = FastLanguageModel.get_peft_model(
                model,
                r=hyperparams.lora_r,
                lora_alpha=hyperparams.lora_alpha,
                lora_dropout=hyperparams.lora_dropout,
                target_modules=target_modules,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            console.print("[green]  LoRA adapters applied[/green]")
            return model
        except ImportError:
            pass  # Fall through to standard PEFT

    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=hyperparams.lora_r,
        lora_alpha=hyperparams.lora_alpha,
        lora_dropout=hyperparams.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    console.print("[green]LoRA adapters applied.[/green]")
    return model


# ---------------------------------------------------------------------------
# Dataset formatting for training
# ---------------------------------------------------------------------------

def format_dataset_for_training(
    dataset: Dataset,
    tokenizer,
    hyperparams: HyperParams,
    data_format: str = "instruction",
) -> Dataset:
    """Format the raw dataset into tokenised training examples.

    For "instruction" format, uses the Alpaca prompt template.
    For "completion" format, simply tokenises the text field.
    """
    import torch

    def _format_instruction_sample(example: dict) -> dict:
        """Alpaca-style formatting."""
        if example.get("input"):
            prompt = (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}"
            )
        else:
            prompt = (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Response:\n{example['output']}"
            )
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=hyperparams.max_seq_length,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def _format_completion_sample(example: dict) -> dict:
        """Plain text completion formatting."""
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=hyperparams.max_seq_length,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    console.print(f"[blue]Tokenising dataset ({data_format} format)...[/blue]")

    if data_format == "instruction":
        format_fn = _format_instruction_sample
    elif data_format == "completion":
        format_fn = _format_completion_sample
    else:
        raise ValueError(f"Unknown data format: {data_format!r}")

    tokenised_dataset = dataset.map(
        format_fn,
        remove_columns=dataset.column_names,
        desc="Tokenising",
    )

    console.print(f"[green]  Tokenised {len(tokenised_dataset)} examples[/green]")
    return tokenised_dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model,
    tokenizer,
    tokenised_dataset,
    config: FinetuneConfig,
) -> Path:
    """Run LoRA / QLoRA fine-tuning and return the checkpoint directory.

    Uses HuggingFace TRL's SFTTrainer under the hood.
    """
    from transformers import TrainingArguments
    from trl import SFTTrainer

    hp = config.hyperparams
    output_dir = Path(config.output_dir) / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=hp.num_train_epochs,
        per_device_train_batch_size=hp.per_device_train_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        learning_rate=hp.learning_rate,
        lr_scheduler_type=hp.lr_scheduler_type,
        warmup_ratio=hp.warmup_ratio,
        weight_decay=hp.weight_decay,
        optim=hp.optim,
        fp16=hp.fp16,
        bf16=hp.bf16,
        gradient_checkpointing=hp.gradient_checkpointing,
        logging_steps=hp.logging_steps,
        save_strategy=hp.save_strategy,
        eval_strategy=hp.eval_strategy if "test" in tokenised_dataset.column_names else "no",
        seed=config.seed,
        report_to="none",  # Disable wandb/tensorboard by default
        max_grad_norm=1.0,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenised_dataset,
        max_seq_length=hp.max_seq_length,
        dataset_text_field="text" if config.data_format == "completion" else None,
    )

    # Show training config
    console.print(Panel(
        f"[bold]Model[/bold]: {config.model_id}\n"
        f"[bold]Method[/bold]: {config.method}\n"
        f"[bold]Epochs[/bold]: {hp.num_train_epochs}\n"
        f"[bold]Batch size[/bold]: {hp.per_device_train_batch_size} × {hp.gradient_accumulation_steps} = "
        f"{hp.per_device_train_batch_size * hp.gradient_accumulation_steps} effective\n"
        f"[bold]Learning rate[/bold]: {hp.learning_rate}\n"
        f"[bold]LoRA r/α[/bold]: {hp.lora_r} / {hp.lora_alpha}\n"
        f"[bold]Output[/bold]: {output_dir}",
        title="Training Configuration",
        border_style="green",
    ))

    # Train!
    console.print("[cyan]Starting training...[/cyan]")
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    # Save the final model
    final_dir = Path(config.output_dir) / "final_adapter"
    console.print(f"\n[blue]Saving adapter to {final_dir}...[/blue]")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    console.print(f"[green]  Adapter saved[/green]")
    return final_dir


# ---------------------------------------------------------------------------
# Merge adapters into base model
# ---------------------------------------------------------------------------

def merge_adapter_to_base(
    adapter_path: Path,
    model_id: str,
    output_dir: Path,
    max_seq_length: int = 2048,
) -> Path:
    """Merge the LoRA adapter weights back into the base model.

    This is required before converting to GGUF since llama.cpp needs
    a full model, not an adapter.

    Returns the path to the merged model directory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    merged_dir = output_dir / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Loading base model for merging...[/cyan]")

    token = _hf_token()
    try:
        # Load base model in full precision for merging
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Merge on CPU to avoid OOM
            trust_remote_code=True,
            token=token,
        )

        # Load adapter
        console.print(f"[cyan]Loading adapter from {adapter_path}...[/cyan]")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))

        # Merge
        console.print("[cyan]Merging adapter weights...[/cyan]")
        model = model.merge_and_unload()

        # Save
        console.print(f"[cyan]Saving merged model to {merged_dir}...[/cyan]")
        model.save_pretrained(str(merged_dir), safe_serialization=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
        tokenizer.save_pretrained(str(merged_dir))
    except Exception as e:
        _handle_model_load_error(e, model_id)
        raise

    console.print(f"[green]  Merged model saved[/green]")
    return merged_dir
