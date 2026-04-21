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


def _handle_gated_repo_error(e: Exception, model_id: str) -> None:
    """Raise a user-friendly error when a gated model cannot be accessed."""
    from huggingface_hub.utils import GatedRepoError

    error_str = str(e).lower()
    if isinstance(e, GatedRepoError) or "gated" in error_str or "401" in error_str:
        raise RuntimeError(
            f"\n[bold red]Model '{model_id}' is gated on HuggingFace.[/bold red]\n"
            "You need to:\n"
            "  1. Accept the model license at "
            f"[blue]https://huggingface.co/{model_id}[/blue]\n"
            "  2. Create a HuggingFace token at [blue]https://huggingface.co/settings/tokens[/blue]\n"
            "  3. Set it as a secret in Colab or export it:\n"
            "     [dim]import os; os.environ['HF_TOKEN'] = 'your_token_here'[/dim]\n"
            "  4. Restart the runtime and try again.\n"
            "\nAlternatively, use a non-gated model such as:\n"
            "  [dim]unsloth/Llama-3.2-1B-Instruct[/dim]\n"
            "  [dim]HuggingFaceTB/SmolLM2-1.7B-Instruct[/dim]"
        ) from e


def load_model_and_tokenizer(
    model_id: str,
    max_seq_length: int = 2048,
    method: FinetuneMethod = FinetuneMethod.QLORA,
    use_unsloth: bool = False,
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

    console.print(f"[bold blue]Loading model [cyan]{model_id}[/cyan]...[/bold blue]")

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
                max_seq_length=max_seq_length,
                token=token,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                max_seq_length=max_seq_length,
                token=token,
            )
    except Exception as e:
        _handle_gated_repo_error(e, model_id)
        raise

    # Enable gradient checkpointing on the base model (before LoRA)
    model.enable_input_require_grads()

    console.print("[green]Model loaded successfully.[/green]")
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
        console.print(
            "[red]Unsloth is not installed. Install with: pip install unsloth[/red]\n"
            "[yellow]Falling back to standard HuggingFace loading.[/yellow]"
        )
        return load_model_and_tokenizer(model_id, max_seq_length, method, use_unsloth=False)

    console.print(f"[bold blue]Loading model with [magenta]Unsloth[/magenta]: [cyan]{model_id}[/cyan]...[/bold blue]")

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
        _handle_gated_repo_error(e, model_id)
        raise

    console.print("[green]Model loaded with Unsloth.[/green]")
    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA adapter application
# ---------------------------------------------------------------------------

def apply_lora_adapters(
    model,
    hyperparams: HyperParams,
    use_unsloth: bool = False,
):
    """Apply LoRA adapters to the base model.

    Returns the model with adapters attached.
    """
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model = FastLanguageModel.get_peft_model(
                model,
                r=hyperparams.lora_r,
                lora_alpha=hyperparams.lora_alpha,
                lora_dropout=hyperparams.lora_dropout,
                target_modules=hyperparams.lora_target_modules,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            console.print("[green]LoRA adapters applied (Unsloth-optimised).[/green]")
            return model
        except ImportError:
            pass  # Fall through to standard PEFT

    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=hyperparams.lora_r,
        lora_alpha=hyperparams.lora_alpha,
        lora_dropout=hyperparams.lora_dropout,
        target_modules=hyperparams.lora_target_modules,
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

    console.print(f"[green]Tokenised {len(tokenised_dataset)} examples.[/green]")
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
    console.print("\n[bold green]Starting training...[/bold green]\n")
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    # Save the final model
    final_dir = Path(config.output_dir) / "final_adapter"
    console.print(f"\n[blue]Saving adapter to {final_dir}...[/blue]")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    console.print(f"[green]Adapter saved to {final_dir}[/green]")
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

    console.print(f"[blue]Loading base model for merging...[/blue]")

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
        console.print(f"[blue]Loading adapter from {adapter_path}...[/blue]")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))

        # Merge
        console.print("[blue]Merging adapter weights into base model...[/blue]")
        model = model.merge_and_unload()

        # Save
        console.print(f"[blue]Saving merged model to {merged_dir}...[/blue]")
        model.save_pretrained(str(merged_dir), safe_serialization=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
        tokenizer.save_pretrained(str(merged_dir))
    except Exception as e:
        _handle_gated_repo_error(e, model_id)
        raise

    console.print(f"[green]Merged model saved to {merged_dir}[/green]")
    return merged_dir
