"""
CLI entry point for tuxtrainer.

Provides a comprehensive command-line interface with the following commands:
  * run     — Full pipeline: PDFs → fine-tune → GGUF → Ollama
  * prep    — Only process PDFs into a JSONL dataset (inspect before training)
  * train   — Fine-tune from a prepared dataset
  * export  — Convert a fine-tuned adapter (+ base) to GGUF via Unsloth
  * push    — Push a GGUF model to Ollama
  * info    — Show system info and dependency check
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tuxtrainer.config import (
    FinetuneConfig,
    FinetuneMethod,
    HyperParams,
    MasterModelBackend,
    Quantization,
    SUPPORTED_QUANTIZATIONS,
)

console = Console()


# ---------------------------------------------------------------------------
# Common options
# ---------------------------------------------------------------------------

def _common_options(f):
    """Decorator that adds common CLI options."""
    f = click.option(
        "--model", "-m",
        envvar="HF_MODEL_ID",
        required=True,
        help="HuggingFace model ID. Unsloth models recommended (e.g. unsloth/Llama-3.2-1B-Instruct).",
    )(f)
    f = click.option(
        "--method",
        type=click.Choice(["lora", "qlora"], case_sensitive=False),
        default="qlora",
        help="Fine-tuning method. QLoRA is more VRAM-efficient.",
    )(f)
    f = click.option(
        "--output", "-o",
        default="./finetune_output",
        help="Working directory for checkpoints and exports.",
    )(f)
    f = click.option(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducibility.",
    )(f)
    return f


def _pdf_options(f):
    """Decorator that adds PDF input options."""
    f = click.option(
        "--pdf", "-p",
        "pdf_paths",
        multiple=True,
        type=click.Path(exists=True, path_type=Path),
        help="Path to a PDF file. Can be specified multiple times.",
    )(f)
    f = click.option(
        "--pdf-dir",
        "pdf_dirs",
        multiple=True,
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        help="Directory containing PDFs. Can be specified multiple times.",
    )(f)
    f = click.option(
        "--data-format",
        type=click.Choice(["instruction", "completion"], case_sensitive=False),
        default="instruction",
        help="How to format PDF content for training.",
    )(f)
    return f


def _hyperparam_options(f):
    """Decorator that adds hyperparameter override options."""
    f = click.option(
        "--auto-hp/--no-auto-hp",
        default=True,
        help="Let the master model choose hyperparameters automatically.",
    )(f)
    f = click.option(
        "--master-backend",
        type=click.Choice(["ollama_cloud", "ollama", "openai", "hf_api", "zai_sdk"], case_sensitive=False),
        default="ollama_cloud",
        help="Backend for the master model. Default: ollama_cloud (Ollama Web API, no local install needed).",
    )(f)
    f = click.option(
        "--master-model",
        default="llama3.1",
        help="Master model for hyperparameter selection.",
    )(f)
    f = click.option(
        "--master-api-key",
        default=None,
        help="API key for the Ollama Web API (or set OLLAMA_API_KEY env var).",
    )(f)
    f = click.option(
        "--lora-r",
        type=int,
        default=None,
        help="Override LoRA rank.",
    )(f)
    f = click.option(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate.",
    )(f)
    f = click.option(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )(f)
    f = click.option(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-device batch size.",
    )(f)
    return f


# Quantization values are shown lowercase (canonical) in help text. The
# Click ``case_sensitive=False`` flag accepts legacy uppercase forms, and
# the Pydantic field validator normalises them before they reach Unsloth.
_QUANT_CLI_CHOICES = sorted(q.value for q in Quantization)


def _export_options(f):
    """Decorator that adds export/push options."""
    f = click.option(
        "--quant",
        type=click.Choice(_QUANT_CLI_CHOICES, case_sensitive=False),
        default="q4_k_m",
        help=f"GGUF quantization level. One of: {sorted(SUPPORTED_QUANTIZATIONS)}.",
    )(f)
    f = click.option(
        "--ollama-name",
        default=None,
        help="Custom name for the model in Ollama (default: derived from model_id).",
    )(f)
    f = click.option(
        "--ollama-namespace",
        default=None,
        help=(
            "Your Ollama registry namespace (username). Required to push to the "
            "registry so the model is available on other devices. "
            "Or set OLLAMA_NAMESPACE env var."
        ),
    )(f)
    f = click.option(
        "--ollama-push/--no-ollama-push",
        default=True,
        help="Push the model to the Ollama registry (default: True, use on any device).",
    )(f)
    f = click.option(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama API host.",
    )(f)
    f = click.option(
        "--skip-ollama/--no-skip-ollama",
        default=False,
        help="Skip Ollama push — just produce the GGUF file.",
    )(f)
    f = click.option(
        "--use-unsloth/--no-unsloth",
        default=True,
        help="Use Unsloth for fine-tuning and GGUF export (default: True).",
    )(f)
    return f


# ---------------------------------------------------------------------------
# Build config from CLI args
# ---------------------------------------------------------------------------

def _build_config(**kwargs) -> FinetuneConfig:
    """Construct a FinetuneConfig from CLI keyword arguments."""
    hp = HyperParams()

    if kwargs.get("lora_r") is not None:
        hp.lora_r = kwargs["lora_r"]
    if kwargs.get("learning_rate") is not None:
        hp.learning_rate = kwargs["learning_rate"]
    if kwargs.get("epochs") is not None:
        hp.num_train_epochs = kwargs["epochs"]
    if kwargs.get("batch_size") is not None:
        hp.per_device_train_batch_size = kwargs["batch_size"]

    config = FinetuneConfig(
        model_id=kwargs["model"],
        method=FinetuneMethod(kwargs.get("method", "qlora")),
        pdf_paths=list(kwargs.get("pdf_paths", [])),
        pdf_dirs=list(kwargs.get("pdf_dirs", [])),
        data_format=kwargs.get("data_format", "instruction"),
        hyperparams=hp,
        master_backend=MasterModelBackend(kwargs.get("master_backend", "ollama_cloud")),
        master_model=kwargs.get("master_model", "llama3.1"),
        master_api_key=kwargs.get("master_api_key"),
        auto_hyperparams=kwargs.get("auto_hp", True),
        output_dir=Path(kwargs.get("output", "./finetune_output")),
        quantization=kwargs.get("quant", "q4_k_m"),
        skip_ollama=kwargs.get("skip_ollama", False),
        ollama_namespace=kwargs.get("ollama_namespace"),
        ollama_model_name=kwargs.get("ollama_name"),
        ollama_push=kwargs.get("ollama_push", True),
        ollama_host=kwargs.get("ollama_host", "http://localhost:11434"),
        seed=kwargs.get("seed", 42),
        use_unsloth=kwargs.get("use_unsloth", True),
    )

    return config


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="1.1.0", prog_name="tuxtrainer")
def main():
    """tuxtrainer: Fine-tune HuggingFace models with PDFs and push to Ollama.

    Uses a master LLM to automatically select optimal hyperparameters.
    """
    pass


@main.command()
@_common_options
@_pdf_options
@_hyperparam_options
@_export_options
def run(**kwargs):
    """Run the full pipeline: PDFs → fine-tune → GGUF → Ollama."""
    config = _build_config(**kwargs)

    from tuxtrainer.pipeline import FinetunePipeline
    pipeline = FinetunePipeline(config)

    try:
        pipeline.run()
    except Exception:
        console.print("[red]Pipeline failed — see error above.[/red]")
        sys.exit(1)


@main.command()
@_pdf_options
@click.option("--output", "-o", default="./dataset.jsonl", help="Output JSONL file path.")
@click.option("--chunk-size", default=512, type=int, help="Chunk size in characters.")
@click.option("--overlap", default=64, type=int, help="Overlap between chunks.")
def prep(**kwargs):
    """Process PDFs into a JSONL dataset for inspection."""
    from tuxtrainer.pdf_processor import PDFProcessor

    pdf_paths = list(kwargs.get("pdf_paths", []))
    pdf_dirs = list(kwargs.get("pdf_dirs", []))

    if not pdf_paths and not pdf_dirs:
        console.print("[red]No PDFs specified. Use --pdf or --pdf-dir.[/red]")
        sys.exit(1)

    all_paths = list(pdf_paths)
    for d in pdf_dirs:
        all_paths.extend(sorted(Path(d).glob("*.pdf")))

    processor = PDFProcessor(
        chunk_size=kwargs.get("chunk_size", 512),
        overlap=kwargs.get("overlap", 64),
        data_format=kwargs.get("data_format", "instruction"),
    )

    output_path = Path(kwargs.get("output", "./dataset.jsonl"))
    stats = processor.process_to_jsonl(
        all_paths,
        output_path,
        kwargs.get("data_format", "instruction"),
    )

    console.print(Panel(
        f"Chunks: {stats.total_chunks}\n"
        f"Tokens: {stats.total_tokens_estimate:,}\n"
        f"Avg/chunk: {stats.avg_chunk_tokens:.0f}",
        title="Dataset",
        border_style="green",
        padding=(0, 1),
    ))


@main.command()
@_common_options
@_hyperparam_options
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True), help="JSONL dataset file.")
def train(**kwargs):
    """Fine-tune a model from a prepared dataset."""
    import json
    from datasets import Dataset
    from tuxtrainer.finetuner import (
        apply_lora_adapters,
        format_dataset_for_training,
        load_model_and_tokenizer,
        train as do_train,
    )

    config = _build_config(**kwargs)

    console.print(f"[blue]Loading dataset from {kwargs['dataset']}...[/blue]")
    with open(kwargs["dataset"], "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    dataset = Dataset.from_list(examples)
    console.print(f"[green]Loaded {len(dataset)} examples.[/green]")

    if config.auto_hyperparams:
        from tuxtrainer.hyperparam_selector import HyperparamSelector
        from tuxtrainer.pdf_processor import DatasetStats

        stats = DatasetStats(
            total_chunks=len(dataset),
            total_tokens_estimate=len(dataset) * 200,
            avg_chunk_tokens=200,
            min_chunk_tokens=50,
            max_chunk_tokens=500,
            num_sources=1,
            format=config.data_format,
        )
        selector = HyperparamSelector(config)
        config.hyperparams = selector.select(stats)

    model, tokenizer = load_model_and_tokenizer(
        config.model_id,
        max_seq_length=config.hyperparams.max_seq_length,
        method=config.method,
        use_unsloth=config.use_unsloth,
    )
    model = apply_lora_adapters(model, config.hyperparams, config.use_unsloth, model_id=config.model_id)
    tokenised = format_dataset_for_training(
        dataset, tokenizer, config.hyperparams, config.data_format,
    )
    adapter_path = do_train(model, tokenizer, tokenised, config)
    console.print(f"[green]Adapter saved to {adapter_path}[/green]")


@main.command()
@click.option("--adapter-path", required=True, type=click.Path(exists=True), help="Path to the LoRA adapter.")
@click.option("--model", "-m", required=True, help="Base HuggingFace model ID.")
@click.option("--output", "-o", default="./finetune_output", help="Output directory.")
@click.option(
    "--quant",
    type=click.Choice(_QUANT_CLI_CHOICES, case_sensitive=False),
    default="q4_k_m",
    help=f"GGUF quantization level. One of: {sorted(SUPPORTED_QUANTIZATIONS)}.",
)
@click.option(
    "--method",
    type=click.Choice(["lora", "qlora"], case_sensitive=False),
    default="qlora",
    help="Whether the base was loaded in 4-bit (qlora) or full precision (lora).",
)
def export(**kwargs):
    """Export a LoRA adapter + base model to a GGUF file via Unsloth.

    Unsloth merges the adapter, dequantizes the base, runs the llama.cpp
    conversion, and quantizes — all in one call.  No intermediate merged
    fp16 checkpoint is produced.
    """
    from tuxtrainer.finetuner import save_gguf_unsloth

    adapter_path = Path(kwargs["adapter_path"])
    output_dir = Path(kwargs["output"])

    config = FinetuneConfig(
        model_id=kwargs["model"],
        method=FinetuneMethod(kwargs.get("method", "qlora")),
        output_dir=output_dir,
        quantization=kwargs["quant"],
    )

    gguf_path = save_gguf_unsloth(
        model=None,
        tokenizer=None,
        config=config,
        adapter_path=adapter_path,
    )

    console.print(f"[green]GGUF model: {gguf_path}[/green]")


@main.command()
@click.option(
    "--gguf",
    required=True,
    type=click.Path(exists=True),
    help="Path to the GGUF file, or a directory containing a .gguf file.",
)
@click.option("--name", required=True, help="Model name in Ollama.")
@click.option("--namespace", default=None, help="Ollama registry namespace (username) for push.")
@click.option("--system-prompt", default=None, help="Custom system prompt.")
@click.option("--push/--no-push", default=True, help="Push to Ollama registry (default: True).")
@click.option("--ollama-host", default="http://localhost:11434")
def push(**kwargs):
    """Push a GGUF model to Ollama and optionally to the registry."""
    from tuxtrainer.ollama_pusher import OllamaPusher

    config = FinetuneConfig(
        model_id="unused",
        ollama_namespace=kwargs.get("namespace"),
        ollama_model_name=kwargs["name"],
        ollama_push=kwargs["push"],
        ollama_host=kwargs["ollama_host"],
    )

    pusher = OllamaPusher(config)
    model_name = pusher.push(
        Path(kwargs["gguf"]),
        system_prompt=kwargs.get("system_prompt"),
    )
    if "/" in model_name:
        console.print(f"\n[green]Pull on any device: ollama pull {model_name}[/green]")


@main.command()
def info():
    """Show system info and check dependencies."""
    table = Table(show_header=True, header_style="bold", border_style="cyan")
    table.add_column("Component")
    table.add_column("Status", justify="center")
    table.add_column("Version")

    checks = [
        ("Python", _check_python),
        ("PyTorch", _check_torch),
        ("CUDA / GPU", _check_cuda),
        ("Transformers", _check_transformers),
        ("PEFT", _check_peft),
        ("TRL", _check_trl),
        ("BitsAndBytes", _check_bnb),
        ("Ollama", _check_ollama),
        ("Unsloth", _check_unsloth),
        ("Unsloth Zoo", _check_unsloth_zoo),
        ("PyMuPDF", _check_pymupdf),
    ]

    for name, check_fn in checks:
        status, detail = check_fn()
        icon = "[green]✓[/green]" if status else "[red]✗[/red]"
        table.add_row(name, icon, detail)

    console.print(table)


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _check_python():
    import sys
    return True, f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def _check_torch():
    try:
        import torch
        return True, torch.__version__
    except ImportError:
        return False, "Not installed"

def _check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            return True, f"{name} ({vram:.1f} GB)"
        return False, "No CUDA GPU detected"
    except ImportError:
        return False, "PyTorch not installed"

def _check_transformers():
    try:
        import transformers
        return True, transformers.__version__
    except ImportError:
        return False, "Not installed"

def _check_peft():
    try:
        import peft
        return True, peft.__version__
    except ImportError:
        return False, "Not installed"

def _check_trl():
    try:
        import trl
        return True, trl.__version__
    except ImportError:
        return False, "Not installed"

def _check_bnb():
    try:
        import bitsandbytes
        return True, bitsandbytes.__version__
    except ImportError:
        return False, "Not installed"

def _check_ollama():
    """Check if the Ollama Web API is reachable (works on Colab)."""
    from tuxtrainer.ollama_pusher import check_ollama_running, list_ollama_models
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        if check_ollama_running(host):
            models = list_ollama_models(host)
            return True, f"API reachable at {host} ({len(models)} model(s))"
        return False, f"Not reachable at {host}"
    except Exception as e:
        return False, f"Error: {e}"

def _check_unsloth():
    try:
        import unsloth
        return True, getattr(unsloth, "__version__", "installed")
    except ImportError:
        return False, "Not installed (required for GGUF export)"

def _check_unsloth_zoo():
    try:
        import unsloth_zoo
        return True, getattr(unsloth_zoo, "__version__", "installed")
    except ImportError:
        return False, "Not installed (required for GGUF export)"

def _check_pymupdf():
    try:
        import fitz
        return True, fitz.version[0]
    except ImportError:
        return False, "Not installed"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
