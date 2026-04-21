"""
CLI entry point for tuxtrainer.

Provides a comprehensive command-line interface with the following commands:
  * run     — Full pipeline: PDFs → fine-tune → GGUF → Ollama
  * prep    — Only process PDFs into a JSONL dataset (inspect before training)
  * train   — Fine-tune from a prepared dataset
  * export  — Convert a fine-tuned model to GGUF
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
    Quantisation,
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
        help="HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B).",
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


def _export_options(f):
    """Decorator that adds export/push options."""
    f = click.option(
        "--quant",
        type=click.Choice([q.value for q in Quantisation], case_sensitive=False),
        default="Q4_K_M",
        help="GGUF quantisation level.",
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
        default=False,
        help="Use Unsloth for faster fine-tuning.",
    )(f)
    return f


# ---------------------------------------------------------------------------
# Build config from CLI args
# ---------------------------------------------------------------------------

def _build_config(**kwargs) -> FinetuneConfig:
    """Construct a FinetuneConfig from CLI keyword arguments."""
    hp = HyperParams()

    # Apply explicit overrides
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
        quantisation=Quantisation(kwargs.get("quant", "Q4_K_M")),
        skip_ollama=kwargs.get("skip_ollama", False),
        ollama_namespace=kwargs.get("ollama_namespace"),
        ollama_model_name=kwargs.get("ollama_name"),
        ollama_push=kwargs.get("ollama_push", True),
        ollama_host=kwargs.get("ollama_host", "http://localhost:11434"),
        seed=kwargs.get("seed", 42),
        use_unsloth=kwargs.get("use_unsloth", False),
    )

    return config


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="1.0.0", prog_name="tuxtrainer")
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
        model_name = pipeline.run()
        from tuxtrainer.ollama_pusher import _is_colab, chat_with_model
        namespace = config.get_ollama_namespace()
        local_name = config.get_ollama_model_name()
        full_name = config.get_ollama_full_name()

        if _is_colab():
            console.print(f"\n[bold green]Done! Test it with the Web API:[/bold green]")
            console.print(f"[dim]  from tuxtrainer.ollama_pusher import chat_with_model")
            console.print(f"  chat_with_model('{local_name}', 'Hello!')[/dim]")
            if namespace:
                console.print(f"\n[bold]Pull on any device:[/bold] ollama pull {full_name}")
        else:
            if namespace and config.ollama_push and not config.skip_ollama:
                console.print(f"\n[bold green]Done! Pull on any device: ollama pull {full_name}[/bold green]")
                console.print(f"[dim]Run locally: ollama run {local_name}[/dim]")
            else:
                console.print(f"\n[bold green]Done! Run: ollama run {local_name}[/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]Pipeline failed: {e}[/bold red]")
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

    # Expand directories
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

    console.print(f"\n[green]Dataset stats:[/green]")
    console.print(f"  Chunks: {stats.total_chunks}")
    console.print(f"  Estimated tokens: {stats.total_tokens_estimate:,}")
    console.print(f"  Avg tokens/chunk: {stats.avg_chunk_tokens:.0f}")


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

    # Load dataset
    console.print(f"[blue]Loading dataset from {kwargs['dataset']}...[/blue]")
    with open(kwargs["dataset"], "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    dataset = Dataset.from_list(examples)
    console.print(f"[green]Loaded {len(dataset)} examples.[/green]")

    # Hyperparameter selection
    if config.auto_hyperparams:
        from tuxtrainer.hyperparam_selector import HyperparamSelector
        from tuxtrainer.pdf_processor import DatasetStats

        # Create dummy stats from dataset size
        stats = DatasetStats(
            total_chunks=len(dataset),
            total_tokens_estimate=len(dataset) * 200,  # Rough estimate
            avg_chunk_tokens=200,
            min_chunk_tokens=50,
            max_chunk_tokens=500,
            num_sources=1,
            format=config.data_format,
        )
        selector = HyperparamSelector(config)
        config.hyperparams = selector.select(stats)

    # Fine-tune
    model, tokenizer = load_model_and_tokenizer(
        config.model_id,
        max_seq_length=config.hyperparams.max_seq_length,
        method=config.method,
        use_unsloth=config.use_unsloth,
    )
    model = apply_lora_adapters(model, config.hyperparams, config.use_unsloth)
    tokenised = format_dataset_for_training(
        dataset, tokenizer, config.hyperparams, config.data_format,
    )
    adapter_path = do_train(model, tokenizer, tokenised, config)
    console.print(f"[green]Adapter saved to {adapter_path}[/green]")


@main.command()
@click.option("--adapter-path", required=True, type=click.Path(exists=True), help="Path to the LoRA adapter.")
@click.option("--model", "-m", required=True, help="Base HuggingFace model ID.")
@click.option("--output", "-o", default="./finetune_output", help="Output directory.")
@click.option("--quant", type=click.Choice([q.value for q in Quantisation]), default="Q4_K_M")
def export(**kwargs):
    """Merge adapters and convert to GGUF."""
    from tuxtrainer.finetuner import merge_adapter_to_base
    from tuxtrainer.gguf_converter import GGUFConverter

    adapter_path = Path(kwargs["adapter_path"])
    model_id = kwargs["model"]
    output_dir = Path(kwargs["output"])
    quant = Quantisation(kwargs["quant"])

    # Merge
    merged_dir = merge_adapter_to_base(adapter_path, model_id, output_dir)

    # Convert
    converter = GGUFConverter(quantisation=quant)
    gguf_path = converter.convert(merged_dir, output_dir / "gguf")

    console.print(f"[green]GGUF model: {gguf_path}[/green]")


@main.command()
@click.option("--gguf", required=True, type=click.Path(exists=True), help="Path to the GGUF file.")
@click.option("--name", required=True, help="Model name in Ollama.")
@click.option("--namespace", default=None, help="Ollama registry namespace (username) for push.")
@click.option("--system-prompt", default=None, help="Custom system prompt.")
@click.option("--push/--no-push", default=True, help="Push to Ollama registry (default: True).")
@click.option("--ollama-host", default="http://localhost:11434")
def push(**kwargs):
    """Push a GGUF model to Ollama and optionally to the registry."""
    from tuxtrainer.ollama_pusher import OllamaPusher

    # Build a minimal config
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
    table = Table(title="System Info", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Version / Details")

    checks = [
        ("Python", _check_python),
        ("PyTorch", _check_torch),
        ("CUDA / GPU", _check_cuda),
        ("Transformers", _check_transformers),
        ("PEFT", _check_peft),
        ("TRL", _check_trl),
        ("BitsAndBytes", _check_bnb),
        ("Ollama", _check_ollama),
        ("llama.cpp", _check_llama_cpp),
        ("Unsloth", _check_unsloth),
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

def _check_llama_cpp():
    from tuxtrainer.gguf_converter import _find_llama_cpp, _find_convert_script
    path = _find_llama_cpp()
    script = _find_convert_script()
    if path or script:
        return True, f"Found at {path or script}"
    return False, "Not found (optional for GGUF conversion)"

def _check_unsloth():
    try:
        import unsloth
        return True, unsloth.__version__
    except ImportError:
        return False, "Not installed (optional, 2× faster training)"

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
