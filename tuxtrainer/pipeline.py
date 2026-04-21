"""
Orchestration pipeline — end-to-end fine-tuning from PDFs to Ollama.

This is the main entry point that coordinates:
  1. PDF extraction and dataset preparation
  2. Master model hyperparameter selection
  3. Model fine-tuning (LoRA / QLoRA)
  4. Adapter merging
  5. GGUF conversion
  6. Ollama model creation and push

Usage::

    from tuxtrainer import FinetuneConfig, FinetunePipeline

    config = FinetuneConfig(
        model_id="meta-llama/Llama-3.1-8B",
        pdf_paths=["doc1.pdf", "doc2.pdf"],
        auto_hyperparams=True,
    )

    pipeline = FinetunePipeline(config)
    model_name = pipeline.run()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from tuxtrainer.config import FinetuneConfig, FinetuneMethod
from tuxtrainer.finetuner import (
    apply_lora_adapters,
    format_dataset_for_training,
    load_model_and_tokenizer,
    merge_adapter_to_base,
    train,
)
from tuxtrainer.gguf_converter import GGUFConverter
from tuxtrainer.hyperparam_selector import HyperparamSelector
from tuxtrainer.ollama_pusher import OllamaPusher
from tuxtrainer.pdf_processor import PDFProcessor

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FinetunePipeline:
    """End-to-end pipeline: PDFs → fine-tuned model → Ollama.

    Usage::

        config = FinetuneConfig(model_id="meta-llama/Llama-3.1-8B", ...)
        pipeline = FinetunePipeline(config)
        model_name = pipeline.run()
    """

    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        self._start_time: float = 0
        self._stage_times: list[tuple[str, float]] = []

    def run(self) -> str:
        """Execute the full fine-tuning pipeline.

        Returns:
            The Ollama model name, or the GGUF file path if Ollama is skipped.
        """
        self._start_time = time.time()

        console.print(Panel(
            self._summary_text(),
            title="tuxtrainer Pipeline",
            border_style="bright_blue",
        ))

        # ── Stage 1: Process PDFs ──────────────────────────────────────
        dataset, stats = self._stage("PDF Processing", self._process_pdfs)

        # ── Stage 2: Select hyperparameters ────────────────────────────
        hyperparams = self._stage("Hyperparameter Selection", self._select_hyperparams, stats)

        # ── Stage 3: Fine-tune ─────────────────────────────────────────
        adapter_path = self._stage("Fine-tuning", self._finetune, dataset, hyperparams)

        # ── Stage 4: Merge adapters ────────────────────────────────────
        merged_dir = self._stage("Merging", self._merge, adapter_path)

        # ── Stage 5: Convert to GGUF ───────────────────────────────────
        gguf_path = self._stage("GGUF Conversion", self._convert_to_gguf, merged_dir)

        # ── Stage 6: Push to Ollama (optional) ─────────────────────────
        if self.config.skip_ollama:
            console.print("\n[yellow]Skipping Ollama push (skip_ollama=True).[/yellow]")
            console.print(f"[green]GGUF file ready at: {gguf_path}[/green]")
            model_name = str(gguf_path)
        else:
            try:
                model_name = self._stage("Ollama Push", self._push_to_ollama, gguf_path)
            except RuntimeError as e:
                console.print(f"\n[yellow]Ollama push skipped (server not available): {e}[/yellow]")
                console.print(f"[green]GGUF file ready at: {gguf_path}[/green]")
                console.print("[dim]You can load it manually with: ollama create my-model -f Modelfile[/dim]")
                model_name = str(gguf_path)

        # ── Done ───────────────────────────────────────────────────────
        total_time = time.time() - self._start_time
        self._print_summary(model_name, total_time, gguf_path)

        return model_name

    # ── Stage implementations ──────────────────────────────────────────

    def _process_pdfs(self):
        """Stage 1: Extract text from PDFs and prepare a training dataset."""
        pdf_paths = self.config.get_all_pdf_paths()

        if not pdf_paths:
            raise ValueError(
                "No PDF files provided. Use --pdf or --pdf-dir to specify input files."
            )

        console.print(f"\n[blue]Found {len(pdf_paths)} PDF file(s):[/blue]")
        for p in pdf_paths:
            console.print(f"  • {p}")

        processor = PDFProcessor(
            chunk_size=self.config.hyperparams.max_seq_length,
            data_format=self.config.data_format,
        )

        dataset, stats = processor.process(pdf_paths)

        # Save dataset stats for the master model
        stats_path = Path(self.config.output_dir) / "dataset_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(stats.__dict__, indent=2, default=str),
            encoding="utf-8",
        )

        return dataset, stats

    def _select_hyperparams(self, stats):
        """Stage 2: Ask the master model to pick hyperparameters."""
        selector = HyperparamSelector(self.config)
        hyperparams = selector.select(stats)

        # Update config with selected hyperparams
        self.config.hyperparams = hyperparams

        # Save selected hyperparams
        hp_path = Path(self.config.output_dir) / "selected_hyperparams.json"
        hp_path.write_text(
            hyperparams.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return hyperparams

    def _finetune(self, dataset, hyperparams):
        """Stage 3: Load model, apply LoRA, and train."""
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            model_id=self.config.model_id,
            max_seq_length=hyperparams.max_seq_length,
            method=self.config.method,
            use_unsloth=self.config.use_unsloth,
        )

        # Apply LoRA adapters
        model = apply_lora_adapters(
            model,
            hyperparams,
            use_unsloth=self.config.use_unsloth,
        )

        # Format dataset
        tokenised_dataset = format_dataset_for_training(
            dataset, tokenizer, hyperparams, self.config.data_format,
        )

        # Split into train/eval
        split = self.config.train_test_split
        if split < 1.0:
            split_dataset = tokenised_dataset.train_test_split(
                train_size=split, seed=self.config.seed,
            )
            tokenised_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            eval_dataset = None

        # Train
        adapter_path = train(model, tokenizer, tokenised_dataset, self.config)

        return adapter_path

    def _merge(self, adapter_path):
        """Stage 4: Merge LoRA adapters back into the base model."""
        merged_dir = merge_adapter_to_base(
            adapter_path=adapter_path,
            model_id=self.config.model_id,
            output_dir=Path(self.config.output_dir),
            max_seq_length=self.config.hyperparams.max_seq_length,
        )
        return merged_dir

    def _convert_to_gguf(self, merged_dir):
        """Stage 5: Convert the merged model to GGUF format."""
        converter = GGUFConverter(
            quantisation=self.config.quantisation,
        )
        model_name = self.config.get_ollama_model_name()
        gguf_path = converter.convert(
            model_dir=merged_dir,
            output_dir=Path(self.config.output_dir) / "gguf",
            model_name=model_name,
        )
        return gguf_path

    def _push_to_ollama(self, gguf_path):
        """Stage 6: Create the model in Ollama and optionally push to registry."""
        pusher = OllamaPusher(self.config)
        model_name = pusher.push(gguf_path)
        return model_name

    # ── Helpers ────────────────────────────────────────────────────────

    def _stage(self, name: str, fn, *args, **kwargs):
        """Run a pipeline stage with timing and error handling."""
        console.print(f"\n{'═' * 60}")
        console.print(f"[bold bright_white]Stage: {name}[/bold bright_white]")
        console.print(f"{'═' * 60}")

        start = time.time()
        try:
            result = fn(*args, **kwargs)
            elapsed = time.time() - start
            self._stage_times.append((name, elapsed))
            console.print(f"[green]✓ {name} completed in {elapsed:.1f}s[/green]")
            return result
        except Exception as e:
            elapsed = time.time() - start
            self._stage_times.append((name, elapsed))
            console.print(f"[red]✗ {name} failed after {elapsed:.1f}s: {e}[/red]")
            raise

    def _summary_text(self) -> str:
        """Build a summary string of the pipeline configuration."""
        pdf_paths = self.config.get_all_pdf_paths() if self.config.pdf_paths or self.config.pdf_dirs else []
        ollama_status = "skipped" if self.config.skip_ollama else (
            "push to registry" if self.config.ollama_push and self.config.get_ollama_namespace()
            else "local only"
        )
        namespace = self.config.get_ollama_namespace() or "(not set)"
        return (
            f"[bold]Model[/bold]: {self.config.model_id}\n"
            f"[bold]Method[/bold]: {self.config.method}\n"
            f"[bold]PDFs[/bold]: {len(pdf_paths)} file(s)\n"
            f"[bold]Auto HP[/bold]: {self.config.auto_hyperparams}\n"
            f"[bold]Master[/bold]: {self.config.master_model} ({self.config.master_backend})\n"
            f"[bold]Quant[/bold]: {self.config.quantisation}\n"
            f"[bold]Ollama[/bold]: {ollama_status}\n"
            f"[bold]Namespace[/bold]: {namespace}\n"
            f"[bold]Output[/bold]: {self.config.output_dir}"
        )

    def _print_summary(self, model_name: str, total_time: float, gguf_path: Path) -> None:
        """Print a final summary of the pipeline run."""
        table = Table(title="Pipeline Summary", show_header=True, header_style="bold cyan")
        table.add_column("Stage", style="dim")
        table.add_column("Time", justify="right")

        for name, elapsed in self._stage_times:
            table.add_row(name, f"{elapsed:.1f}s")

        table.add_row("[bold]Total[/bold]", f"[bold]{total_time:.1f}s[/bold]")

        console.print()
        console.print(table)
        console.print(f"\n[bold green]GGUF file: {gguf_path}[/bold green]")

        if not self.config.skip_ollama and "ollama" not in str(gguf_path).lower():
            namespace = self.config.get_ollama_namespace()
            local_name = self.config.get_ollama_model_name()
            full_name = self.config.get_ollama_full_name()

            if namespace and self.config.ollama_push:
                console.print(
                    f"\n[bold green]Model pushed to Ollama registry![/bold green]\n"
                    f"[bold]Pull on any device:[/bold]  ollama pull {full_name}\n"
                    f"[bold]Run on any device:[/bold]  ollama run {full_name}\n"
                    f"[dim]The model is also available locally as '{local_name}'.[/dim]"
                )
            else:
                console.print(
                    f"\n[bold green]Model '{local_name}' is ready in local Ollama![/bold green]"
                )
                console.print(
                    "[dim]To make it available on other devices, set --ollama-namespace and re-run,[/dim]\n"
                    "[dim]or manually push: ollama push yourname/model-name[/dim]"
                )
        else:
            console.print("[dim]Load the GGUF with llama.cpp, Ollama, or any GGUF-compatible runtime.[/dim]")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def finetune_and_push(config: FinetuneConfig) -> str:
    """One-function API for the full pipeline.

    Usage::

        from tuxtrainer.pipeline import finetune_and_push
        from tuxtrainer.config import FinetuneConfig

        config = FinetuneConfig(
            model_id="meta-llama/Llama-3.1-8B",
            pdf_paths=["doc1.pdf"],
        )
        model_name = finetune_and_push(config)
    """
    pipeline = FinetunePipeline(config)
    return pipeline.run()
