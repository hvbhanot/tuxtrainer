"""
Orchestration pipeline — end-to-end fine-tuning from PDFs to Ollama.

This is the main entry point that coordinates:
  1. PDF extraction and dataset preparation
  2. Master model hyperparameter selection
  3. Model fine-tuning (LoRA / QLoRA) on a 4-bit base
  4. GGUF export via Unsloth's native ``save_pretrained_gguf``
     (merge + dequantize + llama.cpp convert + quantize all in one call)
  5. Ollama model creation and push

Usage::

    from tuxtrainer import FinetuneConfig, FinetunePipeline

    config = FinetuneConfig(
        model_id="unsloth/Llama-3.2-1B-Instruct",
        pdf_paths=["doc1.pdf", "doc2.pdf"],
        auto_hyperparams=True,
    )

    pipeline = FinetunePipeline(config)
    model_name = pipeline.run()
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tuxtrainer.config import FinetuneConfig
from tuxtrainer.finetuner import (
    apply_lora_adapters,
    format_dataset_for_training,
    load_model_and_tokenizer,
    save_gguf_unsloth,
    train,
)
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

        config = FinetuneConfig(model_id="unsloth/Llama-3.2-1B-Instruct", ...)
        pipeline = FinetunePipeline(config)
        model_name = pipeline.run()
    """

    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        self._start_time: float = 0
        self._stage_times: list[tuple[str, float]] = []
        # Kept alive between the training and GGUF-export stages so Unsloth
        # can convert straight from the in-memory LoRA-on-4bit model.
        self._model = None
        self._tokenizer = None

    def run(self) -> str:
        """Execute the full fine-tuning pipeline.

        Returns:
            The Ollama model name, or the GGUF file path if Ollama is skipped.
        """
        self._start_time = time.time()

        console.print(Panel(
            self._summary_text(),
            title="tuxtrainer",
            border_style="cyan",
            padding=(0, 1),
        ))

        # ── Stage 1: Process PDFs ──────────────────────────────────────
        dataset, stats = self._stage("PDF Processing", self._process_pdfs)

        # ── Stage 2: Select hyperparameters ────────────────────────────
        hyperparams = self._stage("Hyperparameter Selection", self._select_hyperparams, stats)

        # ── Stage 3: Fine-tune ─────────────────────────────────────────
        adapter_path = self._stage("Fine-tuning", self._finetune, dataset, hyperparams)

        # ── Stage 4: GGUF export (Unsloth) ─────────────────────────────
        gguf_path = self._stage("GGUF Export", self._export_gguf, adapter_path)

        # ── Stage 5: Push to Ollama (optional) ─────────────────────────
        if self.config.skip_ollama:
            model_name = str(gguf_path)
        else:
            try:
                model_name = self._stage("Ollama Push", self._push_to_ollama, gguf_path)
            except RuntimeError:
                console.print("[yellow]Ollama push skipped — server not available.[/yellow]")
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

        console.print(f"[cyan]Found {len(pdf_paths)} PDF file(s):[/cyan]")
        for p in pdf_paths:
            console.print(f"  [dim]• {p}[/dim]")

        processor = PDFProcessor(
            chunk_size=self.config.hyperparams.max_seq_length,
            data_format=self.config.data_format,
        )

        dataset, stats = processor.process(pdf_paths)

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

        self.config.hyperparams = hyperparams

        hp_path = Path(self.config.output_dir) / "selected_hyperparams.json"
        hp_path.write_text(
            hyperparams.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return hyperparams

    def _finetune(self, dataset, hyperparams):
        """Stage 3: Load model, apply LoRA, and train.

        Keeps references to the Unsloth-wrapped model and tokenizer on the
        pipeline instance so Stage 4 can feed them straight into
        ``save_pretrained_gguf`` without reloading the base.
        """
        model, tokenizer = load_model_and_tokenizer(
            model_id=self.config.model_id,
            max_seq_length=hyperparams.max_seq_length,
            method=self.config.method,
            use_unsloth=self.config.use_unsloth,
        )

        model = apply_lora_adapters(
            model,
            hyperparams,
            use_unsloth=self.config.use_unsloth,
            model_id=self.config.model_id,
        )

        tokenised_dataset = format_dataset_for_training(
            dataset, tokenizer, hyperparams, self.config.data_format,
        )

        split = self.config.train_test_split
        if split < 1.0:
            split_dataset = tokenised_dataset.train_test_split(
                train_size=split, seed=self.config.seed,
            )
            tokenised_dataset = split_dataset["train"]

        adapter_path = train(model, tokenizer, tokenised_dataset, self.config)

        # Preserve refs for the GGUF export stage.
        self._model = model
        self._tokenizer = tokenizer

        return adapter_path

    def _export_gguf(self, adapter_path):
        """Stage 4: Export to GGUF via Unsloth's native save_pretrained_gguf.

        Merges the LoRA adapter, dequantizes the 4-bit base, runs llama.cpp
        conversion, and quantizes to the requested level — all inside
        Unsloth.  No intermediate merged fp16 directory is produced.
        """
        gguf_path = save_gguf_unsloth(
            model=self._model,
            tokenizer=self._tokenizer,
            config=self.config,
            adapter_path=adapter_path,
        )

        # Release VRAM before the Ollama push stage (which can load more models).
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return gguf_path

    def _push_to_ollama(self, gguf_path):
        """Stage 5: Create the model in Ollama and optionally push to registry."""
        pusher = OllamaPusher(self.config)
        model_name = pusher.push(gguf_path)
        return model_name

    # ── Helpers ────────────────────────────────────────────────────────

    def _stage(self, name: str, fn, *args, **kwargs):
        """Run a pipeline stage with timing and error handling."""
        console.print(f"[cyan]▶ {name}[/cyan]")

        start = time.time()
        try:
            result = fn(*args, **kwargs)
            elapsed = time.time() - start
            self._stage_times.append((name, elapsed))
            console.print(f"[green]  ✓ {name} — {elapsed:.1f}s[/green]")
            return result
        except Exception:
            elapsed = time.time() - start
            self._stage_times.append((name, elapsed))
            console.print(f"[red]  ✗ {name} — {elapsed:.1f}s[/red]")
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
            f"[cyan]Model[/cyan]:      {self.config.model_id}\n"
            f"[cyan]Method[/cyan]:     {self.config.method}\n"
            f"[cyan]PDFs[/cyan]:       {len(pdf_paths)} file(s)\n"
            f"[cyan]Auto HP[/cyan]:    {self.config.auto_hyperparams}\n"
            f"[cyan]Master[/cyan]:     {self.config.master_model} ({self.config.master_backend})\n"
            f"[cyan]Quant[/cyan]:      {self.config.get_quantization_method()}\n"
            f"[cyan]Ollama[/cyan]:     {ollama_status}\n"
            f"[cyan]Namespace[/cyan]:  {namespace}\n"
            f"[cyan]Output[/cyan]:     {self.config.output_dir}\n"
            f"[cyan]GGUF out[/cyan]:   {self.config.get_gguf_output_dir()}"
        )

    def _print_summary(self, model_name: str, total_time: float, gguf_path: Path) -> None:
        """Print a final summary of the pipeline run."""
        table = Table(show_header=True, header_style="bold", border_style="cyan")
        table.add_column("Stage")
        table.add_column("Time", justify="right")

        for name, elapsed in self._stage_times:
            table.add_row(name, f"{elapsed:.1f}s")

        table.add_row("[bold]Total[/bold]", f"[bold]{total_time:.1f}s[/bold]")

        console.print()
        console.print(table)
        console.print(f"[dim]GGUF:[/dim] {gguf_path}")

        if not self.config.skip_ollama and "ollama" not in str(gguf_path).lower():
            namespace = self.config.get_ollama_namespace()
            local_name = self.config.get_ollama_model_name()
            full_name = self.config.get_ollama_full_name()

            if namespace and self.config.ollama_push:
                console.print(Panel(
                    f"[green]ollama pull {full_name}[/green]\n"
                    f"[green]ollama run {full_name}[/green]",
                    title="Pushed to registry",
                    border_style="green",
                ))
            else:
                console.print(Panel(
                    f"[green]ollama run {local_name}[/green]",
                    title="Local Ollama model ready",
                    border_style="green",
                ))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def finetune_and_push(config: FinetuneConfig) -> str:
    """One-function API for the full pipeline.

    Usage::

        from tuxtrainer.pipeline import finetune_and_push
        from tuxtrainer.config import FinetuneConfig

        config = FinetuneConfig(
            model_id="unsloth/Llama-3.2-1B-Instruct",
            pdf_paths=["doc1.pdf"],
        )
        model_name = finetune_and_push(config)
    """
    pipeline = FinetunePipeline(config)
    return pipeline.run()
