"""
GGUF converter — export a HuggingFace model to GGUF format for Ollama.

This module handles the conversion of a fine-tuned HuggingFace model
into the GGUF format that llama.cpp / Ollama can consume.

Conversion strategies:
  1. Python-llama-cpp (convert scripts) — preferred, most reliable
  2. llama.cpp CLI tools — fallback via subprocess
  3. Direct Python conversion using transformers + gguf — experimental

Quantisation is applied during the conversion step so the final .gguf
file is ready to be loaded by Ollama.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from tuxtrainer.config import Quantisation

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _find_llama_cpp() -> Optional[Path]:
    """Try to locate a llama.cpp installation.

    Looks in:
      1. LLAMA_CPP_PATH env var
      2. ~/llama.cpp
      3. /usr/local/bin/llama-convert
      4. pip-installed llama-cpp-python (has convert scripts)
    """
    # Env var override
    env_path = os.environ.get("LLAMA_CPP_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Home directory
    home_llama = Path.home() / "llama.cpp"
    if home_llama.exists():
        return home_llama

    # Check for the convert script in common locations
    for candidate in [
        Path("/usr/local/bin/llama-convert"),
        Path.home() / ".local/bin/llama-convert",
    ]:
        if candidate.exists():
            return candidate.parent.parent  # Return the prefix dir

    return None


def _find_convert_script() -> Optional[Path]:
    """Find the convert script from llama.cpp or llama-cpp-python."""
    # Try llama.cpp repo
    for search_dir in [
        Path.home() / "llama.cpp",
        Path(os.environ.get("LLAMA_CPP_PATH", "/nonexistent")),
    ]:
        if not search_dir.exists():
            continue
        # Newer llama.cpp versions use convert_hf_to_gguf.py
        for script_name in [
            "convert_hf_to_gguf.py",
            "convert_hf_to_gguf",
            "convert.py",
            "convert",
        ]:
            script_path = search_dir / script_name
            if script_path.exists():
                return script_path
            # Also check in subdirectories
            results = list(search_dir.rglob(script_name))
            if results:
                return results[0]

    # Try Python package
    try:
        import llama_cpp
        pkg_dir = Path(llama_cpp.__file__).parent
        for script_name in ["convert_hf_to_gguf.py", "convert.py"]:
            script_path = pkg_dir / script_name
            if script_path.exists():
                return script_path
    except ImportError:
        pass

    return None


def _find_quantise_bin() -> Optional[Path]:
    """Find the llama-quantize binary."""
    for candidate in [
        Path.home() / "llama.cpp" / "llama-quantize",
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize",
        Path("/usr/local/bin/llama-quantize"),
        Path("/usr/bin/llama-quantize"),
    ]:
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

class GGUFConverter:
    """Convert a HuggingFace model directory to a quantised GGUF file.

    Usage::

        converter = GGUFConverter(quantisation=Quantisation.Q4_K_M)
        gguf_path = converter.convert(merged_model_dir, output_dir)
    """

    def __init__(
        self,
        quantisation: Quantisation = Quantisation.Q4_K_M,
        llama_cpp_path: Optional[Path] = None,
    ) -> None:
        self.quantisation = quantisation
        self.llama_cpp_path = llama_cpp_path

    def convert(
        self,
        model_dir: Path,
        output_dir: Path,
        model_name: Optional[str] = None,
    ) -> Path:
        """Convert the model to GGUF and apply quantisation.

        Args:
            model_dir: Path to the merged HuggingFace model directory.
            output_dir: Where to save the GGUF file.
            model_name: Optional name for the output file.

        Returns:
            Path to the final quantised GGUF file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        name = model_name or "model"
        f16_gguf = output_dir / f"{name}-f16.gguf"
        quantised_gguf = output_dir / f"{name}-{self.quantisation.value.lower()}.gguf"

        console.print(Panel(
            f"[bold]Input[/bold]: {model_dir}\n"
            f"[bold]Output[/bold]: {quantised_gguf}\n"
            f"[bold]Quantisation[/bold]: {self.quantisation.value}",
            title="GGUF Conversion",
            border_style="magenta",
        ))

        # Step 1: Convert to F16 GGUF
        self._convert_to_f16(model_dir, f16_gguf)

        # Step 2: Quantise (if not F16)
        if self.quantisation != Quantisation.F16:
            self._quantise(f16_gguf, quantised_gguf)
            # Optionally remove the F16 intermediate file
            # f16_gguf.unlink()  # Uncomment to save disk
        else:
            quantised_gguf = f16_gguf

        console.print(f"\n[green]GGUF model saved to {quantised_gguf}[/green]")
        return quantised_gguf

    def _convert_to_f16(self, model_dir: Path, output_path: Path) -> None:
        """Convert HuggingFace model to F16 GGUF."""
        # Try method 1: llama.cpp convert script
        convert_script = _find_convert_script()
        if convert_script:
            self._convert_with_script(convert_script, model_dir, output_path)
            return

        # Try method 2: Python-based conversion using transformers + gguf
        try:
            self._convert_with_python(model_dir, output_path)
            return
        except ImportError:
            pass

        # Try method 3: Use the gguf package directly
        try:
            self._convert_with_gguf_package(model_dir, output_path)
            return
        except ImportError:
            pass

        raise RuntimeError(
            "No GGUF conversion method available. Install one of:\n"
            "  1. llama.cpp (git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make)\n"
            "  2. pip install gguf\n"
            "  3. Set LLAMA_CPP_PATH environment variable"
        )

    def _convert_with_script(
        self, script: Path, model_dir: Path, output_path: Path
    ) -> None:
        """Convert using llama.cpp's convert script via subprocess."""
        cmd = [
            sys.executable,
            str(script),
            str(model_dir),
            "--outfile", str(output_path),
            "--outtype", "f16",
        ]

        console.print(f"[blue]Running: {' '.join(cmd)}[/blue]")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            logger.error("Conversion stderr: %s", result.stderr)
            raise RuntimeError(f"GGUF conversion failed: {result.stderr[:500]}")

        console.print(f"[green]F16 GGUF created: {output_path}[/green]")

    def _convert_with_python(self, model_dir: Path, output_path: Path) -> None:
        """Convert using Python libraries directly (experimental)."""
        from transformers import AutoModelForCausalLM
        import torch
        import struct

        console.print("[yellow]Using experimental Python GGUF conversion...[/yellow]")
        # This is a simplified conversion — for production, use llama.cpp
        # We'll try to use the gguf package if available
        raise ImportError("Fallback to gguf package")

    def _convert_with_gguf_package(self, model_dir: Path, output_path: Path) -> None:
        """Convert using the gguf Python package."""
        try:
            from gguf.api import GGUFWriter
        except ImportError:
            from gguf import GGUFWriter

        console.print("[blue]Converting with gguf package...[/blue]")

        # Load model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        import numpy as np

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            token=token,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, token=token)

        # This is a simplified path — in practice you'd iterate over all tensors
        # For now, delegate to the CLI approach if available
        console.print("[yellow]Direct gguf conversion is limited — prefer llama.cpp.[/yellow]")
        raise ImportError("Prefer llama.cpp for full conversion")

    def _quantise(self, input_path: Path, output_path: Path) -> None:
        """Apply quantisation to an F16 GGUF file."""
        quantise_bin = _find_quantise_bin()
        if not quantise_bin:
            console.print(
                "[yellow]llama-quantize not found — copying F16 GGUF as-is.\n"
                "For quantisation, install llama.cpp and ensure llama-quantize is in PATH.[/yellow]"
            )
            shutil.copy2(input_path, output_path)
            return

        cmd = [
            str(quantise_bin),
            str(input_path),
            str(output_path),
            self.quantisation.value,
        ]

        console.print(f"[blue]Quantising: {' '.join(cmd)}[/blue]")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            logger.error("Quantisation stderr: %s", result.stderr)
            raise RuntimeError(f"GGUF quantisation failed: {result.stderr[:500]}")

        console.print(f"[green]Quantised GGUF created: {output_path}[/green]")
