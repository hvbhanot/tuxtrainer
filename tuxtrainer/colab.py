"""
Colab setup helper — bootstrap environment for Google Colab.

Since tuxtrainer uses the **Ollama Cloud API** by default for the
master model, you don't need to install Ollama locally just to pick
hyperparameters.  However, since the default behaviour is to **push
the model to the Ollama registry** (so you can use it on any device),
Ollama needs to be installed on Colab to create and push the model.

This helper installs everything needed for the full pipeline:
the Unsloth-compatible Python dependency set, PyMuPDF / dataset tooling,
and Ollama for model serving.
Unsloth handles the GGUF conversion toolchain itself during
``save_pretrained_gguf``.

After pushing, you can pull the model on any device:
``ollama pull your-namespace/model-name``
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import warnings

from rich.console import Console

console = Console()


def _is_colab() -> bool:
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        pass
    return bool(os.environ.get("COLAB_GPU") or os.environ.get("COLAB_TPU_ADDR"))


def _run(cmd: str, timeout: int = 180, check: bool = True) -> subprocess.CompletedProcess:
    """Run a bash command with sensible defaults."""
    return subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def _install_system_deps() -> None:
    """Install system-level dependencies that Ollama needs on Colab.

    The Ollama install script requires ``zstd`` for extracting its
    tarball.  Colab VMs don't ship with it, so we install it first.
    """
    console.print("[bold blue]Installing system dependencies (zstd)...[/bold blue]")
    try:
        _run("apt-get update -qq && apt-get install -y -qq zstd", timeout=120)
        console.print("[green]zstd installed.[/green]")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        console.print("[yellow]zstd install failed (non-fatal) — Ollama install may fail without it[/yellow]")


def _is_ollama_installed() -> bool:
    """Check if the ``ollama`` binary is on PATH."""
    return shutil.which("ollama") is not None


def _get_ollama_path() -> str:
    """Return the absolute path to the ollama binary.

    Checks ``shutil.which()`` first, then falls back to common
    installation locations used by the official install script.
    """
    path = shutil.which("ollama")
    if path:
        return path

    candidates = [
        "/usr/local/bin/ollama",
        "/usr/bin/ollama",
        os.path.expanduser("~/.ollama/bin/ollama"),
        "/opt/ollama/bin/ollama",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        "ollama binary not found in PATH or common locations. "
        "Try reinstalling: !apt-get install -y zstd && curl -fsSL https://ollama.com/install.sh | sh"
    )


def setup_colab(
    install_ollama: bool = True,
    install_llama_cpp: bool | None = None,
    pull_master_model: str | None = None,
    ollama_host: str = "http://localhost:11434",
) -> None:
    """Bootstrap the Colab environment for tuxtrainer.

    By default this **installs Ollama** because the model is pushed to the
    Ollama registry by default (so you can pull it on any device).  The
    master model uses the Ollama Cloud API (set ``OLLAMA_API_KEY``) — it
    doesn't need a local Ollama instance.

    Also pins the Python dependency set to the Unsloth-compatible versions
    used by tuxtrainer and installs the rest of tuxtrainer's runtime
    dependencies. Unsloth builds llama.cpp internally when needed for GGUF
    export, so this helper does not install or run any manual
    ``llama.cpp`` conversion scripts.

    Args:
        install_ollama: Install Ollama locally on the Colab VM.
            Default is True because the pipeline pushes to the registry.
        install_llama_cpp: Deprecated and ignored. Unsloth handles the GGUF
            conversion toolchain itself.
        pull_master_model: If installing Ollama, also pull this model
            for local master model usage.  Not needed if using the
            cloud API (the default).
        ollama_host: Ollama API host (only relevant if install_ollama=True).
    """
    import requests

    if install_llama_cpp is not None:
        warnings.warn(
            "'install_llama_cpp' is deprecated and ignored. "
            "Unsloth handles the GGUF conversion toolchain internally.",
            DeprecationWarning,
            stacklevel=2,
        )

    if not _is_colab():
        console.print(
            "[yellow]Warning: Not running on Colab. "
            "Some steps may not work as expected.[/yellow]"
        )

    # ── Step 0: Pin the Python stack to Unsloth-compatible versions ────
    console.print("[bold blue]Pinning the Python stack for Unsloth GGUF export...[/bold blue]")
    try:
        _run(
            (
                'pip install -q --upgrade '
                '"transformers==4.55.4" '
                '"tokenizers>=0.21,<0.22" '
                '"trl==0.22.2" '
                '"peft>=0.13,<0.17" '
                '"accelerate>=0.34" '
                '"bitsandbytes>=0.44" '
                '"unsloth" '
                '"unsloth_zoo" '
                '"datasets>=2.16.0" '
                '"pymupdf>=1.23.0" '
                '"rich>=13.0.0" '
                '"click>=8.1.0" '
                '"pydantic>=2.5.0" '
                '"huggingface-hub>=0.20.0" '
                '"sentencepiece>=0.1.99" '
                '"protobuf>=3.20.0" '
                '"scipy>=1.11.0" '
                '"requests>=2.28.0"'
            ),
            timeout=300,
            check=False,
        )
        console.print("[green]Pinned Python dependencies installed.[/green]")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        console.print("[yellow]Dependency pin step skipped.[/yellow]")

    # ── Step 1: Install Ollama (optional) ──────────────────────────────
    if install_ollama:
        # Check if already installed
        if _is_ollama_installed():
            console.print("[green]Ollama is already installed.[/green]")
        else:
            # Install system deps first (zstd is required by Ollama's install script)
            _install_system_deps()

            console.print("[bold blue]Installing Ollama on Colab VM...[/bold blue]")
            try:
                result = _run(
                    'curl -fsSL https://ollama.com/install.sh | sh',
                    timeout=180,
                    check=False,
                )
                if result.returncode != 0:
                    console.print("[red]Ollama install failed.[/red]")
                    raise RuntimeError(
                        "Ollama installation failed. Try manually:\n"
                        "  !apt-get install -y zstd && curl -fsSL https://ollama.com/install.sh | sh"
                    )
                console.print("[green]Ollama installed.[/green]")
            except subprocess.TimeoutExpired:
                console.print("[red]Ollama install timed out[/red]")
                raise

        # Verify the binary exists before trying to start it
        if not _is_ollama_installed():
            raise RuntimeError(
                "Ollama binary not found after installation. "
                "The install may have failed silently. "
                "Try manually: !apt-get install -y zstd && curl -fsSL https://ollama.com/install.sh | sh"
            )

        # Start the server in the background
        console.print("[blue]Starting Ollama server...[/blue]")
        ollama_path = _get_ollama_path()
        subprocess.Popen(
            [ollama_path, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "OLLAMA_HOST": "0.0.0.0"},
        )

        for _ in range(30):
            try:
                resp = requests.get(f"{ollama_host}/api/tags", timeout=2)
                if resp.status_code == 200:
                    console.print("[green]Ollama server is running.[/green]")
                    break
            except requests.ConnectionError:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("Ollama server did not start within 60 seconds.")

        if pull_master_model:
            console.print(f"[blue]Pulling model '{pull_master_model}'...[/blue]")
            subprocess.run(
                [ollama_path, "pull", pull_master_model],
                capture_output=True,
                text=True,
                timeout=600,
            )
            console.print(f"[green]Model '{pull_master_model}' pulled.[/green]")

    # ── Done ───────────────────────────────────────────────────────────
    console.print("\n[bold green]Colab setup complete![/bold green]")

    # Runtime restart warning
    console.print(
        "\n[yellow]If you just re-installed tuxtrainer, you MUST restart the runtime:[/yellow]\n"
        "  [dim]Runtime → Restart session (Ctrl+M .)[/dim]\n"
        "[yellow]Otherwise the old cached code will be used.[/yellow]"
    )

    # Check if the Ollama API key is set (for master model)
    if os.environ.get("OLLAMA_API_KEY"):
        console.print("[green]OLLAMA_API_KEY is set — master model will use Ollama Cloud.[/green]")
    else:
        console.print(
            "[yellow]OLLAMA_API_KEY not set. The master model will fall back to local Ollama:[/yellow]\n"
            "  [dim]config = FinetuneConfig(..., master_backend='ollama')[/dim]\n"
            "  [dim]Or set OLLAMA_API_KEY for Ollama Cloud[/dim]"
        )

    # Check if the Ollama namespace is set (for registry push)
    if os.environ.get("OLLAMA_NAMESPACE"):
        console.print(f"[green]OLLAMA_NAMESPACE is set — model will be pushed as {os.environ['OLLAMA_NAMESPACE']}/<model>[/green]")
    else:
        console.print(
            "[yellow]OLLAMA_NAMESPACE not set. Set it to push the model to the registry[/yellow]\n"
            "  [dim]so you can pull it on any device:[/dim]\n"
            "  [dim]import os; os.environ['OLLAMA_NAMESPACE'] = 'your-ollama-username'[/dim]\n"
            "  [dim]Or use --ollama-namespace your_username[/dim]"
        )

    console.print("\n[dim]Now run the pipeline:[/dim]")
    console.print(
        "[dim]  from tuxtrainer import FinetuneConfig, FinetunePipeline\n"
        "  config = FinetuneConfig(\n"
        "      model_id='unsloth/Llama-3.2-1B-Instruct',\n"
        "      pdf_paths=[...]\n"
        "  )\n"
        "  FinetunePipeline(config).run()\n"
        "  # → ollama pull your-namespace/model-name[/dim]"
    )
