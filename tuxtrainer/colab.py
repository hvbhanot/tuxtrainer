"""
Colab setup helper — bootstrap environment for Google Colab.

Since tuxtrainer uses the **Ollama Cloud API** by default for the
master model, you don't need to install Ollama locally just to pick
hyperparameters.  However, since the default behaviour is to **push
the model to the Ollama registry** (so you can use it on any device),
Ollama needs to be installed on Colab to create and push the model.

This helper installs everything needed for the full pipeline:
PyTorch, llama.cpp for GGUF conversion, and Ollama for model serving.

After pushing, you can pull the model on any device:
``ollama pull your-namespace/model-name``
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

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


def setup_colab(
    install_ollama: bool = True,
    install_llama_cpp: bool = True,
    pull_master_model: str | None = None,
    ollama_host: str = "http://localhost:11434",
) -> None:
    """Bootstrap the Colab environment for tuxtrainer.

    By default this **installs Ollama** because the model is pushed to the
    Ollama registry by default (so you can pull it on any device).  The
    master model uses the Ollama Cloud API (set ``OLLAMA_API_KEY``) — it
    doesn't need a local Ollama instance.

    Args:
        install_ollama: Install Ollama locally on the Colab VM.
            Default is True because the pipeline pushes to the registry.
        install_llama_cpp: Install llama.cpp for GGUF conversion.
            Recommended — needed to convert the fine-tuned model.
        pull_master_model: If installing Ollama, also pull this model
            for local master model usage.  Not needed if using the
            cloud API (the default).
        ollama_host: Ollama API host (only relevant if install_ollama=True).
    """
    import requests

    if not _is_colab():
        console.print(
            "[yellow]Warning: Not running on Colab. "
            "Some steps may not work as expected.[/yellow]"
        )

    # ── Step 1: Install Ollama (optional) ──────────────────────────────
    if install_ollama:
        console.print("[bold blue]Installing Ollama on Colab VM...[/bold blue]")
        try:
            result = subprocess.run(
                ['bash', '-c', 'curl -fsSL https://ollama.com/install.sh | sh'],
                capture_output=True,
                text=True,
                timeout=180,
            )
            console.print("[green]Ollama installed.[/green]")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            console.print(f"[red]Ollama install failed: {e}[/red]")
            raise

        # Start the server in the background
        console.print("[blue]Starting Ollama server...[/blue]")
        subprocess.Popen(
            ["ollama", "serve"],
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
                ["ollama", "pull", pull_master_model],
                capture_output=True,
                text=True,
                timeout=600,
            )
            console.print(f"[green]Model '{pull_master_model}' pulled.[/green]")

    # ── Step 2: Install llama.cpp for GGUF conversion ──────────────────
    if install_llama_cpp:
        console.print("[bold blue]Installing llama.cpp for GGUF conversion...[/bold blue]")
        try:
            subprocess.run(
                ['bash', '-c',
                 'cd /content && '
                 'git clone --depth 1 https://github.com/ggerganov/llama.cpp && '
                 'cd llama.cpp && make -j$(nproc)'],
                capture_output=True,
                text=True,
                timeout=300,
            )
            os.environ["LLAMA_CPP_PATH"] = "/content/llama.cpp"
            console.print("[green]llama.cpp installed at /content/llama.cpp[/green]")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            console.print(f"[yellow]llama.cpp install failed: {e}[/yellow]")
            console.print("[yellow]You can still fine-tune, but GGUF conversion may not work.[/yellow]")

    # ── Done ───────────────────────────────────────────────────────────
    console.print("\n[bold green]Colab setup complete![/bold green]")

    # Check if the Ollama API key is set (for master model)
    if os.environ.get("OLLAMA_API_KEY"):
        console.print("[green]OLLAMA_API_KEY is set — master model will use Ollama Cloud.[/green]")
    else:
        console.print(
            "[yellow]OLLAMA_API_KEY not set. Set it for the master model to use Ollama Cloud:[/yellow]\n"
            "  [dim]import os; os.environ['OLLAMA_API_KEY'] = 'your-key-here'[/dim]\n"
            "  [dim]Or use --master-backend openai with OPENAI_API_KEY[/dim]"
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
        "  config = FinetuneConfig(model_id='...', pdf_paths=[...])\n"
        "  FinetunePipeline(config).run()\n"
        "  # → ollama pull your-namespace/model-name[/dim]"
    )
