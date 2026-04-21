"""
Ollama pusher — create a model and push it to Ollama via the Web API.

Designed for Google Colab and headless environments where the Ollama CLI
is not available.  All interactions use the Ollama REST API:

  * ``POST /api/create``  — create a model from a Modelfile string
  * ``POST /api/push``    — push a model to the Ollama registry
  * ``GET  /api/tags``    — list available models

By default, the model is **pushed to the Ollama registry** so you can
pull it on any device with ``ollama pull <namespace>/<model-name>``.
Set your Ollama namespace via the ``OLLAMA_NAMESPACE`` env var or the
``--ollama-namespace`` CLI option.

The module also includes a helper to install and start Ollama on a Colab
VM so the GGUF file can be served locally before being pushed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests
from rich.console import Console
from rich.panel import Panel

from tuxtrainer.config import FinetuneConfig

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GGUF resolution
# ---------------------------------------------------------------------------

def _resolve_gguf_file(path: Path) -> Path:
    """Return a concrete ``.gguf`` file from either a file path or a directory.

    The pipeline hands us whatever Unsloth produced — in practice a single
    file in ``config.gguf_output_dir``, but filename conventions have
    changed across Unsloth versions (``unsloth.Q4_K_M.gguf``,
    ``<model>-q4_k_m.gguf``, ...).  Globbing for ``*.gguf`` keeps us robust
    to those renames.
    """
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        matches = sorted(path.glob("*.gguf"))
        if not matches:
            raise FileNotFoundError(
                f"No .gguf file found in {path}. "
                "Did Unsloth's save_pretrained_gguf complete successfully?"
            )
        if len(matches) > 1:
            # Prefer a non-f16 file if the user asked for a quantised model.
            non_f16 = [p for p in matches if "f16" not in p.stem.lower()]
            if non_f16:
                matches = non_f16
        return matches[0]
    raise FileNotFoundError(f"GGUF path does not exist: {path}")


# ---------------------------------------------------------------------------
# Modelfile generation
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, knowledgeable assistant that has been fine-tuned on \
specific domain documents. Answer questions accurately based on the \
knowledge you have acquired during fine-tuning. If you are unsure about \
something, say so rather than making up information.
"""

MODELFILE_TEMPLATE = """\
FROM {gguf_path}

# System prompt
SYSTEM \"\"\"{system_prompt}\"\"\"

# Inference parameters
TEMPLATE \"\"\"{{{{- if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{- end }}}}
<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>\"\"\"

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER top_k {top_k}
PARAMETER num_ctx {num_ctx}
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
"""


def generate_modelfile(
    gguf_path: Path,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    num_ctx: int = 4096,
    template: Optional[str] = None,
) -> str:
    """Generate an Ollama Modelfile string.

    Args:
        gguf_path: Path to the GGUF model file (must be accessible to the
            Ollama server — on Colab this is a local path on the VM).
        system_prompt: System prompt for the model.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        top_k: Top-K sampling parameter.
        num_ctx: Context window size.
        template: Custom prompt template. If None, uses a LLaMA-style chat template.

    Returns:
        The Modelfile content as a string.
    """
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    if template:
        content = (
            f"FROM {gguf_path}\n\n"
            f'SYSTEM """{sys_prompt}"""\n\n'
            f'TEMPLATE """{template}"""\n\n'
            f"PARAMETER temperature {temperature}\n"
            f"PARAMETER top_p {top_p}\n"
            f"PARAMETER top_k {top_k}\n"
            f"PARAMETER num_ctx {num_ctx}\n"
        )
    else:
        content = MODELFILE_TEMPLATE.format(
            gguf_path=gguf_path,
            system_prompt=sys_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_ctx=num_ctx,
        )

    return content


# ---------------------------------------------------------------------------
# Ollama Web API helpers
# ---------------------------------------------------------------------------

def _api_url(host: str, path: str) -> str:
    """Build a full API URL, ensuring proper formatting."""
    return f"{host.rstrip('/')}{path}"


def check_ollama_running(host: str = "http://localhost:11434") -> bool:
    """Check if the Ollama server is reachable via the Web API."""
    try:
        resp = requests.get(_api_url(host, "/api/tags"), timeout=5)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def wait_for_ollama(host: str = "http://localhost:11434", timeout: int = 60) -> bool:
    """Wait for the Ollama server to become available.

    On Colab, Ollama can take a few seconds to start up after
    installation.  This polls until it responds or the timeout expires.
    """
    start = time.time()
    while time.time() - start < timeout:
        if check_ollama_running(host):
            return True
        time.sleep(2)
    return False


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


def install_ollama_colab() -> bool:
    """Install and start Ollama on a Google Colab VM.

    Runs the official install script and starts the server in the
    background.  This only works on Linux (which Colab VMs are).

    The Ollama install script requires ``zstd`` for extraction, which
    is not pre-installed on Colab VMs, so we install it first.

    Returns:
        True if Ollama was installed and started successfully.
    """
    # Install zstd first (required by Ollama's install script on Colab)
    console.print("[blue]Installing zstd (required by Ollama installer)...[/blue]")
    try:
        subprocess.run(
            ['bash', '-c', 'apt-get update -qq && apt-get install -y -qq zstd'],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        console.print("[yellow]zstd install failed — Ollama install may fail without it.[/yellow]")

    console.print("[bold blue]Installing Ollama on Colab VM...[/bold blue]")

    # Download and run the official install script
    try:
        result = subprocess.run(
            ['bash', '-c', 'curl -fsSL https://ollama.com/install.sh | sh'],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            logger.error("Ollama install failed: %s", result.stderr)
            console.print(f"[red]Install failed: {result.stderr[:300]}[/red]")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        console.print("[red]Could not run Ollama install script.[/red]")
        return False

    # Start the server in the background
    console.print("[blue]Starting Ollama server...[/blue]")
    try:
        ollama_path = _get_ollama_path()
    except FileNotFoundError:
        console.print(
            "[red]Ollama install script ran but binary not found. "
            "Try restarting the runtime and running again.[/red]"
        )
        return False

    subprocess.Popen(
        [ollama_path, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "OLLAMA_HOST": "0.0.0.0"},
    )

    # Wait for it to come up
    if wait_for_ollama(timeout=60):
        console.print("[green]Ollama installed and running on Colab![/green]")
        return True
    else:
        console.print("[red]Ollama started but did not respond in time.[/red]")
        return False


def ensure_ollama(host: str = "http://localhost:11434", auto_install: bool = True) -> bool:
    """Make sure Ollama is running; install it on Colab if needed.

    Args:
        host: Ollama API host URL.
        auto_install: If True and Ollama is not running, try to install
            it (works on Colab / Linux VMs).

    Returns:
        True if Ollama is reachable.
    """
    if check_ollama_running(host):
        return True

    if not auto_install:
        raise RuntimeError(
            f"Ollama is not running at {host}. "
            "Start it manually or set auto_install=True."
        )

    console.print("[yellow]Ollama is not running — attempting auto-install...[/yellow]")
    if install_ollama_colab():
        return True

    raise RuntimeError(
        "Could not start Ollama. On Colab, run:\n"
        "  !curl -fsSL https://ollama.com/install.sh | sh\n"
        "  import subprocess; subprocess.Popen(['ollama', 'serve'])\n"
        "Or set --ollama-host to a remote Ollama instance."
    )


# ---------------------------------------------------------------------------
# Create / Push via Web API
# ---------------------------------------------------------------------------

def create_ollama_model(
    model_name: str,
    modelfile_content: str,
    host: str = "http://localhost:11434",
    timeout: int = 600,
) -> bool:
    """Create a model in Ollama using the Web API.

    Uses ``POST /api/create`` which accepts the full Modelfile content
    as a JSON string — no CLI required.

    Args:
        model_name: Name for the model in Ollama (e.g. "my-expert").
        modelfile_content: The full Modelfile text (FROM, SYSTEM, etc.).
        host: Ollama API host URL.
        timeout: Request timeout in seconds (large models take a while).

    Returns:
        True if creation succeeded.
    """
    url = _api_url(host, "/api/create")
    payload = {
        "name": model_name,
        "modelfile": modelfile_content,
        "stream": False,
    }

    console.print(f"[blue]Creating Ollama model '{model_name}' via Web API...[/blue]")
    console.print(f"[dim]POST {url}[/dim]")

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()

        # The API may return streamed JSON lines or a single object
        # With stream=False we get a single response
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Ollama API error: {data['error']}")

        console.print(f"[green]Model '{model_name}' created in Ollama![/green]")
        return True

    except requests.exceptions.HTTPError as e:
        # Try to extract the error message from the response
        try:
            err_body = e.response.json()
            err_msg = err_body.get("error", str(e))
        except (json.JSONDecodeError, AttributeError):
            err_msg = str(e)
        logger.error("Ollama create failed: %s", err_msg)
        raise RuntimeError(f"Failed to create Ollama model: {err_msg}") from e

    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {host}. "
            "Make sure Ollama is running and the host is correct."
        ) from e


def push_ollama_model(
    model_name: str,
    host: str = "http://localhost:11434",
    timeout: int = 600,
) -> bool:
    """Push a local Ollama model to the registry via the Web API.

    Uses ``POST /api/push``.  The model name must include a namespace
    (e.g. ``username/model-name``) to push to the registry.

    Args:
        model_name: Full model name including namespace.
        host: Ollama API host URL.
        timeout: Request timeout in seconds.

    Returns:
        True if push succeeded.
    """
    if "/" not in model_name:
        console.print(
            "[yellow]Model name must include a namespace to push "
            "(e.g. 'myuser/my-model'). Skipping push.[/yellow]"
        )
        return False

    url = _api_url(host, "/api/push")
    payload = {
        "name": model_name,
        "stream": False,
    }

    console.print(f"[blue]Pushing '{model_name}' to Ollama registry via Web API...[/blue]")

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()

        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Ollama push error: {data['error']}")

        console.print(f"[green]Model '{model_name}' pushed to registry![/green]")
        return True

    except requests.exceptions.HTTPError as e:
        try:
            err_body = e.response.json()
            err_msg = err_body.get("error", str(e))
        except (json.JSONDecodeError, AttributeError):
            err_msg = str(e)
        logger.error("Ollama push failed: %s", err_msg)
        raise RuntimeError(f"Failed to push model: {err_msg}") from e


def list_ollama_models(host: str = "http://localhost:11434") -> list[str]:
    """List all models currently available in Ollama via the Web API.

    Uses ``GET /api/tags``.
    """
    try:
        resp = requests.get(_api_url(host, "/api/tags"), timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except (requests.ConnectionError, requests.Timeout):
        return []


def chat_with_model(
    model_name: str,
    message: str,
    host: str = "http://localhost:11434",
    timeout: int = 120,
) -> str:
    """Quick chat test with a model in Ollama via the Web API.

    Uses ``POST /api/chat``.  Handy for verifying that a pushed model
    works, especially on Colab where you can't use ``ollama run``.
    """
    url = _api_url(host, "/api/chat")
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": message}],
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


# ---------------------------------------------------------------------------
# Main pusher class
# ---------------------------------------------------------------------------

class OllamaPusher:
    """Push a fine-tuned GGUF model to Ollama via the Web API.

    Works on Google Colab and any headless environment — no CLI needed.
    All interactions use the Ollama REST API over HTTP.

    By default, the model is pushed to the Ollama registry so it can be
    pulled on any device.  Set ``ollama_namespace`` (or ``OLLAMA_NAMESPACE``
    env var) to your Ollama username for registry push.

    Usage::

        pusher = OllamaPusher(config)
        model_name = pusher.push(gguf_path)
        # → Model available on any device: ollama pull myuser/my-model
    """

    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config

    def push(
        self,
        gguf_path: Path,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Create the model in Ollama and push it to the registry.

        On Colab, this will auto-install Ollama if it's not running,
        then create the model via the Web API and push it to the
        Ollama registry so it's available on any device.

        The local model name is always created (e.g. ``llama-3.1-8b-finetuned``).
        If ``ollama_namespace`` is set, it is also pushed as
        ``namespace/llama-3.1-8b-finetuned`` to the registry.

        Args:
            gguf_path: Path to the quantised GGUF file, OR a directory
                containing a single ``.gguf`` file (the pipeline passes
                the directory Unsloth wrote into and we glob for the file
                here to be robust to Unsloth's renaming conventions
                across versions).
            system_prompt: Custom system prompt for the model.

        Returns:
            The full Ollama model name (with namespace if set).
        """
        gguf_path = _resolve_gguf_file(gguf_path)

        local_name = self.config.get_ollama_model_name()
        full_name = self.config.get_ollama_full_name()
        namespace = self.config.get_ollama_namespace()
        host = self.config.ollama_host

        # Ensure Ollama is running (auto-installs on Colab)
        ensure_ollama(host=host, auto_install=True)

        # Generate Modelfile content
        modelfile_content = generate_modelfile(
            gguf_path=gguf_path,
            system_prompt=system_prompt,
        )

        # Save Modelfile for inspection / debugging
        output_dir = gguf_path.parent
        modelfile_path = output_dir / "Modelfile"
        modelfile_path.write_text(modelfile_content, encoding="utf-8")

        push_target = full_name if namespace else local_name

        # Create the local model first
        create_ollama_model(
            model_name=local_name,
            modelfile_content=modelfile_content,
            host=host,
        )

        # If namespace is set, also create with full name for registry push
        if namespace and full_name != local_name:
            console.print(f"[cyan]Creating namespaced model '{full_name}'...[/cyan]")
            create_ollama_model(
                model_name=full_name,
                modelfile_content=modelfile_content,
                host=host,
            )

        # Push to registry (default behaviour)
        if self.config.ollama_push:
            if namespace:
                push_ollama_model(full_name, host=host)
                console.print(f"[green]  Pushed {full_name} to registry[/green]")
            else:
                console.print("[yellow]  Registry push skipped — no namespace set[/yellow]")

        # Verify local model exists
        existing = list_ollama_models(host)
        found_local = local_name in existing or f"{local_name}:latest" in existing
        if found_local:
            if not self.config.ollama_push or not namespace:
                console.print(
                    f"\n[bold green]Model '{local_name}' is ready in local Ollama![/bold green]"
                )
            # On Colab, suggest testing via API
            if _is_colab():
                console.print(
                    f"[dim]Test it with the API:\n"
                    f"  from tuxtrainer.ollama_pusher import chat_with_model\n"
                    f"  chat_with_model('{local_name}', 'Hello!')[/dim]"
                )
            else:
                console.print(f"[dim]Test it locally: ollama run {local_name}[/dim]")

        return full_name


# ---------------------------------------------------------------------------
# Colab detection
# ---------------------------------------------------------------------------

def _is_colab() -> bool:
    """Detect whether we're running inside Google Colab."""
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        pass
    # Also check for the Colab environment variable
    return bool(os.environ.get("COLAB_GPU") or os.environ.get("COLAB_TPU_ADDR"))
