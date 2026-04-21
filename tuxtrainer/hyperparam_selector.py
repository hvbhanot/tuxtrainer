"""
Master model hyperparameter selector.

Uses an LLM (the "master model") to analyse the dataset statistics and
pick optimal hyperparameters for fine-tuning.  Supports multiple backends:
  * Ollama Cloud — remote Ollama Web API (default, no local install needed)
  * Ollama      — local model via Ollama API
  * OpenAI      — GPT-4 / GPT-4o etc.
  * HF API      — HuggingFace Inference Endpoints
  * ZAI SDK     — z-ai-web-dev-sdk

The master model receives a structured prompt with dataset statistics,
available GPU memory (if detectable), and returns a JSON blob of
hyperparameter choices that are validated against the HyperParams schema.

**Ollama Cloud (default)** uses the native Ollama Web API format
(``POST /api/chat``) over HTTPS with API-key authentication — no local
Ollama installation required.  Set ``OLLAMA_API_KEY`` or pass
``--master-api-key``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from typing import Any, Optional

import requests
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from tuxtrainer.config import (
    FinetuneConfig,
    HyperParams,
    MasterModelBackend,
)

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt for the master model
# ---------------------------------------------------------------------------

MASTER_SYSTEM_PROMPT = """\
You are an expert machine-learning engineer specialising in LLM fine-tuning.
Your task is to select optimal hyperparameters for LoRA / QLoRA fine-tuning
given the dataset statistics and hardware constraints below.

RULES:
1. Return ONLY valid JSON matching the HyperParams schema — no markdown,
   no explanation, no commentary, just the JSON object.
2. Every field must be present (use the defaults if unsure).
3. Be conservative with memory: prefer lower lora_r and batch sizes unless
   you are confident the hardware can handle more.
4. For small datasets (<10k tokens), use fewer epochs and higher learning
   rates.  For large datasets (>100k tokens), use more epochs and lower LR.
5. For QLoRA, lora_r of 16–64 is typical.  For LoRA on a big model, 32–128.
6. Always set bf16=true for Ampere+ GPUs (A100, RTX 30xx/40xx), fp16=true
   only for older GPUs (V100, T4).
7. gradient_checkpointing should be true when VRAM is tight.
"""

MASTER_USER_PROMPT_TEMPLATE = """\
## Dataset Statistics
- Total chunks: {total_chunks}
- Estimated total tokens: {total_tokens:,}
- Average tokens per chunk: {avg_chunk_tokens:.0f}
- Min / Max chunk tokens: {min_chunk_tokens} / {max_chunk_tokens}
- Number of source documents: {num_sources}
- Data format: {data_format}

## Model
- HuggingFace model ID: {model_id}
- Fine-tuning method: {method}

## Hardware
- Detected GPU: {gpu_info}
- Estimated available VRAM: {vram_gb} GB

## Schema (return JSON with these fields)
```json
{{
  "lora_r": int (1-256),
  "lora_alpha": int (≥1, typically 2× lora_r),
  "lora_dropout": float (0.0-0.5),
  "lora_target_modules": [str],
  "num_train_epochs": int (1-50),
  "per_device_train_batch_size": int (1-64),
  "gradient_accumulation_steps": int (1-64),
  "learning_rate": float (1e-6 to 1e-2),
  "lr_scheduler_type": "cosine" | "linear" | "constant",
  "warmup_ratio": float (0.0-0.5),
  "weight_decay": float (0.0-0.1),
  "max_seq_length": int (128-8192),
  "optim": "adamw_8bit" | "adamw_torch",
  "fp16": bool,
  "bf16": bool,
  "gradient_checkpointing": bool,
  "logging_steps": int,
  "save_strategy": "epoch" | "steps",
  "eval_strategy": "epoch" | "steps"
}}
```

Select the optimal hyperparameters now. Return ONLY the JSON.
"""


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_gpu_info() -> tuple[str, float]:
    """Try to detect the GPU name and available VRAM.

    Returns:
        (gpu_name, vram_gb) — defaults to ("Unknown", 16.0) if detection fails.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split(",")
                gpu_name = parts[0].strip()
                vram_mb = float(parts[1].strip())
                return gpu_name, vram_mb / 1024.0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "Unknown (nvidia-smi not available)", 16.0


# ---------------------------------------------------------------------------
# Backend callers
# ---------------------------------------------------------------------------

def _call_ollama_cloud(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 120,
) -> str:
    """Call a model via the Ollama Cloud Web API.

    Uses the native Ollama ``POST /api/chat`` endpoint over HTTPS with
    API-key authentication.  No local Ollama installation is needed.

    Authentication: the API key is sent as a Bearer token in the
    ``Authorization`` header.  Set ``OLLAMA_API_KEY`` in your environment
    or pass it via the ``api_key`` parameter.

    Default base URL: ``https://ollama.com``
    Override with ``OLLAMA_CLOUD_URL`` env var or ``base_url`` parameter.
    """
    resolved_key = api_key or os.environ.get("OLLAMA_API_KEY")
    resolved_url = base_url or os.environ.get(
        "OLLAMA_CLOUD_URL", "https://ollama.com"
    )

    url = f"{resolved_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
        },
    }

    headers = {"Content-Type": "application/json"}
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"

    logger.info("Ollama Cloud: POST %s (model=%s)", url, model)

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)

    if resp.status_code == 401:
        raise PermissionError(
            "Ollama Cloud API returned 401 Unauthorized. "
            "Set OLLAMA_API_KEY env var or pass --master-api-key."
        )
    resp.raise_for_status()

    data = resp.json()

    # Ollama returns errors in the JSON body with HTTP 200
    if "error" in data:
        error_msg = data["error"]
        if "not found" in error_msg.lower():
            raise RuntimeError(
                f"Model '{model}' is not available in Ollama Cloud. "
                f"Check the model name or try a different model."
            )
        raise RuntimeError(f"Ollama Cloud API error: {error_msg}")

    content = data.get("message", {}).get("content", "")
    if not content or not content.strip():
        raise RuntimeError(
            f"Model '{model}' returned an empty response from Ollama Cloud."
        )
    return content


def _call_ollama(
    model: str,
    system_prompt: str,
    user_prompt: str,
    host: str = "http://localhost:11434",
    timeout: int = 120,
) -> str:
    """Call a model via the Ollama API."""
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
        },
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # Ollama returns errors in the JSON body with HTTP 200
    if "error" in data:
        error_msg = data["error"]
        if "not found" in error_msg.lower():
            raise RuntimeError(
                f"Model '{model}' is not available in local Ollama. "
                f"Pull it first: ollama pull {model}"
            )
        raise RuntimeError(f"Ollama API error: {error_msg}")

    content = data.get("message", {}).get("content", "")
    if not content or not content.strip():
        raise RuntimeError(
            f"Model '{model}' returned an empty response. "
            "The model may be loading or misconfigured. "
            f"Try pulling it again: ollama pull {model}"
        )
    return content


def _call_openai(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 120,
) -> str:
    """Call a model via the OpenAI-compatible API."""
    import openai

    client = openai.OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
        response_format={"type": "json_object"},
        timeout=timeout,
    )
    return response.choices[0].message.content or ""


def _call_zai_sdk(
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = 120,
) -> str:
    """Call a model via the z-ai-web-dev-sdk.

    The ZAI SDK is Node.js-based, so we use the OpenAI-compatible
    REST endpoint that ZAI exposes as the simplest integration path.
    Set ZAI_API_KEY and ZAI_BASE_URL environment variables to configure.
    """
    # The ZAI SDK exposes an OpenAI-compatible chat completions endpoint.
    # We delegate to _call_openai with the ZAI-specific base URL and API key.
    return _call_openai(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        api_key=os.environ.get("ZAI_API_KEY"),
        base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/v1"),
        timeout=timeout,
    )


def _call_hf_api(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> str:
    """Call a model via the HuggingFace Inference API."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=api_key or os.environ.get("HF_TOKEN"))
    response = client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Master model selector
# ---------------------------------------------------------------------------

class HyperparamSelector:
    """Use a master LLM to pick optimal fine-tuning hyperparameters.

    Usage::

        selector = HyperparamSelector(config)
        hyperparams = selector.select(dataset_stats)
    """

    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config

    def select(self, dataset_stats: Any) -> HyperParams:
        """Ask the master model to choose hyperparameters.

        If auto_hyperparams is False, simply returns the defaults / user
        overrides from the config.
        """
        if not self.config.auto_hyperparams:
            console.print("[yellow]Auto-hyperparams disabled — using config values.[/yellow]")
            return self.config.hyperparams

        console.print(Panel(
            f"[cyan]Master model[/cyan]: {self.config.master_model}  |  "
            f"[cyan]Backend[/cyan]: {self.config.master_backend}",
            title="Hyperparameter Selection",
            border_style="cyan",
        ))

        # Build the prompt
        gpu_info, vram_gb = detect_gpu_info()
        user_prompt = MASTER_USER_PROMPT_TEMPLATE.format(
            total_chunks=dataset_stats.total_chunks,
            total_tokens=dataset_stats.total_tokens_estimate,
            avg_chunk_tokens=dataset_stats.avg_chunk_tokens,
            min_chunk_tokens=dataset_stats.min_chunk_tokens,
            max_chunk_tokens=dataset_stats.max_chunk_tokens,
            num_sources=dataset_stats.num_sources,
            data_format=dataset_stats.format,
            model_id=self.config.model_id,
            method=self.config.method,
            gpu_info=gpu_info,
            vram_gb=vram_gb,
        )

        # Call the backend
        raw_response = self._call_backend(user_prompt)

        if not raw_response:
            console.print("[yellow]Master model returned empty response — using defaults.[/yellow]")
            return self.config.hyperparams

        # Parse the JSON
        try:
            # Strip markdown code fences if present
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            hp_dict = json.loads(cleaned)
        except json.JSONDecodeError as e:
            console.print("[yellow]Failed to parse master model response — using defaults.[/yellow]")
            logger.debug("Raw master model response: %s", raw_response[:500])
            return self.config.hyperparams

        # Validate against schema
        try:
            hyperparams = HyperParams(**hp_dict)
        except ValidationError as e:
            console.print("[yellow]Master model returned invalid hyperparams — using valid fields only.[/yellow]")
            # Merge what we can
            valid_fields = {}
            for key, value in hp_dict.items():
                if key in HyperParams.model_fields:
                    valid_fields[key] = value
            hyperparams = HyperParams(**valid_fields)

        # Display the chosen hyperparameters
        self._display_hyperparams(hyperparams)
        return hyperparams

    def _call_backend(self, user_prompt: str) -> str:
        """Dispatch the prompt to the right backend."""
        backend = self.config.master_backend
        model = self.config.master_model

        try:
            if backend == MasterModelBackend.OLLAMA_CLOUD or backend == "ollama_cloud":
                return _call_ollama_cloud(
                    model=model,
                    system_prompt=MASTER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    api_key=self.config.master_api_key,
                    base_url=self.config.master_base_url,
                )
            elif backend == MasterModelBackend.OLLAMA or backend == "ollama":
                return _call_ollama(
                    model=model,
                    system_prompt=MASTER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    host=self.config.ollama_host,
                )
            elif backend == MasterModelBackend.OPENAI or backend == "openai":
                return _call_openai(
                    model=model,
                    system_prompt=MASTER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    api_key=self.config.master_api_key,
                    base_url=self.config.master_base_url,
                )
            elif backend == MasterModelBackend.HF_API or backend == "hf_api":
                return _call_hf_api(
                    model=model,
                    system_prompt=MASTER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    api_key=self.config.master_api_key,
                )
            elif backend == MasterModelBackend.ZAI_SDK or backend == "zai_sdk":
                return _call_zai_sdk(
                    model=model,
                    system_prompt=MASTER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
            else:
                raise ValueError(f"Unknown master backend: {backend}")
        except requests.ConnectionError:
            # If OLLAMA_CLOUD can't connect, try local Ollama as a fallback
            if backend in (MasterModelBackend.OLLAMA_CLOUD, "ollama_cloud"):
                console.print(
                    "[yellow]Ollama Cloud unreachable. "
                    "Trying local Ollama instance...[/yellow]"
                )
                try:
                    return _call_ollama(
                        model=model,
                        system_prompt=MASTER_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        host=self.config.ollama_host,
                    )
                except Exception:
                    console.print("[red]Local Ollama also failed.[/red]")
            console.print("[yellow]Master model unavailable — using default hyperparameters.[/yellow]")
            return ""
        except Exception:
            console.print("[yellow]Master model unavailable — using default hyperparameters.[/yellow]")
            return ""

    @staticmethod
    def _display_hyperparams(hp: HyperParams) -> None:
        """Pretty-print the selected hyperparameters."""
        from rich.table import Table

        table = Table(title="Selected Hyperparameters", show_header=False, border_style="green")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")

        for key, value in hp.model_dump().items():
            table.add_row(key, str(value))

        console.print(table)
