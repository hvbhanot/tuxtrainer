"""
Fine-tuning module — LoRA / QLoRA fine-tuning of HuggingFace models.

Supports:
  * QLoRA (4-bit quantised base model + LoRA adapters) — default, most VRAM efficient
  * LoRA (full-precision base + LoRA adapters) — for when you have plenty of VRAM
  * Optional Unsloth integration for ~2× faster training

The module loads the base model, applies LoRA adapters, formats the dataset
into the appropriate chat template, and runs SFTTrainer.

The GGUF export lives in :func:`save_gguf_unsloth`, which delegates to
Unsloth's ``save_pretrained_gguf``.  This deliberately bypasses
``transformers.save_pretrained`` on merged 4-bit models — transformers 5.x
introduced a ``ConversionOps`` system whose bitsandbytes dequantize op does
not implement ``reverse_op``, so the standard HF save path raises
``NotImplementedError``.  Unsloth runs its own merge + dequant + llama.cpp
conversion pipeline and sidesteps the broken code path entirely.
"""

from __future__ import annotations

import contextlib
import importlib
import gc
import importlib.machinery
import io
import logging
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Optional

os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

from datasets import Dataset
from rich.console import Console
from rich.panel import Panel

from tuxtrainer.config import FinetuneConfig, FinetuneMethod, HyperParams

console = Console()
logger = logging.getLogger(__name__)

_LLAMA_CPP_TARGETS = (
    "llama-quantize",
    "llama-cli",
    "llama-server",
)
_UNSLOTH_OPTIONAL_IMPORT_NOISE = (
    "Unsloth: Could not import trl.trainer.alignprop_trainer",
    "Unsloth: Could not import trl.trainer.ddpo_trainer",
    "Failed to import trl.models.modeling_sd_base",
    "peft>=0.17.0 is required for a normal functioning of this module",
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _disable_problematic_wandb() -> None:
    """Keep a broken preinstalled ``wandb`` from breaking ``trl`` imports.

    Google Colab images sometimes ship with a ``wandb`` / ``protobuf``
    combination that crashes on plain ``import wandb`` before training even
    starts. tuxtrainer does not use W&B reporting, so we explicitly disable
    it and, if the installed package is broken, shadow it with a tiny stub so
    ``transformers.integrations.is_wandb_available()`` returns ``False``.
    """
    os.environ.pop("WANDB_DISABLED", None)

    try:
        import wandb  # noqa: F401
        return
    except Exception as exc:
        logger.warning("wandb import failed; disabling wandb integration: %s", exc)

    for name in list(sys.modules):
        if name == "wandb" or name.startswith("wandb."):
            sys.modules.pop(name, None)

    stub = types.ModuleType("wandb")
    stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    sys.modules["wandb"] = stub

    integrations = sys.modules.get("transformers.integrations")
    integration_utils = sys.modules.get("transformers.integrations.integration_utils")

    if integrations is not None:
        integrations.is_wandb_available = lambda: False
    if integration_utils is not None:
        integration_utils.is_wandb_available = lambda: False


class _LineFilter(io.TextIOBase):
    """Forward output while dropping known low-signal Unsloth import noise."""

    def __init__(self, target, suppressed_patterns: tuple[str, ...]) -> None:
        self._target = target
        self._suppressed_patterns = suppressed_patterns
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._forward(line + "\n")
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._forward(self._buffer)
            self._buffer = ""
        self._target.flush()

    def _forward(self, text: str) -> None:
        if any(pattern in text for pattern in self._suppressed_patterns):
            return
        self._target.write(text)


def _import_unsloth_module(module_name: str):
    """Import an Unsloth module while suppressing optional dependency chatter."""
    stdout_filter = _LineFilter(sys.stdout, _UNSLOTH_OPTIONAL_IMPORT_NOISE)
    stderr_filter = _LineFilter(sys.stderr, _UNSLOTH_OPTIONAL_IMPORT_NOISE)
    with contextlib.redirect_stdout(stdout_filter), contextlib.redirect_stderr(stderr_filter):
        return importlib.import_module(module_name)


def _sync_gradient_checkpointing(model, enabled: bool) -> None:
    """Keep Unsloth and Transformers gradient-checkpointing state aligned.

    Unsloth's ``for_training`` helper toggles ``module.gradient_checkpointing``
    booleans directly, but recent Transformers Mistral layers also expect
    ``_gradient_checkpointing_func`` to exist whenever the flag is enabled.
    This helper normalises both sides before ``trainer.train()`` starts.
    """
    try:
        from torch.utils.checkpoint import checkpoint
    except ImportError:
        checkpoint = None

    if enabled and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            logger.debug("gradient_checkpointing_enable() failed; using manual sync", exc_info=True)
    elif not enabled and hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            logger.debug("gradient_checkpointing_disable() failed; using manual sync", exc_info=True)

    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = enabled
            if enabled and checkpoint is not None and not hasattr(module, "_gradient_checkpointing_func"):
                module._gradient_checkpointing_func = checkpoint

    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = enabled
        if enabled and checkpoint is not None and not hasattr(model, "_gradient_checkpointing_func"):
            model._gradient_checkpointing_func = checkpoint


def _llama_cpp_install_dir() -> Path:
    """Return the llama.cpp directory that Unsloth should use."""
    override = os.environ.get("UNSLOTH_LLAMA_CPP_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".unsloth" / "llama.cpp"


def _llama_cpp_quantizer_candidates(llama_cpp_dir: Path) -> list[Path]:
    """Return candidate quantizer locations across llama.cpp layouts."""
    search_dirs = [
        llama_cpp_dir,
        llama_cpp_dir / "build" / "bin",
        llama_cpp_dir / "build" / "bin" / "Release",
    ]
    names = [
        "llama-quantize",
        "quantize",
        "llama-quantize.exe",
        "quantize.exe",
    ]
    return [search_dir / name for search_dir in search_dirs for name in names]


def _llama_cpp_converter_candidates(llama_cpp_dir: Path) -> list[Path]:
    """Return candidate converter script locations."""
    return [
        llama_cpp_dir / "convert_hf_to_gguf.py",
        llama_cpp_dir / "convert-hf-to-gguf.py",
    ]


def _detect_llama_cpp_tools(llama_cpp_dir: Path) -> tuple[Path, Path]:
    """Return the working quantizer binary and converter script."""
    if not llama_cpp_dir.exists():
        raise RuntimeError(f"llama.cpp folder '{llama_cpp_dir}' does not exist")

    quantizer = next(
        (
            path
            for path in _llama_cpp_quantizer_candidates(llama_cpp_dir)
            if path.exists() and (os.name == "nt" or os.access(path, os.X_OK))
        ),
        None,
    )
    converter = next(
        (path for path in _llama_cpp_converter_candidates(llama_cpp_dir) if path.exists()),
        None,
    )

    if quantizer is None or converter is None:
        raise RuntimeError(
            f"Unsloth-compatible llama.cpp install is incomplete at {llama_cpp_dir}"
        )

    return quantizer, converter


def _run_llama_cpp_command(command: list[str], cwd: Optional[Path] = None) -> None:
    """Run a llama.cpp bootstrap command and stream output to the console."""
    subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
    )


def _ensure_llama_cpp_checkout(llama_cpp_dir: Path) -> None:
    """Clone llama.cpp if the requested checkout is missing or corrupted."""
    if llama_cpp_dir.exists():
        if (llama_cpp_dir / "CMakeLists.txt").exists():
            return
        if llama_cpp_dir == _llama_cpp_install_dir():
            shutil.rmtree(llama_cpp_dir, ignore_errors=True)
        else:
            raise RuntimeError(
                f"llama.cpp directory exists but is invalid: {llama_cpp_dir}"
            )

    llama_cpp_dir.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Bootstrapping llama.cpp in {llama_cpp_dir}...[/cyan]")
    _run_llama_cpp_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/ggml-org/llama.cpp",
            str(llama_cpp_dir),
        ]
    )


def _build_llama_cpp(llama_cpp_dir: Path) -> None:
    """Build llama.cpp with the modern CMake flow."""
    build_dir = llama_cpp_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)

    console.print("[cyan]Building llama.cpp for Unsloth GGUF export...[/cyan]")
    _run_llama_cpp_command(
        [
            "cmake",
            "-S",
            str(llama_cpp_dir),
            "-B",
            str(build_dir),
            "-DBUILD_SHARED_LIBS=OFF",
            "-DGGML_CUDA=OFF",
            "-Wno-dev",
        ]
    )
    _run_llama_cpp_command(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "-j",
            str(os.cpu_count() or 1),
            "--target",
            *_LLAMA_CPP_TARGETS,
        ]
    )

    bin_dir = build_dir / "bin"
    if bin_dir.exists():
        for binary in bin_dir.glob("llama-*"):
            destination = llama_cpp_dir / binary.name
            shutil.copy2(binary, destination)
            if os.name != "nt":
                destination.chmod(destination.stat().st_mode | 0o111)


def _install_local_llama_cpp(llama_cpp_dir: Path) -> tuple[Path, Path]:
    """Ensure a working llama.cpp install exists for Unsloth."""
    try:
        return _detect_llama_cpp_tools(llama_cpp_dir)
    except RuntimeError:
        pass

    _ensure_llama_cpp_checkout(llama_cpp_dir)
    _build_llama_cpp(llama_cpp_dir)
    return _detect_llama_cpp_tools(llama_cpp_dir)


def _patch_unsloth_llama_cpp_helpers() -> None:
    """Patch older Unsloth builds to use a modern CMake llama.cpp bootstrap."""
    _disable_problematic_wandb()
    unsloth_save = _import_unsloth_module("unsloth.save")

    install_dir = _llama_cpp_install_dir()
    os.environ["UNSLOTH_LLAMA_CPP_PATH"] = str(install_dir)

    def _check_llama_cpp(llama_cpp_folder=None):
        quantizer, converter = _detect_llama_cpp_tools(
            Path(llama_cpp_folder or install_dir).expanduser()
        )
        return str(quantizer), str(converter)

    def _install_llama_cpp(llama_cpp_folder=None, *args, **kwargs):
        quantizer, converter = _install_local_llama_cpp(
            Path(llama_cpp_folder or install_dir).expanduser()
        )
        return str(quantizer), str(converter)

    unsloth_save.check_llama_cpp = _check_llama_cpp
    unsloth_save.install_llama_cpp = _install_llama_cpp
    if hasattr(unsloth_save, "LLAMA_CPP_DEFAULT_DIR"):
        unsloth_save.LLAMA_CPP_DEFAULT_DIR = str(install_dir)


def _hf_token() -> Optional[str]:
    """Return the HuggingFace token from the environment, if any."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _handle_model_load_error(e: Exception, model_id: str) -> None:
    """Raise a user-friendly error for common model loading failures."""
    from huggingface_hub.utils import GatedRepoError

    error_str = str(e).lower()

    # Gated model
    if isinstance(e, GatedRepoError) or "gated" in error_str or "401" in error_str:
        console.print(Panel(
            f"Model '{model_id}' is gated on HuggingFace.\n\n"
            "1. Accept the license: https://huggingface.co/" + model_id + "\n"
            "2. Create a token: https://huggingface.co/settings/tokens\n"
            "3. Set HF_TOKEN and restart the runtime\n\n"
            "Non-gated alternatives:\n"
            "  • unsloth/Llama-3.2-1B-Instruct\n"
            "  • HuggingFaceTB/SmolLM2-1.7B-Instruct",
            title="Gated Model",
            border_style="red",
        ))
        raise RuntimeError(
            f"Model '{model_id}' is gated. Set HF_TOKEN or use a non-gated model."
        ) from e

    # Unsupported architecture
    if "does not recognize this architecture" in error_str or "model_type" in error_str:
        console.print(Panel(
            f"Model '{model_id}' uses an architecture not supported by your transformers version.\n\n"
            "Fixes:\n"
            "  1. Upgrade transformers:\n"
            "     [dim]pip install --upgrade transformers accelerate[/dim]\n"
            "  2. Or use a model with a known architecture:\n"
            "     [dim]unsloth/Llama-3.2-1B-Instruct[/dim]",
            title="Unsupported Model Architecture",
            border_style="red",
        ))
        raise RuntimeError(
            f"Model '{model_id}' architecture not supported. Upgrade transformers or use a different model."
        ) from e


def load_model_and_tokenizer(
    model_id: str,
    max_seq_length: int = 2048,
    method: FinetuneMethod = FinetuneMethod.QLORA,
    use_unsloth: bool = True,
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

    console.print(f"[cyan]Loading model {model_id}...[/cyan]")

    token = _hf_token()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
            token=token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

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
                token=token,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                token=token,
            )
    except Exception as e:
        _handle_model_load_error(e, model_id)
        raise

    model.enable_input_require_grads()

    console.print("[green]  Model loaded[/green]")
    return model, tokenizer


def _load_with_unsloth(
    model_id: str,
    max_seq_length: int = 2048,
    method: FinetuneMethod = FinetuneMethod.QLORA,
):
    """Load model using Unsloth's FastLanguageModel for faster training."""
    _disable_problematic_wandb()
    try:
        FastLanguageModel = _import_unsloth_module("unsloth").FastLanguageModel
    except Exception as exc:
        console.print(
            "[yellow]  Unsloth import failed — falling back to standard HF loading[/yellow]"
        )
        logger.warning("Unsloth import failed during model load: %s", exc)
        return load_model_and_tokenizer(model_id, max_seq_length, method, use_unsloth=False)

    console.print(f"[cyan]Loading model with Unsloth: {model_id}...[/cyan]")

    load_in_4bit = method == FinetuneMethod.QLORA
    token = _hf_token()
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,
            token=token,
        )
    except Exception as e:
        _handle_model_load_error(e, model_id)
        raise

    console.print("[green]  Model loaded[/green]")
    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA adapter application
# ---------------------------------------------------------------------------

def _default_target_modules_for_model(model) -> list[str]:
    """Return the recommended LoRA target modules for the given model architecture.

    Different model families use different names for their linear projections.
    This function inspects the model's module tree and returns the names that
    correspond to linear-like modules (or their wrappers) inside the
    transformer blocks.  Wrapped modules like ``Gemma4ClippableLinear`` are
    detected via their inner linear children.

    Falls back to the classic LLaMA-style defaults if the architecture is
    unrecognised.
    """
    import torch.nn as nn

    _SUPPORTED_LINEARS = [nn.Linear]
    try:
        from bitsandbytes.nn import Linear4bit as _L4, Linear8bit as _L8
        _SUPPORTED_LINEARS.extend([_L4, _L8])
    except ImportError:
        pass
    try:
        from transformers.pytorch_utils import Conv1D as _C1D
        _SUPPORTED_LINEARS.append(_C1D)
    except ImportError:
        pass
    _SUPPORTED_LINEARS = tuple(_SUPPORTED_LINEARS)

    _LAYER_PATTERNS = (".layers.", ".blocks.", ".h.", "layers.", "blocks.", "h.")

    layer_targets: dict[str, int] = {}

    for name, mod in model.named_modules():
        if not any(pat in name for pat in _LAYER_PATTERNS):
            continue

        local = name.split(".")[-1]

        if isinstance(mod, _SUPPORTED_LINEARS):
            layer_targets[local] = layer_targets.get(local, 0) + 1
            continue

        has_linear_child = any(
            isinstance(child, _SUPPORTED_LINEARS)
            for _cn, child in mod.named_children()
        )
        if has_linear_child:
            layer_targets[local] = layer_targets.get(local, 0) + 1

    _GENERIC_NAMES = {"linear", "weight", "bias"}
    if layer_targets and any(n in _GENERIC_NAMES for n in layer_targets):
        specific_names = {n for n in layer_targets if n not in _GENERIC_NAMES}
        if specific_names:
            for gn in _GENERIC_NAMES:
                layer_targets.pop(gn, None)

    _CONTAINER_NAMES = {"self_attn", "self_attention", "attention", "mlp",
                        "feed_forward", "ffn", "block", "layer"}
    for cn in _CONTAINER_NAMES:
        layer_targets.pop(cn, None)

    all_names_in_model = {}
    for name, mod in model.named_modules():
        if any(pat in name for pat in _LAYER_PATTERNS):
            all_names_in_model[name.split(".")[-1]] = mod

    leaf_targets = dict(layer_targets)
    for name in list(leaf_targets):
        mod = all_names_in_model.get(name)
        if mod is not None:
            child_locals = {cn.split(".")[-1] for cn, _ in mod.named_children()}
            if child_locals & set(leaf_targets) and name not in child_locals:
                del leaf_targets[name]

    if layer_targets:
        candidates = sorted(
            name for name, cnt in layer_targets.items() if cnt >= 2
        )
        if candidates:
            return candidates

    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


# ---------------------------------------------------------------------------
# Well-known defaults per model family (used before the model is loaded)
# ---------------------------------------------------------------------------

_MODEL_FAMILY_TARGETS: list[tuple[str, list[str]]] = [
    ("gemma2", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]),
    ("gemma", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]),
    ("phi3", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]),
    ("phi", [
        "q_proj", "k_proj", "v_proj", "dense",
        "fc1", "fc2",
    ]),
    ("qwen2", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]),
    ("llama", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]),
    ("mistral", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]),
    ("falcon", [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]),
    ("gpt_neox", [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]),
    ("gpt2", [
        "c_attn",
        "c_proj",
        "c_fc",
    ]),
    ("pythia", [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]),
    ("mpt", [
        "Wqkv",
        "out_proj",
        "up_proj",
        "down_proj",
    ]),
    ("opt", [
        "q_proj", "k_proj", "v_proj", "out_proj",
        "fc1", "fc2",
    ]),
]


def guess_target_modules_from_model_id(model_id: str) -> list[str]:
    """Guess reasonable default LoRA target modules from a HuggingFace model ID."""
    model_lower = model_id.lower().replace("-", "_")

    for family, targets in _MODEL_FAMILY_TARGETS:
        if family in model_lower:
            return list(targets)

    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


def resolve_target_modules_for_model(model, model_id: str, requested: list[str]) -> list[str]:
    """Auto-detect and resolve the best LoRA target modules for a model."""
    auto_detected = _default_target_modules_for_model(model)

    auto_set = set(auto_detected)
    req_set = set(requested)

    final_requested = requested
    if auto_set != req_set:
        import torch.nn as nn
        _SUPPORTED_LINEARS = [nn.Linear]
        try:
            from bitsandbytes.nn import Linear4bit as _L4, Linear8bit as _L8
            _SUPPORTED_LINEARS.extend([_L4, _L8])
        except ImportError:
            pass
        try:
            from transformers.pytorch_utils import Conv1D as _C1D
            _SUPPORTED_LINEARS.append(_C1D)
        except ImportError:
            pass
        _SUPPORTED_LINEARS = tuple(_SUPPORTED_LINEARS)

        match_count = 0
        for name, mod in model.named_modules():
            local = name.split(".")[-1]
            if local in req_set and isinstance(mod, (nn.Module,) + _SUPPORTED_LINEARS):
                if isinstance(mod, _SUPPORTED_LINEARS):
                    match_count += 1

        if match_count < len(requested) // 2 + 1:
            console.print(
                f"[yellow]⚠ Default target modules {requested} don't fully match "
                f"the {model_id} architecture.[/yellow]"
            )
            console.print(
                f"[cyan]  Auto-detected target modules for this model: "
                f"{auto_detected}[/cyan]"
            )
            final_requested = auto_detected

    return _resolve_target_modules(model, final_requested)


def _resolve_target_modules(model, requested_modules: list[str]) -> list[str]:
    """Resolve target module names against the actual model layers."""
    import torch.nn as nn

    _SUPPORTED_BASES = [nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.Conv3d]

    try:
        from transformers.pytorch_utils import Conv1D as _Conv1D
        _SUPPORTED_BASES.append(_Conv1D)
    except ImportError:
        pass

    try:
        from bitsandbytes.nn import Linear4bit as _Linear4bit, Linear8bit as _Linear8bit
        _SUPPORTED_BASES.extend([_Linear4bit, _Linear8bit])
    except ImportError:
        pass

    _SUPPORTED_BASES = tuple(_SUPPORTED_BASES)

    requested_set = set(requested_modules)
    resolved_names: dict[str, str] = {}

    for full_name, module in model.named_modules():
        parts = full_name.split(".")
        local_name = parts[-1]

        if local_name not in requested_set:
            continue

        if local_name in resolved_names:
            continue

        if isinstance(module, _SUPPORTED_BASES):
            resolved_names[local_name] = local_name
            continue

        found_inner = False
        for inner_name, inner_mod in module.named_modules():
            if inner_name and isinstance(inner_mod, _SUPPORTED_BASES):
                resolved_names[local_name] = f"{local_name}.{inner_name}"
                found_inner = True
                break

        if not found_inner:
            resolved_names[local_name] = local_name

    if not resolved_names:
        return requested_modules

    result = [resolved_names.get(name, name) for name in requested_modules]

    if result != requested_modules:
        console.print(
            f"[dim]  Resolved LoRA target modules: "
            f"{requested_modules} → {result}[/dim]"
        )

    return result


def apply_lora_adapters(
    model,
    hyperparams: HyperParams,
    use_unsloth: bool = False,
    model_id: str = "",
):
    """Apply LoRA adapters to the base model.

    Returns the model with adapters attached.
    """
    console.print(f"[cyan]  Resolving LoRA target modules for {model_id or 'unknown model'}...[/cyan]")
    console.print(f"[dim]  Original target modules: {hyperparams.lora_target_modules}[/dim]")

    target_modules = resolve_target_modules_for_model(
        model, model_id, hyperparams.lora_target_modules,
    )

    console.print(f"[cyan]  Final target modules: {target_modules}[/cyan]")

    if use_unsloth:
        _disable_problematic_wandb()
        try:
            FastLanguageModel = _import_unsloth_module("unsloth").FastLanguageModel
            use_gc = "unsloth" if hyperparams.gradient_checkpointing else False
            model = FastLanguageModel.get_peft_model(
                model,
                r=hyperparams.lora_r,
                lora_alpha=hyperparams.lora_alpha,
                lora_dropout=hyperparams.lora_dropout,
                target_modules=target_modules,
                bias="none",
                use_gradient_checkpointing=use_gc,
                random_state=42,
            )
            _sync_gradient_checkpointing(model, hyperparams.gradient_checkpointing)
            console.print("[green]  LoRA adapters applied[/green]")
            return model
        except Exception as exc:
            logger.warning("Unsloth LoRA path failed; falling back to PEFT: %s", exc)

    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=hyperparams.lora_r,
        lora_alpha=hyperparams.lora_alpha,
        lora_dropout=hyperparams.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    _sync_gradient_checkpointing(model, hyperparams.gradient_checkpointing)
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
    """Format the raw dataset into tokenised training examples."""
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer, "image_processor"):
        tok = tokenizer.tokenizer
    else:
        tok = tokenizer

    def _format_instruction_sample(example: dict) -> dict:
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
        result = tok(
            prompt,
            truncation=True,
            max_length=hyperparams.max_seq_length,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def _format_completion_sample(example: dict) -> dict:
        result = tok(
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

    console.print(f"[green]  Tokenised {len(tokenised_dataset)} examples[/green]")
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
    """Run LoRA / QLoRA fine-tuning and return the adapter checkpoint directory.

    Uses HuggingFace TRL's SFTTrainer under the hood.  The caller retains
    its reference to ``model`` and ``tokenizer`` so that the GGUF export
    step can operate on the in-memory Unsloth-wrapped model without having
    to reload the base.
    """
    _disable_problematic_wandb()
    from trl import SFTConfig, SFTTrainer

    hp = config.hyperparams
    output_dir = Path(config.output_dir) / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    import math
    num_examples = len(tokenised_dataset)
    steps_per_epoch = math.ceil(
        num_examples / (hp.per_device_train_batch_size * hp.gradient_accumulation_steps)
    )
    total_steps = steps_per_epoch * hp.num_train_epochs
    warmup_steps = max(1, int(total_steps * hp.warmup_ratio)) if hp.warmup_ratio > 0 else 0

    sft_args = {}
    if config.data_format == "completion":
        sft_args["dataset_text_field"] = "text"

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=hp.num_train_epochs,
        per_device_train_batch_size=hp.per_device_train_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        learning_rate=hp.learning_rate,
        lr_scheduler_type=hp.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=hp.weight_decay,
        optim=hp.optim,
        fp16=hp.fp16,
        bf16=hp.bf16,
        gradient_checkpointing=hp.gradient_checkpointing,
        logging_steps=hp.logging_steps,
        save_strategy=hp.save_strategy,
        eval_strategy=hp.eval_strategy if "test" in tokenised_dataset.column_names else "no",
        seed=config.seed,
        report_to="none",
        max_grad_norm=1.0,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        max_length=hp.max_seq_length,
        **sft_args,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=tokenised_dataset,
    )

    if hasattr(trainer.model, "for_training"):
        try:
            trainer.model.for_training(use_gradient_checkpointing=hp.gradient_checkpointing)
        except Exception:
            logger.debug("Unsloth for_training() sync failed", exc_info=True)
    _sync_gradient_checkpointing(trainer.model, hp.gradient_checkpointing)

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

    console.print("[cyan]Starting training...[/cyan]")
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    final_dir = Path(config.output_dir) / "final_adapter"
    console.print(f"\n[blue]Saving adapter to {final_dir}...[/blue]")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    console.print(f"[green]  Adapter saved[/green]")
    return final_dir


# ---------------------------------------------------------------------------
# GGUF export (Unsloth-native)
# ---------------------------------------------------------------------------

def _ensure_unsloth_model(model, tokenizer, model_id: str, adapter_path: Optional[Path],
                         max_seq_length: int, method: FinetuneMethod):
    """Return a (model, tokenizer) pair that exposes ``save_pretrained_gguf``.

    If the caller's ``model`` already came out of Unsloth's
    ``FastLanguageModel``, it is returned unchanged.  Otherwise we reload
    the base model via Unsloth and attach the saved LoRA adapter so
    ``save_pretrained_gguf`` is available.
    """
    if model is not None and hasattr(model, "save_pretrained_gguf"):
        return model, tokenizer

    _disable_problematic_wandb()
    try:
        FastLanguageModel = _import_unsloth_module("unsloth").FastLanguageModel
    except Exception as exc:
        raise RuntimeError(
            "Unsloth is required for GGUF export. Install it with:\n"
            "  pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth'"
        ) from exc

    if adapter_path is None:
        raise RuntimeError(
            "Cannot export to GGUF: no in-memory Unsloth model and no adapter "
            "path was provided to reload from."
        )

    console.print("[cyan]Reloading adapter with Unsloth for GGUF export...[/cyan]")
    reloaded_model, reloaded_tok = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=max_seq_length,
        load_in_4bit=method == FinetuneMethod.QLORA,
        dtype=None,
        token=_hf_token(),
    )

    if not hasattr(reloaded_model, "save_pretrained_gguf"):
        raise RuntimeError(
            "Reloaded model still does not expose save_pretrained_gguf. "
            "This usually means the installed Unsloth version is too old — "
            "upgrade with: pip install -U unsloth unsloth_zoo"
        )

    return reloaded_model, reloaded_tok


def save_gguf_unsloth(
    model,
    tokenizer,
    config: FinetuneConfig,
    adapter_path: Optional[Path] = None,
) -> Path:
    """Merge, dequantize, convert, and quantize to GGUF via Unsloth — all in one call.

    This is the *only* save path the pipeline uses.  It deliberately avoids
    ``transformers.save_pretrained`` and ``peft.merge_and_unload`` because
    transformers 5.x's ``ConversionOps.reverse_op`` is not implemented for
    bitsandbytes, which crashes the standard HF save path on merged 4-bit
    models.  Unsloth runs its own merge + dequant + llama.cpp conversion
    internally and bypasses the broken code entirely.

    Args:
        model: The fine-tuned model (ideally still Unsloth-wrapped with the
            LoRA adapter attached and the base in 4-bit).
        tokenizer: Tokenizer that matches the model.
        config: Pipeline configuration (for quantisation and output dir).
        adapter_path: Optional LoRA adapter directory — used as a fallback
            to reload the model via Unsloth when ``model`` is ``None`` or
            does not expose ``save_pretrained_gguf``.

    Returns:
        Path to the single ``.gguf`` file Unsloth wrote.
    """
    gguf_dir = config.get_gguf_output_dir()
    gguf_dir.mkdir(parents=True, exist_ok=True)
    quant = config.get_quantization_method()

    console.print(Panel(
        f"[cyan]Output[/cyan]: {gguf_dir}\n"
        f"[cyan]Quant[/cyan]:  {quant}",
        title="GGUF Export (Unsloth)",
        border_style="cyan",
        padding=(0, 1),
    ))

    model, tokenizer = _ensure_unsloth_model(
        model=model,
        tokenizer=tokenizer,
        model_id=config.model_id,
        adapter_path=adapter_path,
        max_seq_length=config.hyperparams.max_seq_length,
        method=config.method if isinstance(config.method, FinetuneMethod)
            else FinetuneMethod(config.method),
    )

    _patch_unsloth_llama_cpp_helpers()

    console.print("[cyan]Calling Unsloth save_pretrained_gguf...[/cyan]")
    model.save_pretrained_gguf(
        save_directory=str(gguf_dir),
        tokenizer=tokenizer,
        quantization_method=quant,
    )

    gguf_files = sorted(gguf_dir.glob("*.gguf"))
    if not gguf_files:
        raise RuntimeError(
            f"Unsloth's save_pretrained_gguf produced no .gguf file in {gguf_dir}. "
            "Check the logs above for errors from llama.cpp."
        )

    # Prefer a file whose name contains the requested quant tag (case-insensitive).
    quant_tag = quant.replace("-", "_").lower()
    selected = next(
        (p for p in gguf_files if quant_tag in p.name.lower()),
        gguf_files[0],
    )

    console.print(f"[green]  GGUF saved: {selected}[/green]")

    # Free any temporary tensors allocated during conversion.
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return selected
