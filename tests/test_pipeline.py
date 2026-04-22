"""
Pipeline-level tests for the Unsloth-only GGUF save path.

The full pipeline needs a GPU, model weights, and a working llama.cpp
toolchain inside Unsloth, so the end-to-end test is skipped whenever any
of those are unavailable — typical local dev machines and CI on CPU.
"""

from __future__ import annotations

import pytest

from tuxtrainer.config import FinetuneConfig


def _gpu_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _unsloth_available() -> bool:
    """Return True only if ``import unsloth`` actually succeeds.

    Unsloth's own device check raises ``NotImplementedError`` on unsupported
    platforms (e.g. macOS with no CUDA/ROCm/XPU), so we have to catch a
    broader set of exceptions than just ``ImportError`` — otherwise test
    collection fails on dev machines.
    """
    try:
        import unsloth  # noqa: F401
    except Exception:
        return False
    return True


gpu_only = pytest.mark.skipif(
    not _gpu_available(),
    reason="No CUDA GPU available — Unsloth's GGUF export requires one.",
)
unsloth_only = pytest.mark.skipif(
    not _unsloth_available(),
    reason="Unsloth not installed.",
)


def _create_test_pdf(text: str, path: Path) -> Path:
    """Write a minimal PDF with the given text — shared with test_pdf_processor."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


@gpu_only
@unsloth_only
def test_pipeline_produces_single_gguf(tmp_path):
    """A completed pipeline run writes exactly one .gguf into config.gguf_output_dir.

    Uses a tiny model and 1 epoch — the assertion is about the artefact,
    not the training quality.
    """
    from tuxtrainer.pipeline import FinetunePipeline

    pdf = _create_test_pdf(
        "Machine learning is a subset of artificial intelligence. "
        "It enables systems to learn from data without explicit programming. ",
        tmp_path / "sample.pdf",
    )

    config = FinetuneConfig(
        model_id="unsloth/Llama-3.2-1B-Instruct",
        pdf_paths=[str(pdf)],
        output_dir=tmp_path / "run",
        auto_hyperparams=False,
        skip_ollama=True,
        use_unsloth=True,
    )
    config.hyperparams.num_train_epochs = 1
    config.hyperparams.per_device_train_batch_size = 1
    config.hyperparams.gradient_accumulation_steps = 1
    config.hyperparams.max_seq_length = 256

    FinetunePipeline(config).run()

    gguf_dir = config.get_gguf_output_dir()
    gguf_files = list(gguf_dir.glob("*.gguf"))

    assert gguf_dir.exists(), f"GGUF output dir missing: {gguf_dir}"
    assert len(gguf_files) == 1, (
        f"Expected exactly one .gguf in {gguf_dir}, found: {gguf_files}"
    )


def test_gguf_resolver_finds_file_in_directory(tmp_path):
    """The Ollama pusher accepts a directory and globs for the .gguf inside."""
    from tuxtrainer.ollama_pusher import _resolve_gguf_file

    gguf = tmp_path / "unsloth.Q4_K_M.gguf"
    gguf.write_bytes(b"fake-gguf")

    assert _resolve_gguf_file(tmp_path) == gguf
    assert _resolve_gguf_file(gguf) == gguf


def test_gguf_resolver_prefers_quantised_over_f16(tmp_path):
    """When both an f16 and a quantised GGUF exist, pick the quantised one."""
    from tuxtrainer.ollama_pusher import _resolve_gguf_file

    (tmp_path / "unsloth.F16.gguf").write_bytes(b"fake-f16")
    quantised = tmp_path / "unsloth.Q4_K_M.gguf"
    quantised.write_bytes(b"fake-q4")

    assert _resolve_gguf_file(tmp_path) == quantised


def test_gguf_resolver_missing_raises(tmp_path):
    from tuxtrainer.ollama_pusher import _resolve_gguf_file

    with pytest.raises(FileNotFoundError):
        _resolve_gguf_file(tmp_path)
