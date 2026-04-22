"""
Unit tests for the Unsloth integration helpers in ``tuxtrainer.finetuner``.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

from tuxtrainer.config import FinetuneMethod


def test_finetuner_defaults_to_torch_only_runtime():
    """The training path should disable TF/Flax before Unsloth imports."""
    import tuxtrainer.finetuner as finetuner

    assert finetuner.os.environ["USE_TORCH"] == "1"
    assert finetuner.os.environ["USE_TF"] == "0"
    assert finetuner.os.environ["USE_FLAX"] == "0"
    assert finetuner.os.environ["TRANSFORMERS_NO_TF"] == "1"
    assert finetuner.os.environ["TRANSFORMERS_NO_FLAX"] == "1"


def test_ensure_unsloth_model_reloads_adapter_directory(monkeypatch, tmp_path):
    """GGUF fallback reload should point Unsloth at the adapter directory."""
    import tuxtrainer.finetuner as finetuner

    calls = {}

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.update(kwargs)
            model = types.SimpleNamespace(
                save_pretrained_gguf=lambda *args, **kwargs: None,
            )
            tokenizer = object()
            return model, tokenizer

    fake_unsloth = types.ModuleType("unsloth")
    fake_unsloth.__path__ = []
    fake_unsloth.FastLanguageModel = FakeFastLanguageModel

    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    monkeypatch.setattr(finetuner, "_disable_problematic_wandb", lambda: None)

    adapter_dir = tmp_path / "final_adapter"
    adapter_dir.mkdir()

    model, tokenizer = finetuner._ensure_unsloth_model(
        model=None,
        tokenizer=None,
        model_id="base/model",
        adapter_path=adapter_dir,
        max_seq_length=512,
        method=FinetuneMethod.QLORA,
    )

    assert calls["model_name"] == str(adapter_dir)
    assert calls["load_in_4bit"] is True
    assert hasattr(model, "save_pretrained_gguf")
    assert tokenizer is not None


def test_patch_unsloth_llama_cpp_helpers_uses_custom_install_dir(monkeypatch, tmp_path):
    """Older Unsloth builds should be patched to look at our managed llama.cpp dir."""
    import tuxtrainer.finetuner as finetuner

    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir(parents=True)

    quantizer = install_dir / "llama-quantize"
    quantizer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    quantizer.chmod(0o755)

    converter = install_dir / "convert_hf_to_gguf.py"
    converter.write_text("print('ok')\n", encoding="utf-8")

    fake_unsloth = types.ModuleType("unsloth")
    fake_unsloth.__path__ = []
    fake_save = types.ModuleType("unsloth.save")
    fake_save.check_llama_cpp = lambda *args, **kwargs: None
    fake_save.install_llama_cpp = lambda *args, **kwargs: None
    fake_save.LLAMA_CPP_DEFAULT_DIR = "llama.cpp"

    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    monkeypatch.setitem(sys.modules, "unsloth.save", fake_save)
    monkeypatch.setattr(finetuner, "_disable_problematic_wandb", lambda: None)
    monkeypatch.setattr(finetuner, "_llama_cpp_install_dir", lambda: install_dir)

    old_env = os.environ.get("UNSLOTH_LLAMA_CPP_PATH")
    try:
        finetuner._patch_unsloth_llama_cpp_helpers()

        detected_quantizer, detected_converter = fake_save.check_llama_cpp()
        installed_quantizer, installed_converter = fake_save.install_llama_cpp()

        assert Path(detected_quantizer) == quantizer
        assert Path(detected_converter) == converter
        assert Path(installed_quantizer) == quantizer
        assert Path(installed_converter) == converter
        assert os.environ["UNSLOTH_LLAMA_CPP_PATH"] == str(install_dir)
        assert fake_save.LLAMA_CPP_DEFAULT_DIR == str(install_dir)
    finally:
        if old_env is None:
            os.environ.pop("UNSLOTH_LLAMA_CPP_PATH", None)
        else:
            os.environ["UNSLOTH_LLAMA_CPP_PATH"] = old_env
