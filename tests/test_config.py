"""
Tests for the configuration module.
"""

import json
import warnings
from pathlib import Path

import pytest

from tuxtrainer.config import (
    FinetuneConfig,
    FinetuneMethod,
    HyperParams,
    Quantisation,
    SUPPORTED_QUANTIZATIONS,
)


class TestHyperParams:
    """Tests for the HyperParams model."""

    def test_defaults(self):
        hp = HyperParams()
        assert hp.lora_r == 16
        assert hp.lora_alpha == 32
        assert hp.lora_dropout == 0.05
        assert hp.num_train_epochs == 3
        assert hp.learning_rate == 2e-4
        assert hp.bf16 is True
        assert hp.gradient_checkpointing is True

    def test_custom_values(self):
        hp = HyperParams(lora_r=64, lora_alpha=128, learning_rate=1e-4)
        assert hp.lora_r == 64
        assert hp.lora_alpha == 128
        assert hp.learning_rate == 1e-4

    def test_serialization(self):
        hp = HyperParams()
        json_str = hp.model_dump_json()
        data = json.loads(json_str)
        assert data["lora_r"] == 16

    def test_roundtrip(self):
        hp = HyperParams(lora_r=32, learning_rate=3e-4)
        json_str = hp.model_dump_json()
        hp2 = HyperParams.model_validate_json(json_str)
        assert hp2.lora_r == 32
        assert hp2.learning_rate == 3e-4


class TestFinetuneConfig:
    """Tests for the FinetuneConfig model."""

    def test_minimal_config(self):
        config = FinetuneConfig(model_id="unsloth/Llama-3.2-1B-Instruct")
        assert config.model_id == "unsloth/Llama-3.2-1B-Instruct"
        assert config.method == FinetuneMethod.QLORA
        assert config.auto_hyperparams is True

    def test_ollama_model_name_default(self):
        config = FinetuneConfig(model_id="unsloth/Llama-3.2-1B-Instruct")
        name = config.get_ollama_model_name()
        assert name == "llama-3.2-1b-instruct-finetuned"

    def test_ollama_model_name_custom(self):
        config = FinetuneConfig(
            model_id="unsloth/Llama-3.2-1B-Instruct",
            ollama_model_name="my-custom-model",
        )
        assert config.get_ollama_model_name() == "my-custom-model"

    def test_pdf_path_resolution(self, tmp_path):
        pdf1 = tmp_path / "doc1.pdf"
        pdf1.write_text("fake")
        pdf2 = tmp_path / "doc2.pdf"
        pdf2.write_text("fake")

        config = FinetuneConfig(
            model_id="test/model",
            pdf_paths=[str(pdf1), str(pdf2)],
        )
        all_paths = config.get_all_pdf_paths()
        assert len(all_paths) == 2

    def test_pdf_dir_expansion(self, tmp_path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "a.pdf").write_text("fake")
        (pdf_dir / "b.pdf").write_text("fake")
        (pdf_dir / "c.pdf").write_text("fake")
        (pdf_dir / "readme.txt").write_text("ignore me")

        config = FinetuneConfig(
            model_id="test/model",
            pdf_dirs=[str(pdf_dir)],
        )
        all_paths = config.get_all_pdf_paths()
        assert len(all_paths) == 3
        assert all(p.suffix == ".pdf" for p in all_paths)


class TestGGUFOutputDir:
    """The pipeline produces a single .gguf file — no merged fp16 artefact."""

    def test_default_gguf_output_dir(self, tmp_path):
        config = FinetuneConfig(
            model_id="test/model",
            output_dir=tmp_path / "run",
        )
        assert config.get_gguf_output_dir() == tmp_path / "run" / "gguf"

    def test_custom_gguf_output_dir(self, tmp_path):
        config = FinetuneConfig(
            model_id="test/model",
            output_dir=tmp_path / "run",
            gguf_output_dir=tmp_path / "custom_gguf",
        )
        assert config.get_gguf_output_dir() == tmp_path / "custom_gguf"


class TestQuantisation:
    """Quantisation values must match what Unsloth's save_pretrained_gguf accepts."""

    def test_enum_values_are_unsloth_compatible(self):
        expected = {"q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"}
        assert {q.value for q in Quantisation} == expected
        assert SUPPORTED_QUANTIZATIONS == frozenset(expected)

    def test_legacy_uppercase_accepted(self):
        """Old CLI/YAML configs with 'Q4_K_M' should still parse."""
        config = FinetuneConfig(model_id="test/model", quantisation="Q4_K_M")
        assert config.get_quantization_method() == "q4_k_m"

    def test_method_string_is_lowercase(self):
        config = FinetuneConfig(model_id="test/model", quantisation=Quantisation.Q8_0)
        assert config.get_quantization_method() == "q8_0"


class TestDeprecatedKwargs:
    """Old merged-fp16 directory kwargs warn and are dropped."""

    @pytest.mark.parametrize(
        "legacy_kwarg",
        ["merged_model_dir", "merged_output_dir", "merged_dir"],
    )
    def test_legacy_merged_kwargs_warn(self, legacy_kwarg, tmp_path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = FinetuneConfig(
                model_id="test/model",
                **{legacy_kwarg: str(tmp_path / "merged")},
            )
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        assert not hasattr(config, legacy_kwarg)
