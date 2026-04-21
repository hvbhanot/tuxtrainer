"""
Tests for the configuration module.
"""

import json
import tempfile
from pathlib import Path

from tuxtrainer.config import FinetuneConfig, HyperParams, FinetuneMethod, Quantisation


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
        assert name == "llama-3.1-8b-finetuned"

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
        # Non-PDF file should be ignored
        (pdf_dir / "readme.txt").write_text("ignore me")

        config = FinetuneConfig(
            model_id="test/model",
            pdf_dirs=[str(pdf_dir)],
        )
        all_paths = config.get_all_pdf_paths()
        assert len(all_paths) == 3
        assert all(p.suffix == ".pdf" for p in all_paths)
