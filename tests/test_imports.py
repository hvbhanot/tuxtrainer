"""
Import-level regression tests.
"""

from __future__ import annotations

import importlib
import sys
import types


def _clear_tuxtrainer_modules() -> None:
    """Remove tuxtrainer modules from sys.modules for a fresh import."""
    for name in list(sys.modules):
        if name == "tuxtrainer" or name.startswith("tuxtrainer."):
            sys.modules.pop(name, None)


def test_importing_colab_stays_lightweight():
    """Importing ``tuxtrainer.colab`` must not eagerly import the PDF stack."""
    _clear_tuxtrainer_modules()

    colab = importlib.import_module("tuxtrainer.colab")

    assert hasattr(colab, "setup_colab")
    assert "tuxtrainer.pipeline" not in sys.modules
    assert "tuxtrainer.pdf_processor" not in sys.modules


def test_top_level_exports_are_lazy():
    """Top-level package exports should resolve on demand."""
    _clear_tuxtrainer_modules()

    pkg = importlib.import_module("tuxtrainer")

    assert "tuxtrainer.config" not in sys.modules
    assert "tuxtrainer.pipeline" not in sys.modules

    config_cls = pkg.FinetuneConfig

    assert config_cls.__name__ == "FinetuneConfig"
    assert "tuxtrainer.config" in sys.modules
    assert "tuxtrainer.pipeline" not in sys.modules


def test_broken_wandb_is_stubbed(monkeypatch):
    """A broken ``wandb`` install must not break training imports."""
    _clear_tuxtrainer_modules()

    import tuxtrainer.finetuner as finetuner

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "wandb":
            raise ImportError("broken wandb")
        return real_import(name, globals, locals, fromlist, level)

    for name in list(sys.modules):
        if name == "wandb" or name.startswith("wandb."):
            sys.modules.pop(name, None)

    integrations = types.ModuleType("transformers.integrations")
    integrations.is_wandb_available = lambda: True
    integration_utils = types.ModuleType("transformers.integrations.integration_utils")
    integration_utils.is_wandb_available = lambda: True

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.setitem(sys.modules, "transformers.integrations", integrations)
    monkeypatch.setitem(
        sys.modules,
        "transformers.integrations.integration_utils",
        integration_utils,
    )

    finetuner._disable_problematic_wandb()

    assert "wandb" in sys.modules
    assert integrations.is_wandb_available() is False
    assert integration_utils.is_wandb_available() is False
