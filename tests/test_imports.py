"""
Import-level regression tests.
"""

from __future__ import annotations

import importlib
import sys


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
