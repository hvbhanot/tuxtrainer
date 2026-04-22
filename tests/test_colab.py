"""
Tests for the lightweight Colab bootstrap helper.
"""

from __future__ import annotations

import subprocess


def test_setup_colab_pins_protobuf_below_6(monkeypatch):
    """Colab bootstrap must not allow protobuf 6.x onto the runtime."""
    import tuxtrainer.colab as colab

    commands: list[str] = []

    def fake_run(cmd: str, timeout: int = 180, check: bool = True):
        commands.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(colab, "_run", fake_run)
    monkeypatch.setattr(colab, "_is_colab", lambda: True)

    colab.setup_colab(install_ollama=False)

    assert any('"protobuf>=3.20,<6"' in command for command in commands)
