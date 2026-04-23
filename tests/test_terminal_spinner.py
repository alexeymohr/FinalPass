from __future__ import annotations

import io
import time

from finalpass import terminal_spinner
from finalpass.terminal_spinner import processing_spinner


class _FakeStream(io.StringIO):
    def __init__(self, *, isatty: bool) -> None:
        super().__init__()
        self._isatty = isatty

    def isatty(self) -> bool:
        return self._isatty


def test_processing_spinner_is_silent_on_non_tty() -> None:
    stream = _FakeStream(isatty=False)
    with processing_spinner("Running loudness...", stream=stream):
        pass
    assert stream.getvalue() == ""


def test_processing_spinner_emits_frames_on_tty(monkeypatch) -> None:
    stream = _FakeStream(isatty=True)
    monkeypatch.setattr(terminal_spinner, "DOTS12_FRAMES", ("a", "b"))
    monkeypatch.setattr(terminal_spinner, "DOTS12_INTERVAL_MS", 1)

    with processing_spinner("Running null check...", stream=stream):
        time.sleep(0.01)

    output = stream.getvalue()
    assert "Running null check..." in output
    assert "a" in output or "b" in output
    assert "\033[2K" in output
