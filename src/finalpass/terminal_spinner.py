from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread
import time
from typing import TextIO

# Adapted from cli-spinners "dots12" (MIT):
# https://github.com/sindresorhus/cli-spinners
DOTS12_INTERVAL_MS = 80
DOTS12_FRAMES: tuple[str, ...] = (
    "⢀⠀",
    "⡀⠀",
    "⠄⠀",
    "⢂⠀",
    "⡂⠀",
    "⠅⠀",
    "⢃⠀",
    "⡃⠀",
    "⠍⠀",
    "⢋⠀",
    "⡋⠀",
    "⠍⠁",
    "⢋⠁",
    "⡋⠁",
    "⠍⠉",
    "⠋⠉",
    "⠋⠉",
    "⠉⠙",
    "⠉⠙",
    "⠉⠩",
    "⠈⢙",
    "⠈⡙",
    "⢈⠩",
    "⡀⢙",
    "⠄⡙",
    "⢂⠩",
    "⡂⢘",
    "⠅⡘",
    "⢃⠨",
    "⡃⢐",
    "⠍⡐",
    "⢋⠠",
    "⡋⢀",
    "⠍⡁",
    "⢋⠁",
    "⡋⠁",
    "⠍⠉",
    "⠋⠉",
    "⠋⠉",
    "⠉⠙",
    "⠉⠙",
    "⠉⠩",
    "⠈⢙",
    "⠈⡙",
    "⠈⠩",
    "⠀⢙",
    "⠀⡙",
    "⠀⠩",
    "⠀⢘",
    "⠀⡘",
    "⠀⠨",
    "⠀⢐",
    "⠀⡐",
    "⠀⠠",
    "⠀⢀",
    "⠀⡀",
)

_CLEAR_LINE = "\r\033[2K"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"


def processing_spinner(message: str, *, stream: TextIO, enabled: bool = True) -> "_TerminalSpinner":
    return _TerminalSpinner(message=message, stream=stream, enabled=enabled)


@dataclass
class _TerminalSpinner:
    message: str
    stream: TextIO
    enabled: bool = True

    def __post_init__(self) -> None:
        self._stop_event = Event()
        self._write_lock = Lock()
        self._thread: Thread | None = None
        self._active = bool(self.enabled and _is_tty(self.stream))

    def __enter__(self) -> "_TerminalSpinner":
        if not self._active:
            return self
        self._write(_HIDE_CURSOR)
        self._thread = Thread(target=self._run, name="finalpass-spinner", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._active:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._write(f"{_CLEAR_LINE}{_SHOW_CURSOR}")

    def _run(self) -> None:
        index = 0
        interval = DOTS12_INTERVAL_MS / 1000.0
        while not self._stop_event.is_set():
            frame = DOTS12_FRAMES[index % len(DOTS12_FRAMES)]
            self._write(f"\r{frame} {self.message}")
            index += 1
            if self._stop_event.wait(interval):
                return

    def _write(self, text: str) -> None:
        with self._write_lock:
            self.stream.write(text)
            self.stream.flush()


def _is_tty(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False
