"""Fail-closed network guard for the client-audio phase.

Once real audio is opened there must be no downloading, update checking,
telemetry, or remote inference. This installs a process-local guard that raises
on any outbound socket connection rather than logging and continuing.
"""
from __future__ import annotations

import os
import socket


class NetworkAccessDenied(RuntimeError):
    """Raised when guarded code attempts an outbound connection."""


class NetworkGuard:
    """Blocks outbound sockets and counts every attempt.

    `attempts` must be 0 for a client-audio run to be reported as local-only.
    Loopback is blocked too: nothing in this evaluation needs it, and allowing
    it would leave a hole a local proxy could use.
    """

    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._original_connect = None
        self._original_connect_ex = None
        self._original_getaddrinfo = None

    def _deny(self, address) -> None:
        self.attempts.append(repr(address))
        raise NetworkAccessDenied(
            f"outbound network access attempted during client-audio evaluation: {address!r}"
        )

    def install(self) -> "NetworkGuard":
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[var] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        guard = self
        self._original_connect = socket.socket.connect
        self._original_connect_ex = socket.socket.connect_ex
        self._original_getaddrinfo = socket.getaddrinfo

        def connect(self, address, *a, **k):        # noqa: ANN001
            guard._deny(address)

        def connect_ex(self, address, *a, **k):     # noqa: ANN001
            guard._deny(address)

        def getaddrinfo(host, port, *a, **k):       # noqa: ANN001
            guard._deny((host, port))

        socket.socket.connect = connect
        socket.socket.connect_ex = connect_ex
        socket.getaddrinfo = getaddrinfo
        return self

    def uninstall(self) -> None:
        if self._original_connect is not None:
            socket.socket.connect = self._original_connect
            socket.socket.connect_ex = self._original_connect_ex
            socket.getaddrinfo = self._original_getaddrinfo

    def __enter__(self) -> "NetworkGuard":
        return self.install()

    def __exit__(self, *exc) -> None:
        self.uninstall()
