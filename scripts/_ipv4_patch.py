"""Force Python networking to use IPv4 only.

On some networks (notably parts of UK academic IPv6), Python's `requests`
library doesn't fall back to IPv4 like curl does; AAAA lookups succeed but
the TCP connect to the v6 address hangs.  This module monkey-patches
``socket.getaddrinfo`` to drop IPv6 results.

Import this BEFORE huggingface_hub / transformers / lerobot / requests.

Enable in any Python entry point with:

    import scripts._ipv4_patch  # noqa: F401

Or, equivalently, set ``MOP_DSRL_IPV4_ONLY=1`` and import
``scripts._ipv4_patch`` — it is a no-op if the env var is "0"/"false".
"""

from __future__ import annotations

import os
import socket

_DISABLED = os.environ.get("MOP_DSRL_IPV4_ONLY", "1").lower() in ("0", "false", "no")


def _install_ipv4_only_patch() -> None:
    if _DISABLED:
        return
    if getattr(socket, "_mop_dsrl_ipv4_patched", False):
        return
    _orig_getaddrinfo = socket.getaddrinfo

    def _getaddrinfo_ipv4_only(*args, **kwargs):
        return [
            info for info in _orig_getaddrinfo(*args, **kwargs)
            if info[0] == socket.AF_INET
        ]

    socket.getaddrinfo = _getaddrinfo_ipv4_only
    socket._mop_dsrl_ipv4_patched = True


_install_ipv4_only_patch()
