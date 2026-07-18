"""Make user-supplied paths work regardless of whether SCAT runs on Windows or WSL, and whether the
path was given in Windows form (``D:\\imgs``, ``D:/imgs``) or POSIX/WSL form (``/mnt/d/imgs``,
``/home/…``).

Under WSL a Windows drive path is rewritten to its ``/mnt`` mount; on native Windows a ``/mnt`` mount
is rewritten back to a drive path. Anything already native to the current environment (and anything we
don't recognise) passes through unchanged, so ``normalize_path`` is safe to call on every incoming
path and is idempotent.
"""
from __future__ import annotations

import os
import platform
import re

_WIN_DRIVE = re.compile(r"^([A-Za-z]):[\\/](.*)$")     # C:\x  or  C:/x
_WSL_MOUNT = re.compile(r"^/mnt/([A-Za-z])/(.*)$")      # /mnt/c/x


def _is_wsl() -> bool:
    return "microsoft" in platform.uname().release.lower()


def normalize_path(path, *, is_wsl: bool | None = None, is_windows: bool | None = None):
    """Return ``path`` translated to the form the current OS can open.

    - On **WSL**, ``D:\\Jin\\imgs`` / ``D:/Jin/imgs`` -> ``/mnt/d/Jin/imgs``.
    - On **native Windows**, ``/mnt/d/Jin/imgs`` -> ``D:\\Jin\\imgs``.
    - Already-native paths, relative paths, and unrecognised forms are returned unchanged.

    ``is_wsl`` / ``is_windows`` override the environment probe (used by tests). Falsy input is
    returned as-is.
    """
    if not path:
        return path
    s = str(path).strip().strip('"').strip("'")
    if not s:
        return s
    if is_wsl is None:
        is_wsl = _is_wsl()
    if is_windows is None:
        is_windows = os.name == "nt"

    m = _WIN_DRIVE.match(s)
    if m and is_wsl:
        drive, rest = m.group(1).lower(), m.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"

    if is_windows:
        mm = _WSL_MOUNT.match(s)
        if mm:
            drive, rest = mm.group(1).upper(), mm.group(2).replace("/", "\\")
            return f"{drive}:\\{rest}" if rest else f"{drive}:\\"

    return s
