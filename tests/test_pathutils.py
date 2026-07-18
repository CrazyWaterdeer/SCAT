"""Windows <-> WSL path normalization so any path a user gives works from either environment."""
from scat.pathutils import normalize_path

WSL = dict(is_wsl=True, is_windows=False)
WIN = dict(is_wsl=False, is_windows=True)
LINUX = dict(is_wsl=False, is_windows=False)


def test_windows_drive_to_wsl_mount():
    assert normalize_path(r"D:\Jin\images", **WSL) == "/mnt/d/Jin/images"
    assert normalize_path("D:/Jin/images", **WSL) == "/mnt/d/Jin/images"
    assert normalize_path(r"C:\Users\jin\a.tif", **WSL) == "/mnt/c/Users/jin/a.tif"
    assert normalize_path(r"D:\Jin/mixed\slash", **WSL) == "/mnt/d/Jin/mixed/slash"
    assert normalize_path("D:\\", **WSL) == "/mnt/d"          # drive root


def test_posix_and_wsl_paths_pass_through_on_wsl():
    assert normalize_path("/mnt/d/Jin/images", **WSL) == "/mnt/d/Jin/images"
    assert normalize_path("/home/lab/x", **WSL) == "/home/lab/x"
    assert normalize_path("relative/dir", **WSL) == "relative/dir"


def test_wsl_mount_to_windows_on_windows():
    assert normalize_path("/mnt/d/Jin/images", **WIN) == r"D:\Jin\images"
    assert normalize_path("/mnt/c/Users/jin", **WIN) == r"C:\Users\jin"
    assert normalize_path(r"D:\already\win", **WIN) == r"D:\already\win"   # unchanged


def test_plain_linux_leaves_everything_alone():
    # not WSL, not Windows: a D:\ path is genuinely foreign — don't invent a /mnt that isn't there
    assert normalize_path(r"D:\Jin\images", **LINUX) == r"D:\Jin\images"
    assert normalize_path("/home/lab/x", **LINUX) == "/home/lab/x"


def test_strips_wrapping_quotes_and_whitespace():
    assert normalize_path('  "D:\\Jin\\imgs"  ', **WSL) == "/mnt/d/Jin/imgs"


def test_idempotent_and_edge_inputs():
    once = normalize_path(r"D:\Jin\images", **WSL)
    assert normalize_path(once, **WSL) == once                # idempotent
    assert normalize_path("") == ""
    assert normalize_path(None) is None


# --- open_in_os: cross-platform, WSL-aware file/folder opening ---
def test_open_in_os_wsl_hands_off_to_windows_explorer(monkeypatch):
    import subprocess
    import scat.pathutils as P
    monkeypatch.setattr(P, "_is_wsl", lambda: True)
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": "C:\\x\\report.html"})())
    launched = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **k: launched.append(cmd))
    assert P.open_in_os("/mnt/c/x/report.html") is True
    assert launched[0] == ["explorer.exe", "C:\\x\\report.html"]   # not xdg-open


def test_open_in_os_plain_linux_uses_xdg_open(monkeypatch):
    import subprocess
    import scat.pathutils as P
    monkeypatch.setattr(P, "_is_wsl", lambda: False)
    launched = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **k: launched.append(cmd))
    assert P.open_in_os("/home/x/out") is True
    assert launched[0][0] == "xdg-open"


def test_open_in_os_returns_false_when_no_opener(monkeypatch):
    import subprocess
    import scat.pathutils as P
    monkeypatch.setattr(P, "_is_wsl", lambda: False)
    def _boom(cmd, **k):
        raise FileNotFoundError()
    monkeypatch.setattr(subprocess, "Popen", _boom)
    assert P.open_in_os("/x") is False   # never raises; reports failure so the caller can show the path
