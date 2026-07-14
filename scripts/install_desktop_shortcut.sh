#!/usr/bin/env bash
# Create a Windows desktop shortcut that launches the SCAT GUI (with the Assistant) from this
# WSL source install via WSLg — no console window. Re-run any time to refresh it.
#
# Why this instead of a PyInstaller .exe: the Assistant's subscription backend needs the external
# `claude` CLI + login, which can't be bundled into a frozen exe. Running the source install keeps
# the Assistant fully working; this shortcut just makes it a one-click desktop icon.
#
# Requires: WSL2 with Windows interop (powershell.exe on PATH) and WSLg (Windows 11) for the GUI.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISTRO="${WSL_DISTRO_NAME:-Ubuntu}"

command -v powershell.exe >/dev/null || { echo "ERROR: needs WSL with Windows interop (powershell.exe not found)."; exit 1; }
[ -f "$REPO/scat/resources/icon.ico" ] || { echo "ERROR: icon not found at scat/resources/icon.ico"; exit 1; }
[ -x "$REPO/.venv/bin/python" ] || echo "WARNING: $REPO/.venv/bin/python not found — install the venv first (uv venv && pip install -e .[agent])."

# Resolve Windows paths via PowerShell (GetFolderPath handles a OneDrive-redirected Desktop).
DESKTOP_WIN="$(powershell.exe -NoProfile -Command "[Environment]::GetFolderPath('Desktop')" | tr -d '\r')"
APPDIR_WIN="$(powershell.exe -NoProfile -Command '$env:LOCALAPPDATA' | tr -d '\r')\\SCAT"
APPDIR_WSL="$(wslpath "$APPDIR_WIN")"
mkdir -p "$APPDIR_WSL"

# Copy the icon into a Windows-local dir (a reliable .lnk IconLocation).
cp "$REPO/scat/resources/icon.ico" "$APPDIR_WSL/icon.ico"
ICON_WIN="$APPDIR_WIN\\icon.ico"

# Hidden VBS launcher — runs the GUI via WSLg with NO console window (Run style 0).
VBS_WSL="$APPDIR_WSL/scat_launch.vbs"
printf 'CreateObject("WScript.Shell").Run "wsl.exe -d %s --cd %s -- .venv/bin/python -m scat.cli gui", 0, False\n' \
    "$DISTRO" "$REPO" > "$VBS_WSL"
VBS_WIN="$APPDIR_WIN\\scat_launch.vbs"

# Create the desktop .lnk (target = wscript running the hidden VBS), with the SCAT icon.
PS1_WSL="$APPDIR_WSL/_mkshortcut.ps1"
cat > "$PS1_WSL" <<PS
\$s = (New-Object -ComObject WScript.Shell).CreateShortcut("$DESKTOP_WIN\\SCAT.lnk")
\$s.TargetPath = "wscript.exe"
\$s.Arguments = '"' + "$VBS_WIN" + '"'
\$s.IconLocation = "$ICON_WIN"
\$s.WorkingDirectory = "$APPDIR_WIN"
\$s.Description = "SCAT - Spot Classification and Analysis Tool"
\$s.Save()
PS
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$PS1_WSL")"

echo "Created: $DESKTOP_WIN\\SCAT.lnk"
echo "Launches (no console, WSLg GUI): wsl -d $DISTRO --cd $REPO -- .venv/bin/python -m scat.cli gui"
