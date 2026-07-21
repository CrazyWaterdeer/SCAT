# Create a Windows desktop shortcut for SCAT (NATIVE Windows / PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_desktop_shortcut.ps1
#
# Puts a "SCAT" icon on your desktop that launches the GUI with no console window
# (via pythonw.exe). Run it after `uv sync` (it needs the .venv). Runs from anywhere
# in the repo. WSL2 users: use scripts/install_desktop_shortcut.sh instead.

$ErrorActionPreference = "Stop"

# Repo root = the parent of this script's folder (scripts\..), so cwd doesn't matter.
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Prefer pythonw.exe (no console window); fall back to python.exe.
$pyw = Join-Path $repo ".venv\Scripts\pythonw.exe"
$py  = Join-Path $repo ".venv\Scripts\python.exe"
if     (Test-Path $pyw) { $target = $pyw }
elseif (Test-Path $py)  { $target = $py }
else {
    Write-Error "No virtual environment found at $repo\.venv. Run 'uv sync' in the repo first."
    exit 1
}

$icon    = Join-Path $repo "scat\resources\icon.ico"
$desktop = [Environment]::GetFolderPath("Desktop")   # honors a OneDrive-redirected Desktop
$lnk     = Join-Path $desktop "SCAT.lnk"

$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnk)
$sc.TargetPath       = $target
$sc.Arguments        = "-m scat.cli gui"
$sc.WorkingDirectory = $repo
if (Test-Path $icon) { $sc.IconLocation = $icon }
$sc.Description       = "SCAT - Spot Classification and Analysis Tool"
$sc.Save()

Write-Host "Created desktop shortcut: $lnk"
Write-Host "  launches: $target -m scat.cli gui  (in $repo)"
