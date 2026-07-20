# SCAT — Spot Classification and Analysis Tool

**Version 2.2.2**

A machine-learning tool for *Drosophila* excreta analysis. SCAT detects deposits in scanned
images, classifies each as **Normal**, **ROD** (Reproductive Oblong Deposit), or **Artifact**,
and turns a folder of images into per-image measurements, group statistics, and a shareable
HTML report — from the GUI, the command line, or a plain-language chat with a built-in Claude
assistant.

📖 **[Detailed Workflow Guide](WORKFLOW.md)**

> **New in 2.1.0** — **per-fly normalization**: because a vial's total deposit count and total IOD
> scale with how many flies are in it, SCAT now reports and compares deposit count and IOD **per fly**
> whenever a per-image fly count is available (read from the filenames, e.g. "… 3 flies …", or supplied
> as `n_flies`); fractions and per-deposit means are unaffected, and runs without fly counts fall back
> to per-image totals with a warning. Also: the assistant's replies are no longer cut off at a low
> output-token limit (raised + auto-migrated).
>
> **New in 2.0.1** — the assistant can now re-render an existing results folder a different way
> **without re-detecting** (change the primary metric, group order, or plot colours, or produce the
> report *after* you have manually reviewed the detections), and it can **train or update the
> classifier** from your reviewed results. Plus fixes: a plotting error no longer aborts an analysis,
> and `scat train` now reads labels written on Windows when run under WSL.
>
> **What's new in 2.0** — a codebase-wide hardening pass: sounder statistics (Welch's
> *t*-test, a corrected Cohen's *d*, a consistent *n* ≥ 3 significance gate), a group-comparison
> "Deposit Count" that now excludes artifacts everywhere (matching the rest of the report),
> grayscale/RGBA images no longer crash detection, HTML-escaped filenames/group names in the
> report, and CLI `--version` / `--spatial` flags. Plus many smaller bug fixes and cleanups.

## Features

- **Automatic detection** — rule-based adaptive-threshold segmentation (a standard and a
  "sensitive" mode for dilute deposits), or optional U-Net deep segmentation.
- **Classification** — pick per run: a simple circularity **Threshold** rule, a **Random Forest**
  over shape+colour features (the shipped model), or a **CNN** (ResNet-18). Labels: Normal / ROD /
  Artifact.
- **Statistics** — image-level tests auto-selected by normality: t-test / paired t-test /
  Mann-Whitney U / Wilcoxon for two groups, one-way ANOVA / Kruskal-Wallis with Holm-corrected
  post-hoc pairs for 3+, plus Cohen's *d*, 95% CIs, and domain analyses (pH·hue, pigment/IOD,
  size, density, spatial dispersion, correlations).
- **Visualisations** — summary dashboard, PCA, feature z-score heatmap, scatter matrix, area–IOD
  scatter, and per-metric violin / mean-CI plots.
- **Interactive review** — the results surface flags per-image low-confidence deposits to
  re-check; double-click a row to correct classifications and regenerate.
- **Reporting** — a single self-contained `report.html` with charts embedded inline; CSV data
  exports; optional PDF (via the `pdf` extra).
- **Conversational assistant** — an optional Claude agent that runs the whole pipeline from a
  request like *"analyze this folder and compare the groups."*

## Getting started

First-time setup, top to bottom — **copy-paste each block in order.** SCAT uses
**[uv](https://docs.astral.sh/uv/)**, and ships a pre-trained Random Forest model
(`models/model_rf.pkl`), so you can analyze images right after step 3.

**1 — Install uv** (Python/venv manager; once per machine):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell (or `source $HOME/.local/bin/env`) so `uv` is on your PATH.

**2 — Get SCAT and install it** (creates a `.venv` and installs core + GUI + the assistant):

```bash
git clone https://github.com/CrazyWaterdeer/SCAT.git
cd SCAT
uv sync --extra agent
```

`uv sync` builds a reproducible environment from the committed `uv.lock`. `--extra agent`
installs the assistant's Python SDK (`claude-agent-sdk`) — **required for the chat even if the
`claude` CLI is already installed.** Omit it (`uv sync`) if you don't want the assistant; add
`--extra deep` for the U-Net/CNN (torch) or `--extra pdf` for PDF export.

**3 — Launch the GUI:**

```bash
uv run scat gui
```

That's all you need to analyze images with the bundled model. Steps 4–5 are optional.

**4 — (Optional) Connect the AI Assistant.** Choose one backend:

- **Subscription — no API charges** (uses your Claude Pro/Max login). **If you run SCAT under WSL2,
  install and log in to the `claude` CLI _inside_ the WSL distro** — a login done in native Windows is
  not visible to SCAT running under WSL. (Native-Windows users: see [Native Windows](#native-windows-powershell--advanced).)

  ```bash
  curl -fsSL https://claude.ai/install.sh | bash   # installs the `claude` CLI to ~/.local/bin
  claude                                            # opens a browser to log in, then quit with Ctrl+C
  ```

  Confirm SCAT sees the login:

  ```bash
  uv run python -c "from scat.agent.claude_subscription import subscription_available; print(subscription_available())"
  # (True, None) = connected.  (False, 'SDK missing') = re-run step 2 with --extra agent.
  # (False, 'claude not found' / 'not logged in') = install/log in the claude CLI *inside WSL*.
  ```

- **API — billed.** Skip the CLI; paste a key into **Settings › Assistant** in the GUI, or export
  `ANTHROPIC_API_KEY=sk-ant-…` before launching.

**5 — (Optional) One-click desktop icon** (**WSL2 only** — Windows 11 + WSLg):

```bash
bash scripts/install_desktop_shortcut.sh
```

Puts an **SCAT** icon on your Windows desktop (launches `.venv/bin/python -m scat.cli gui`, no
console window). This script is a WSL→Windows bridge (it uses `wslpath`/`powershell.exe`) and
**requires** WSL2 interop — it does not run on native Windows. Run it after step 2. On native
Windows, just use `uv run scat gui` (or make a shortcut to `.venv\Scripts\pythonw.exe -m scat.cli gui`).

### Native Windows (PowerShell) — advanced

The steps above are written for **Linux or WSL2** (bash). **WSL2 is the recommended way to run SCAT
on Windows** — the commands above work verbatim there, and PDF export, the ~9× parallel speed-up, and
the subscription assistant all work out of the box. SCAT also runs on **native Windows** (the code is
cross-platform), but a few setup commands differ. The core happy path
(`git clone` → `uv sync --extra agent` → `uv run scat gui`) and every `uv run scat <cmd>` are
**identical on both** — only these differ:

| Step / task | Linux / WSL2 | Native Windows (PowerShell) |
| ----------- | ------------ | --------------------------- |
| Install uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| PATH after install | `source $HOME/.local/bin/env` | not needed — restart the shell (the installer updates PATH) |
| Install `claude` CLI | `curl -fsSL https://claude.ai/install.sh \| bash` | `irm https://claude.ai/install.ps1 \| iex` (needs a Windows-capable Claude Code build — confirm with `claude --version`) |
| API key (env var) | `export ANTHROPIC_API_KEY=sk-ant-…` | `$env:ANTHROPIC_API_KEY = "sk-ant-…"` (or paste it into Settings › Assistant) |
| Activate the venv | `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` |
| Run via the interpreter | `.venv/bin/python -m scat.cli <cmd>` | `.venv\Scripts\python.exe -m scat.cli <cmd>` |
| Desktop icon (step 5) | `bash scripts/install_desktop_shortcut.sh` | not available — use `uv run scat gui` |
| Path arguments | `/path/to/images` | `C:\path\to\images` |

**Tip:** prefer `uv run scat <cmd>` — it is the same on every OS and sidesteps the venv-activation and
interpreter-path differences entirely.

Two native-Windows caveats (both are non-issues under WSL2/Linux):

- **PDF export** (the `pdf` extra / WeasyPrint) installs fine but needs the **GTK3 runtime** installed
  separately and on `PATH`, or it errors at render time (`cannot load library 'libgobject-2.0-0'`). If
  you need PDF reports on Windows, run SCAT under WSL2. (The default `report.html` needs none of this.)
- **Speed:** native Windows runs analysis single-process (GIL-bound) — the ~9× fork parallelism is
  Linux/WSL2-only. Large batches are noticeably faster under WSL2.
- **Python version:** on native Windows stay on **Python 3.10–3.13**; PySide6/torch wheels can lag a new
  CPython (e.g. 3.14) for a while.

### Extras & ways to run

| Extra    | Adds                                   | Enables                                   |
| -------- | -------------------------------------- | ----------------------------------------- |
| `agent`  | pydantic, anthropic, claude-agent-sdk  | the conversational Assistant (Claude)     |
| `deep`   | torch, torchvision                     | U-Net segmentation and the CNN classifier |
| `pdf`    | weasyprint                             | PDF export of the report (native Windows needs the GTK3 runtime — see above) |
| `dev`    | pytest                                 | running the test suite                    |

Compose extras (`uv sync --extra agent --extra deep`). Once installed, `uv run scat <cmd>` is the
portable way to run (identical on every OS). You can also use an activated venv
(`source .venv/bin/activate`, or `.venv\Scripts\Activate.ps1` on native Windows) then `scat <cmd>`, or
call the interpreter directly (`.venv/bin/python` / `.venv\Scripts\python.exe` `-m scat.cli <cmd>`).

## Quick Start (GUI)

1. **Launch** — `uv run scat gui` (or the desktop icon). One window opens on the **Configure**
   screen; Labeling, Train model, and Settings live behind the top-bar **More** menu.
2. **Add images** — drag image files or a whole folder onto the drop zone (folders are searched
   recursively), or use **Choose images… / Choose folder…**.
3. **Options** (all optional) — pick the classification **Method** (Threshold / Random Forest /
   CNN; default Random Forest) and a **Classifier model** file (blank = the bundled
   `models/model_rf.pkl`). Toggle **Compare groups** to enable **Group by subfolder** or **Load
   grouping CSV…**. Output toggles: Annotated images · Visualizations · HTML report (all on).
4. **Run Analysis** — the sticky footer button. Progress streams per image; when it finishes the
   window switches in place to the **results surface** (the report is generated automatically if
   *HTML report* was on).
5. **Review** — the per-image table's **Review** column shows, per image, how many low-confidence
   deposits to double-check. **Double-click a row** to open the editor, fix classifications, and
   save; the hero card's **Open report / Rebuild after edits** and **Open folder** actions live at
   the top. Use **New analysis** / **Re-run** to start over.

> 💡 The pre-trained model works well for standard Bromophenol-Blue-stained deposits. If results
> are unsatisfactory, see [Building Custom Models](WORKFLOW.md#building-custom-models).

## Command line

Every step is scriptable — `scat <command> --help` for details:

| Command          | What it does                                                              |
| ---------------- | ------------------------------------------------------------------------ |
| `scat gui`       | Launch the desktop GUI.                                                   |
| `scat chat`      | Conversational agent in the terminal (see below).                        |
| `scat analyze`   | Headless folder analysis → CSVs + report.                                |
| `scat train`     | Train an `rf`/`cnn` classifier from labelled data.                       |
| `scat label`     | Launch the labeling GUI.                                                  |
| `scat cluster`   | Unsupervised-cluster deposits to speed up labeling.                      |
| `scat propagate` | Propagate cluster labels to deposits.                                    |

Headless run, grouped, with stats + report:

```bash
uv run scat analyze /path/to/images \
    -m groups.csv --group-by group \
    --stats --report --annotate --visualize
```

Note: `--annotate` and `--visualize` are **off** by default on the CLI (the GUI defaults them on),
and `--threshold` sets the detector's **circularity** parameter, not a classifier probability.

## AI Assistant (conversational agent)

With the `agent` extra, SCAT includes a Claude-powered agent that runs the pipeline from plain
language ("analyze this folder and compare the groups"). It infers grouping from your
filenames/subfolders, runs detection → classification → statistics → report, and streams
per-image progress. It can also **re-graph or re-report an existing results folder without
re-detecting** (change the primary metric, group order, or plot colours), **wait while you manually
review the detections** and then produce the outputs, and **train or update the classifier** from
your reviewed results.

```bash
uv run scat chat        # in the terminal
uv run scat gui         # in the GUI — click Assistant in the top bar to open the dock
```

**Providers** (dock's Provider picker, or `scat chat --backend {auto,subscription,api}`,
default `auto`):

- **Subscription** — no API charges; uses your local Claude Code login. Install the
  [`claude` CLI](https://claude.com/claude-code) and log in. If no login is detected the picker
  shows **"Subscription — not connected."**
- **API** — billed. Provide a key in **Settings › Assistant** (password-masked, saved to your
  config) *or* via the `ANTHROPIC_API_KEY` environment variable (the env var wins if both are set).
- **Auto** (default) — uses the subscription when logged in, otherwise **falls back to the billed
  API** and warns in the status line so a transient login failure never bills you by surprise.

Pick the model too (Opus 4.8 · Fable 5 · Sonnet 5 · Haiku 4.5); the GUI remembers your last model
and provider across restarts. Without the `agent` extra (or any backend) the GUI still runs — the
dock just shows an install/login hint.

Settings — including the API key, model, and provider — live in `~/.scat/config.json`, written
with owner-only (`0600`) permissions because it may hold the key. The key is stored in plain text.

## Output Files

A run writes a timestamped folder **next to** the analyzed image folder (override with
`-o/--output`):

```
results_YYYYMMDD_HHMMSS/
├── image_summary.csv       # per-image statistics                     (always)
├── all_deposits.csv        # every deposit, one row each              (always)
├── run_manifest.json       # reproducibility sidecar (see below)      (always)
├── deposits/               # per image: <stem>_deposits.csv + <stem>.labels.json   (always)
├── report.html             # self-contained report, charts inline     (default on)
├── condition_summary.csv   # group-level statistics                   (grouped runs)
├── groups/                 # <group>_deposits.csv per group           (grouped runs)
├── annotated/              # <stem>_annotated.png                     (if annotate on)
├── visualizations/         # standalone stat PNGs                     (if visualize on)
└── spatial_stats.json      # spatial dispersion stats                 (if spatial on)
```

`report.html` embeds its charts as inline images — it does not depend on the `visualizations/`
folder. `run_manifest.json` records the SCAT version, git commit, Python/platform, a dataset
fingerprint, the model + its hash, detection settings, and the analysis contract, so any run can
be reproduced and audited.

## Groups CSV format

To compare experimental conditions, map filenames to groups:

```csv
filename,group
image001.tif,Control
image002.tif,Control
image003.tif,Treatment
image004.tif,Treatment
```

(Or simply organise images into per-condition subfolders and use **Group by subfolder**.)

## Requirements

- Python **3.10+** (developed and tested on the uv-managed **Python 3.14**; on native Windows stay
  on **3.10–3.13** for now — PySide6/torch wheels can lag a new CPython).
- [uv](https://docs.astral.sh/uv/) for the install/run flow above.
- **Windows:** runs under **WSL2** (recommended — commands above are verbatim) or **native Windows**
  (see [Native Windows](#native-windows-powershell--advanced)). The one-click desktop icon is WSL2-only.
- The `deep` extra (torch) for U-Net/CNN; the `agent` extra + a Claude login or API key for the
  Assistant.
- 8 GB RAM recommended.

## Acknowledgments

Developed with assistance from AI tools (Anthropic Claude, GitHub Copilot).

## License

MIT License — see [LICENSE](LICENSE).
