# SCAT — Spot Classification and Analysis Tool

**Version 1.0.0**

A machine-learning tool for *Drosophila* excreta analysis. SCAT detects deposits in scanned
images, classifies each as **Normal**, **ROD** (Reproductive Oblong Deposit), or **Artifact**,
and turns a folder of images into per-image measurements, group statistics, and a shareable
HTML report — from the GUI, the command line, or a plain-language chat with a built-in Claude
assistant.

📖 **[Detailed Workflow Guide](WORKFLOW.md)**

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

## Installation

SCAT is developed with **[uv](https://docs.astral.sh/uv/)** and a project lockfile (`uv.lock`),
so a reproducible environment is one command. It ships a pre-trained Random Forest model
(`models/model_rf.pkl`) — no training required to start.

```bash
git clone https://github.com/CrazyWaterdeer/SCAT.git
cd SCAT

uv sync --extra agent      # reproducible env from uv.lock: core + GUI + Claude assistant
uv run scat gui            # launch the GUI
```

Prefer a plain editable install (no lockfile pinning)? Use:

```bash
uv venv                       # create .venv (uv picks a Python ≥3.10; the repo is tested on 3.14)
uv pip install -e ".[agent]"  # core + GUI + assistant  (drop [agent] for analysis + GUI only)
```

**Optional extras** (compose them, e.g. `uv sync --extra agent --extra deep`):

| Extra    | Adds                                   | Enables                                   |
| -------- | -------------------------------------- | ----------------------------------------- |
| `agent`  | pydantic, anthropic, claude-agent-sdk  | the conversational Assistant (Claude)     |
| `deep`   | torch, torchvision                     | U-Net segmentation and the CNN classifier |
| `pdf`    | weasyprint                             | PDF export of the report                  |
| `dev`    | pytest                                 | running the test suite                    |

Once installed, the `scat` command is available (`scat gui`, `scat chat`, `scat analyze`, …).
Run it via `uv run scat <cmd>`, an activated venv (`source .venv/bin/activate` then `scat <cmd>`),
or directly as `.venv/bin/python -m scat.cli <cmd>`.

### One-click desktop icon (WSL2 + WSLg)

On Windows 11 with WSL2, create a desktop shortcut that launches the GUI (assistant included)
through WSLg with no console window:

```bash
bash scripts/install_desktop_shortcut.sh
```

This puts an **SCAT** icon on your Windows desktop. It requires Windows interop (`powershell.exe`
on PATH) and expects the `.venv` to already exist — so **run the uv install first**. The shortcut
launches `.venv/bin/python -m scat.cli gui`; re-run the script any time to refresh it.

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
per-image progress.

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

- Python **3.10+** (developed and tested on the uv-managed **Python 3.14**).
- [uv](https://docs.astral.sh/uv/) for the install/run flow above.
- Windows 11 + WSL2/WSLg for the one-click desktop icon (optional).
- The `deep` extra (torch) for U-Net/CNN; the `agent` extra + a Claude login or API key for the
  Assistant.
- 8 GB RAM recommended.

## Acknowledgments

Developed with assistance from AI tools (Anthropic Claude, GitHub Copilot).

## License

MIT License — see [LICENSE](LICENSE).
