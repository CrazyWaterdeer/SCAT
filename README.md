# SCAT - Spot Classification and Analysis Tool

**Version 1.0.0**

A machine learning-based analysis tool for *Drosophila* excreta classification and quantification. SCAT automatically detects, classifies, and analyzes deposits to distinguish Normal deposits from ROD (Reproductive Oblong Deposits).

📖 **[Detailed Workflow Guide](WORKFLOW.md)**

## Features

- **Automatic Detection**: Rule-based adaptive thresholding or U-Net deep learning segmentation
- **ML Classification**: Random Forest or CNN classifiers to categorize deposits as Normal, ROD, or Artifact
- **Statistical Analysis**: Comprehensive statistics including t-test, Mann-Whitney U, ANOVA, Kruskal-Wallis with effect sizes
- **Visualizations**: Violin plots, box plots, PCA, correlation heatmaps, and more
- **Interactive Editing**: Review and correct classifications in the Results tab
- **Reporting**: HTML reports with embedded visualizations, CSV/Excel export

## Installation

### From source

```bash
git clone https://github.com/CrazyWaterdeer/SCAT.git
cd SCAT
pip install -e .            # core analysis + GUI
python -m scat.cli gui
```

The repo ships a pre-trained Random Forest model (`models/model_rf.pkl`).

**Optional extras:**
```bash
pip install -e .[agent]     # the conversational Assistant (Claude) — see "AI Assistant" below
pip install torch torchvision   # U-Net segmentation or CNN classification
```

### One-click desktop icon (WSL2 + WSLg)

If you run SCAT under WSL2 on Windows 11, create a desktop shortcut that launches the GUI
(with the Assistant) via WSLg — no console window:

```bash
bash scripts/install_desktop_shortcut.sh
```

This puts an **SCAT** icon on your Windows desktop — one-click launch of the full GUI (Assistant
included), no console window.

## AI Assistant (conversational agent)

SCAT includes an optional Claude-powered agent that runs the whole pipeline from a plain-language
request ("analyze this folder and compare the groups"). It infers the experimental grouping from
your filenames/subfolders, runs detection → classification → statistics → report, and streams
per-image progress.

```bash
pip install -e .[agent]
# then either:
python -m scat.cli chat        # conversational agent in the terminal
python -m scat.cli gui         # GUI — click "💬 Assistant" to open the dock
```

**Backends** (pick in the dock's Provider selector, or `--backend`):
- **Subscription** (default, no API charges) — uses your local Claude Code login; install the
  [`claude` CLI](https://claude.com/claude-code) and log in.
- **API** (billed) — set `ANTHROPIC_API_KEY`.

Without the `[agent]` extra (or a backend), the GUI still runs — the Assistant dock just shows an
install/login hint.

## Quick Start

1. Launch SCAT
2. In the **Analysis** tab:
   - Select your image folder
   - (Optional) Load a groups CSV for statistical comparison
   - Select classifier model (`Models/model_rf.pkl`)
   - Click **Run Analysis**
3. View results in the **Results** tab
4. Double-click any image to edit classifications if needed

> 💡 The pre-trained model works well for standard Bromophenol Blue-stained deposits. If results are unsatisfactory, see [Building Custom Models](WORKFLOW.md#building-custom-models).

## Output Files

Analysis creates a timestamped folder with:

```
results_YYYYMMDD_HHMMSS/
├── image_summary.csv       # Per-image statistics
├── all_deposits.csv        # Individual deposit measurements
├── condition_summary.csv   # Group-level statistics (if groups provided)
├── report.html             # Interactive HTML report
├── annotated/              # Images with deposit annotations
├── deposits/               # Per-image JSON and CSV files
├── groups/                 # Per-group deposit data
└── visualizations/         # Statistical plots (violin, PCA, heatmap, etc.)
```

## Groups CSV Format

To compare experimental conditions, provide a CSV mapping filenames to groups:

```csv
filename,group
image001.tif,Control
image002.tif,Control
image003.tif,Treatment
image004.tif,Treatment
```

## Requirements

- Python 3.10+
- Windows 11 + WSL2/WSLg for the one-click desktop icon (optional)
- 8GB RAM recommended

## Acknowledgments

This project was developed with assistance from AI tools (GitHub Copilot, Anthropic Claude).

## License

MIT License - see [LICENSE](LICENSE).
