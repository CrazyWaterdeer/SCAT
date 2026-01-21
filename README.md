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

### Option 1: Windows Executable (Recommended)

1. Download `SCAT.zip` from [Releases](https://github.com/CrazyWaterdeer/SCAT/releases)
2. Extract the folder
3. Double-click `SCAT.exe` to run

The distribution includes a pre-trained Random Forest model (`Models/model_rf.pkl`).

### Option 2: From Source

```bash
git clone https://github.com/CrazyWaterdeer/SCAT.git
cd SCAT
pip install -e .
python -m scat.cli gui
```

**Optional** (for U-Net segmentation or CNN classification):
```bash
pip install torch torchvision
```

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

- Windows 10/11 (for executable)
- Python 3.10+ (for source installation)
- 8GB RAM recommended

## Acknowledgments

This project was developed with assistance from AI tools (GitHub Copilot, Anthropic Claude).

## License

MIT License - see [LICENSE](LICENSE).
