# SCAT - Spot Classification and Analysis Tool

**Version 1.0.0**

A machine learning-based analysis tool for *Drosophila* excreta classification and quantification. SCAT automatically detects, classifies, and analyzes deposits to distinguish ROD (Reproductive Oblong Deposits) from Normal deposits.

📖 **[Detailed Workflow Guide](WORKFLOW.md)**

## Features

- **Detection**: Rule-based adaptive thresholding or U-Net segmentation
- **Classification**: Random Forest or CNN-based ML classifiers
- **Statistical Analysis**: t-test, Mann-Whitney U, ANOVA, Kruskal-Wallis, effect size
- **Visualizations**: Violin plots, box plots, PCA, heatmaps, and more
- **Reporting**: HTML reports, CSV/Excel export, annotated images

## Installation & Launch

### Option 1: Windows Executable (Recommended)

1. Download `SCAT.exe` from [Releases](https://github.com/CrazyWaterdeer/SCAT/releases)
2. Double-click `SCAT.exe` to run

### Option 2: From Source

```bash
git clone https://github.com/CrazyWaterdeer/SCAT.git
cd SCAT
pip install -e .
python -m scat.cli gui
```

**Optional** (for U-Net/CNN training):
```bash
pip install torch torchvision
```

## Quick Start

1. In Analysis tab, select `models/model_rf.pkl` as classifier
2. Select image folder and click "Run Analysis"
3. View results in Results tab

> 💡 Try the pre-trained model first. If it doesn't fit your data, see [Building Custom Models](WORKFLOW.md#8-building-custom-models).

## Pre-trained Models

| Model | File | Description |
|-------|------|-------------|
| Random Forest | `models/model_rf.pkl` | Recommended classifier |
| U-Net | `models/model_unet.pt` | Segmentation (if available) |

## Output

```
results_YYYYMMDD_HHMMSS/
├── image_summary.csv      # Per-image statistics
├── all_deposits.csv       # Individual deposit data
├── report.html            # HTML report with visualizations
├── annotated/             # Annotated images
└── visualizations/        # Generated plots
```

## Acknowledgments

This project was developed with assistance from AI tools (GitHub Copilot Claude Opus 4.5, Anthropic Claude Opus 4.5).

## License

MIT License - see [LICENSE](LICENSE).
