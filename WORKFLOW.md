# SCAT Workflow Guide

**Version 1.0.0**

## Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────────┐    ┌──────────────┐
│  Labeling   │ →  │   Training   │ →  │  Detection  │ →  │ Classification │ →  │   Analysis   │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────────┘    └──────────────┘
```

**Most users**: Start with pre-trained model → Analysis → Review results

**Custom model**: Labeling → Training → Analysis

---

## 1. Labeling

Create training data by annotating deposits.

### Access
- Main GUI → Setup Tab → Labeling

### Keyboard Shortcuts

| Key | Function |
|-----|----------|
| **1 / 2 / 3** | Label as Normal / ROD / Artifact |
| **Q** | Pan mode |
| **S** | Select mode |
| **A** | Add mode (draw new deposit) |
| **Delete** | Delete selected |
| **R** | Merge selected |
| **G** | Group selected |
| **F** | Ungroup |
| **Ctrl+S** | Save |
| **Ctrl+Z** | Undo |
| **Space** (hold) | Pan while held |

### Merge vs Group

- **Merge**: Combine multiple contours into one (single deposit detected as multiple)
- **Group**: Keep separate contours but count as one (fragmented ROD)

### Output

Labels saved as `image_name.labels.json`.

---

## 2. Training

Build models from labeled data.

### Model Types

| Model | Use | Recommended Data |
|-------|-----|------------------|
| **Random Forest** | Classification (Normal/ROD/Artifact) | 200+ deposits |
| **U-Net** | Detection (pixel-level segmentation) | 20-30 images |
| **CNN** | Classification (image patches) | 500+ deposits |

### Commands

```bash
# Random Forest
python -m scat.cli train --image-dir ./labeled --output model.pkl

# U-Net (requires PyTorch)
python -m scat.cli train --model-type unet --image-dir ./labeled --output model_unet.pt
```

---

## 3. Detection

Find deposit locations in images.

| Method | Description | Training Required |
|--------|-------------|-------------------|
| **Rule-based** | Adaptive thresholding (default) | No |
| **U-Net** | Deep learning segmentation | Yes |

---

## 4. Classification

Assign labels to detected deposits.

| Method | Description | Training Required |
|--------|-------------|-------------------|
| **Threshold** | Circularity < 0.6 → ROD, ≥ 0.6 → Normal | No |
| **Random Forest** | 7-feature ML classifier | Yes |
| **CNN** | Image patch classifier | Yes |

> ⚠️ Threshold method cannot detect Artifacts. Use Random Forest for accurate results.

---

## 5. Analysis

### Running Analysis

1. Select image folder
2. (Optional) Load groups CSV for statistical comparison
3. Select classifier model
4. Click "Run Analysis"

### Groups CSV Format

```csv
filename,group
image1.tif,Control
image2.tif,Treatment
```

### Statistical Tests

| Groups | Parametric | Non-parametric |
|--------|------------|----------------|
| 2 | t-test | Mann-Whitney U |
| 3+ | ANOVA + Tukey HSD | Kruskal-Wallis + Dunn |

Multiple comparisons corrected with Holm-Bonferroni method.

---

## 6. Visualizations

- **Statistical**: Violin plot, box plot, mean ± 95% CI, effect size forest plot
- **Feature**: PCA, correlation heatmap
- **Spatial**: Density heatmap, nearest neighbor distance

---

## 7. Output Files

| File | Description |
|------|-------------|
| `image_summary.csv` | Per-image statistics (counts, ROD fraction, etc.) |
| `all_deposits.csv` | Individual deposit data (coordinates, features, labels) |
| `report.html` | HTML report with embedded visualizations |
| `annotated/` | Images with deposit annotations |
| `deposits/` | Per-image JSON files |

---

## 8. Building Custom Models

If pre-trained models don't work well for your data:

### Option A: Refine from Pre-trained (Recommended)

1. Run analysis with pre-trained model
2. Review results → Correct misclassified deposits
3. Train new model with corrected data

### Option B: Manual Labeling

1. Open Labeling tool
2. Annotate deposits manually
3. Train model

Both approaches can be combined.

---

## 9. Troubleshooting

**Too many false positives**
- Use trained model instead of rule-based detection
- Increase minimum area threshold

**Missing deposits**
- Enable sensitive detection mode
- Train U-Net on similar images

**Low classification accuracy**
- Add more labeled data
- Ensure balanced classes

---

## 10. Keyboard Shortcuts Summary

### Labeling

| Key | Action |
|-----|--------|
| 1, 2, 3 | Label Normal, ROD, Artifact |
| Q, S, A | Pan, Select, Add mode |
| Delete | Delete selected |
| R | Merge |
| G | Group |
| F | Ungroup |
| Ctrl+S | Save |
| Ctrl+Z | Undo |

### Results Viewer

| Key | Action |
|-----|--------|
| 1, 2, 3 | Relabel selected |
| Delete | Delete selected |
| Ctrl+Z | Undo |
| Esc | Close |
