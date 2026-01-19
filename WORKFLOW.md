# SCAT Workflow Guide

**Version 1.0.0**

## Overview

SCAT provides three main tabs for different workflows:

| Tab | Purpose |
|-----|---------|
| **Analysis** | Run deposit detection and classification on images |
| **Results** | View, edit, and export analysis results |
| **Setup** | Create training data (Labeling) and build custom models (Training) |

### Typical Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Most Users: Analysis → Results → (Edit if needed) → Export                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Custom Model: Setup/Labeling → Setup/Training → Analysis → Results        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Analysis Tab

Run automated deposit detection and classification.

### Steps

1. **Image Folder**: Select folder containing your images (.tif, .tiff, .png, .jpg)
2. **Groups CSV** (Optional): Load a CSV file mapping filenames to experimental groups
3. **Classifier**: Select the model file
   - Use `Models/model_rf.pkl` (included) or your custom trained model
4. **Detection Method**: Choose Rule-based (default) or U-Net (requires trained model)
5. Click **Run Analysis**

### Groups CSV Format

```csv
filename,group
CS_control_01.tif,Control
CS_control_02.tif,Control
CS_treated_01.tif,Treatment
CS_treated_02.tif,Treatment
```

### Output Options

| Option | Description |
|--------|-------------|
| Generate Annotated Images | Save images with colored deposit outlines |
| Generate Visualizations | Create statistical plots (violin, PCA, heatmap, etc.) |
| Generate Report | Create HTML report with embedded statistics |

---

## 2. Results Tab

View and edit analysis results.

### Features

- **Image List**: Shows all analyzed images with deposit counts
- **Preview**: Thumbnail of selected image with annotations
- **Statistics**: Summary statistics for selected image or group
- **Edit Mode**: Double-click an image to open the editing window

### Editing Results

1. Double-click an image in the list
2. Edit window opens with the same controls as Labeling
3. Correct any misclassified deposits (1/2/3 keys)
4. Click **Save Changes** or press Ctrl+S
5. Click **Regenerate Report** in Results tab to update statistics

### Export Options

- **Open Folder**: Open results directory in file explorer
- **Open Report**: Open HTML report in browser
- **Load Previous**: Load results from a previous analysis session

---

## 3. Setup Tab

### 3.1 Labeling

Create training data by manually annotating deposits in images.

#### Steps

1. Click **Launch Labeling Window**
2. Open an image (File → Open or toolbar)
3. Deposits are auto-detected; adjust labels as needed
4. Save labels (Ctrl+S) - creates `imagename.labels.json`

#### Labeling Controls

> 💡 Shortcuts can be customized in Settings. Changes require restart.

| Key | Action |
|-----|--------|
| **1** | Label selected as Normal (green) |
| **2** | Label selected as ROD (red) |
| **3** | Label selected as Artifact (gray) |
| **Q** | Pan mode - drag to move view |
| **S** | Select mode - click or drag box to select |
| **A** | Add mode - draw to detect new deposits |
| **Delete** | Delete selected deposits |
| **R** | Merge selected into one deposit |
| **G** | Group selected (keeps separate contours) |
| **F** | Ungroup selected |
| **Ctrl+S** | Save labels |
| **Ctrl+Z** | Undo |
| **Space** (hold) | Temporary pan mode |

#### Add Mode Shapes

| Shape | Description |
|-------|-------------|
| Rectangle | Draw rectangle, auto-detect deposits inside |
| Circle | Draw circle, auto-detect deposits inside |
| Freeform | Draw freehand area, auto-detect deposits inside |
| Manual | Draw deposit boundary directly (no auto-detection) |

#### Merge vs Group

- **Merge**: Combines multiple contours into a single deposit (use when one deposit was incorrectly split)
- **Group**: Keeps separate contours but counts as one deposit (use for fragmented ROD)

### 3.2 Training

Build a custom classifier from labeled data.

#### Steps

1. Select **Image Folder** containing your labeled images
2. Select **Label Folder** (or check "Same as image folder")
3. Choose **Model Type**:
   - Random Forest (recommended, fast)
   - CNN (requires PyTorch)
   - U-Net Segmentation (requires PyTorch)
4. Set output path for the model file
5. Click **Train Model**

#### Recommended Training Data

| Model | Minimum Data |
|-------|--------------|
| Random Forest | 200+ labeled deposits |
| CNN | 500+ labeled deposits |
| U-Net | 20-30 fully labeled images |

---

## 4. Building Custom Models

If the pre-trained model doesn't work well for your data:

### Option A: Refine from Analysis Results (Recommended)

1. Run analysis with pre-trained model
2. Go to Results tab, double-click images with errors
3. Correct misclassified deposits
4. Save changes
5. Use corrected data to train a new model in Setup/Training

### Option B: Manual Labeling from Scratch

1. Go to Setup/Labeling
2. Open your images one by one
3. Manually label all deposits
4. Train model in Setup/Training

---

## 5. Command Line Interface

SCAT can also be run from the command line:

```bash
# Launch GUI
python -m scat.cli gui

# Run analysis
python -m scat.cli analyze ./images -o ./results --model-type rf --model-path model.pkl

# Train model
python -m scat.cli train --image-dir ./labeled --output model.pkl --model-type rf

# Launch labeling GUI only
python -m scat.cli label
```

### CLI Analysis Options

| Option | Description |
|--------|-------------|
| `--model-type` | threshold, rf, or cnn |
| `--model-path` | Path to trained model file |
| `--threshold` | Circularity threshold (default: 0.6) |
| `--min-area` | Minimum deposit area in pixels (default: 20) |
| `--max-area` | Maximum deposit area in pixels (default: 10000) |
| `--group-by` | Column name for grouping |
| `--annotate` | Generate annotated images |
| `--visualize` | Generate visualization plots |
| `--stats` | Perform statistical analysis |

---

## 6. Output Files Reference

### image_summary.csv

Per-image statistics with columns:
- `filename`: Image filename
- `n_total`: Total deposits (Normal + ROD)
- `n_normal`, `n_rod`, `n_artifact`: Counts by classification
- `rod_fraction`: ROD / (Normal + ROD)
- `normal_mean_area`, `rod_mean_area`: Mean deposit sizes
- `normal_mean_hue`, `rod_mean_hue`: Mean color (pH indicator)
- `total_iod`: Total Integrated Optical Density
- `group`: Experimental group (if provided)

### all_deposits.csv

Individual deposit measurements with columns:
- `id`: Unique deposit ID
- `filename`: Source image
- `label`: Classification (normal/rod/artifact)
- `area_px`: Area in pixels
- `circularity`: Shape circularity (0-1)
- `mean_hue`, `mean_saturation`, `mean_lightness`: Color features
- `iod`: Integrated Optical Density
- `x`, `y`, `width`, `height`: Bounding box

### deposits/*.labels.json

Per-image deposit data with full contour coordinates. Used for re-editing results.

### visualizations/

Statistical plots generated during analysis:
- `dashboard.png`: Summary dashboard with multiple plots
- `pca_plot.png`: PCA analysis of image features
- `heatmap.png`: Feature correlation heatmap
- `scatter_matrix.png`: Pairwise feature scatter plots
- `violin_*.png`: Violin plots for each metric by group
- `mean_ci_*.png`: Mean ± 95% CI plots for each metric

---

## 7. Keyboard Shortcuts

Shortcuts can be customized in **Settings** (⚙ button in the top-right corner). Changes take effect after restarting SCAT.

### Default Shortcuts

#### Global

| Key | Action |
|-----|--------|
| Ctrl+S | Save |
| Ctrl+Q | Quit application |
| Ctrl+Z | Undo |
| Ctrl+R | Run analysis |

#### Labeling / Edit Mode

| Key | Action |
|-----|--------|
| 1 | Label as Normal |
| 2 | Label as ROD |
| 3 | Label as Artifact |
| Q | Pan mode |
| S | Select mode |
| A | Add mode |
| Delete | Delete selected |
| R | Merge selected |
| G | Group selected |
| F | Ungroup selected |
| Space (hold) | Temporary pan |

---

## 8. Troubleshooting

### Too many false positives (non-deposits detected)

- Increase **Min Area** in Settings
- Train a custom model with your data

### Missing deposits (not detected)

- Decrease **Min Area** in Settings
- Check image quality and contrast
- Train U-Net segmentation model for difficult images

### Low classification accuracy

- Add more labeled training data
- Ensure balanced classes (similar numbers of Normal and ROD)
- Check that your labeling is consistent

### Slow performance

- Enable parallel processing in Settings
- Reduce image resolution if very large
- Use SSD storage for faster I/O
