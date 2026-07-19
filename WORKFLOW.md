# SCAT Workflow Guide

**Version 2.2.1**

## Overview

SCAT runs in **one window**. It opens on the **Configure** screen; when a run finishes the
same window switches in place to the **Results** surface. Labeling, Train model, and Settings
live behind the top-bar **More** menu. The Assistant (optional Claude agent) opens as a side
dock from the top bar.

| Screen | Purpose |
|--------|---------|
| **Configure** (start) | Pick images + options, then **Run Analysis** |
| **Results** (after a run) | Review the answer, flag low-confidence deposits, edit, open the report |
| **More → Labeling** | Manually annotate deposits to build training data |
| **More → Train model** | Build a custom Random-Forest / CNN / U-Net model |
| **More → Settings** | Shortcuts, parallel-processing, the Assistant API key |

### Typical Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Most users:   Configure → Run Analysis → Results → (edit if needed) → report │
├─────────────────────────────────────────────────────────────────────────────┤
│  Custom model: More/Labeling → More/Train model → Configure → Run → Results   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Configure & Run

The start screen. Everything needed for a headless-quality run in one place.

### Steps

1. **Add images** — drag image files or a whole folder onto the drop zone (folders are searched
   recursively), or use **Choose images… / Choose folder…**. Formats: `.tif`, `.tiff`, `.png`, `.jpg`.
2. **Method** (optional) — the classification method: **Threshold** (simple circularity rule),
   **Random Forest** (the shipped `models/model_rf.pkl`, default), or **CNN** (ResNet-18, needs
   the `deep` extra). A blank **Classifier model** uses the bundled Random Forest.
3. **Compare groups** (optional) — toggle it on, then either **Group by subfolder** or
   **Load grouping CSV…** (see the format below).
4. **Output toggles** — Annotated images · Visualizations · HTML report (all on by default in the GUI).
5. Press the sticky **Run Analysis** button. Progress streams per image; the window then switches
   to the Results surface (the report is generated automatically if *HTML report* is on).

### Groups CSV Format

```csv
filename,group
CS_control_01.tif,Control
CS_control_02.tif,Control
CS_treated_01.tif,Treatment
CS_treated_02.tif,Treatment
```

(Or simply organise images into per-condition subfolders and use **Group by subfolder**.)

---

## 2. Results

The in-place results surface — answer first, then the worklist.

### What you see

- **Hero card** — the run's primary metric (deposits/image by default), image + deposit counts,
  and a neutral trust line (how many deposits fall below the confidence threshold). Actions:
  **Open report** · **Rebuild after edits** · **Open folder**.
- **Per-image table** — every image with its counts and a **Review** column showing how many
  low-confidence deposits to double-check.
- **Visualizations & statistics** below, in reading order.

### Editing results

1. **Double-click a row** to open the editor (same controls as Labeling, in *edit* mode).
2. Correct any misclassified deposits (1 / 2 / 3 keys).
3. **Save Changes** (or Ctrl+S) — rewrites that image's rows in the CSVs and refreshes the surface.
4. Use **Rebuild after edits** on the hero card to regenerate the report/stats/visualizations.

Use **New analysis** / **Re-run** to start over, or **Load previous results** to reopen an older
run's folder.

---

## 3. Labeling & Training (More menu)

### 3.1 Labeling

Create training data by manually annotating deposits. Open it from **More → Labeling**.

#### Steps

1. Open an image (File → Open or the toolbar).
2. Deposits are auto-detected; adjust labels as needed.
3. Save labels (Ctrl+S) — creates `imagename.labels.json`.

#### Labeling Controls

> 💡 Shortcuts can be customized in **More → Settings**. Changes require a restart.

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

Build a custom classifier from labeled data. Open it from **More → Train model**.

#### Steps

1. Select **Image Folder** containing your labeled images.
2. Select **Label Folder** (or check "Same as image folder").
3. Choose **Model Type**: Random Forest (recommended, fast), CNN (requires PyTorch), or
   U-Net Segmentation (requires PyTorch).
4. Set the output path for the model file.
5. Click **Train Model**.

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

1. Run analysis with the pre-trained model.
2. On the Results surface, double-click images with errors and correct the misclassified deposits.
3. Save changes.
4. Use the corrected data to train a new model in **More → Train model**.

### Option B: Manual Labeling from Scratch

1. Open **More → Labeling** and label your images one by one.
2. Train a model in **More → Train model**.

---

## 5. Command Line Interface

Every step is scriptable. Run via `uv run scat <cmd>`, an activated venv (`scat <cmd>`), or
`python -m scat.cli <cmd>`.

```bash
scat --version                 # print the SCAT version
scat gui                       # launch the GUI
scat analyze ./images -o ./results --model-type rf --stats --report
scat train --image-dir ./labeled --output model.pkl --model-type rf
scat label                     # launch the labeling GUI
scat chat                      # conversational agent in the terminal
```

### CLI `analyze` Options

| Option | Description |
|--------|-------------|
| `--model-type` | `threshold`, `rf`, or `cnn` |
| `--model-path` | Path to a trained model file (blank = bundled Random Forest) |
| `--threshold` | Detector **circularity** parameter (default: 0.6) — not a classifier probability |
| `--min-area` / `--max-area` | Deposit area bounds in pixels (defaults: 20 / 10000) |
| `-m/--metadata` + `--group-by` | Grouping CSV + the column to group by |
| `--annotate` | Save annotated images (**off** by default on the CLI) |
| `--visualize` | Save visualization plots (**off** by default on the CLI) |
| `--spatial` | Compute spatial dispersion metrics (writes `spatial_stats.json`) |
| `--stats` | Run group statistics |
| `--report` / `--no-report` | Force / skip the HTML report (default: on) |

> Note: unlike the GUI (which defaults Annotate/Visualize **on**), the CLI leaves them **off**
> so scripted runs stay fast; add the flags when you want those outputs.

---

## 6. Output Files Reference

A run writes a timestamped `results_YYYYMMDD_HHMMSS/` folder next to the analyzed images
(override with `-o/--output`).

### image_summary.csv (always)

Per-image statistics:
- `filename` — image filename
- `n_total` — **all** detected objects (Normal + ROD + Artifact)
- `n_normal`, `n_rod`, `n_artifact` — counts by classification
- `rod_fraction` — ROD / (Normal + ROD)
- `normal_mean_area`, `rod_mean_area` — mean deposit sizes
- `normal_mean_hue`, `rod_mean_hue` — mean color (pH indicator)
- `total_iod` — Total Integrated Optical Density
- `group` — experimental group (if provided)

> The report's **Deposit Count** and the "Total Deposits" figures use Normal + ROD only —
> artifacts are the reject class, excluded from deposit counts and metrics.

### all_deposits.csv (always)

Individual deposit measurements: `id`, `filename`, `label` (normal/rod/artifact), `area_px`,
`circularity`, `mean_hue`/`mean_saturation`/`mean_lightness`, `iod`, and the bounding box
`x`, `y`, `width`, `height`.

### Other outputs

| File / folder | When | Contents |
|---------------|------|----------|
| `run_manifest.json` | always | Reproducibility sidecar: SCAT version, git commit, dataset fingerprint, model + hash, detection settings, grouping |
| `deposits/*.labels.json` | always | Per-image deposit data with full contour coordinates (used for re-editing) |
| `report.html` | report on | Self-contained HTML report, charts embedded inline |
| `condition_summary.csv` | grouped runs | Group-level statistics |
| `groups/<group>_deposits.csv` | grouped runs | Per-group deposit tables |
| `annotated/*_annotated.png` | annotate on | Images with colored deposit outlines |
| `visualizations/*.png` | visualize on | `dashboard`, `pca_plot`, `heatmap`, `scatter_matrix`, `violin_*`, `mean_ci_*` |
| `spatial_stats.json` | spatial on | Spatial dispersion statistics |

---

## 7. Keyboard Shortcuts

Shortcuts can be customized in **More → Settings**. Changes take effect after restarting SCAT.

### Global

| Key | Action |
|-----|--------|
| Ctrl+S | Save |
| Ctrl+Q | Quit application |
| Ctrl+Z | Undo |
| Ctrl+R | Run analysis |

### Labeling / Edit Mode

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

- Increase **Min Area** (the analysis option / `--min-area`).
- Train a custom model with your data.

### Missing deposits (not detected)

- Decrease **Min Area** (`--min-area`).
- Check image quality and contrast.
- Train a U-Net segmentation model for difficult images.

### Low classification accuracy

- Add more labeled training data.
- Ensure balanced classes (similar numbers of Normal and ROD).
- Keep your labeling consistent.

### Slow performance

- Enable parallel processing in **More → Settings**.
- Reduce image resolution if very large.
- Use SSD storage for faster I/O.
