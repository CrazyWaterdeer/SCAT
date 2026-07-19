"""
Analysis pipeline - combines detection, feature extraction, and classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

from PIL import Image
import cv2

from .artifacts import IMAGE_SUMMARY, ALL_DEPOSITS, CONDITION_SUMMARY

from .detector import DepositDetector, Deposit
from .features import FeatureExtractor
from .classifier import get_classifier, ClassifierConfig


def to_rgb(img: Image.Image) -> Image.Image:
    """A PIL image as 3-channel RGB, compositing any alpha over WHITE first. Excreta scans are
    light paper, so transparent padding must become white — a plain ``convert('RGB')`` would drop
    alpha and expose the hidden (usually black) RGB, which adaptive thresholding then reads as dark
    foreground (false deposits) and which leaves OpenCV-drawn annotations invisible. A no-op
    (identical pixels) for opaque 8-bit RGB, so pipeline CSV parity is preserved."""
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGBA')
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert('RGB')


class AnalysisResult:
    """Container for analysis results of a single image."""
    
    def __init__(self, filename: str, deposits: List[Deposit], dpi: float):
        self.filename = filename
        self.deposits = deposits
        self.dpi = dpi
        self.timestamp = datetime.now().isoformat()
        # True only for a placeholder produced by a real analysis failure (unreadable
        # image / worker give-up), NOT for a legitimately clean 0-deposit image.
        self.failed = False
    
    @property
    def n_total(self) -> int:
        return len(self.deposits)
    
    @property
    def n_rod(self) -> int:
        return sum(1 for d in self.deposits if d.label == "rod")
    
    @property
    def n_normal(self) -> int:
        return sum(1 for d in self.deposits if d.label == "normal")
    
    @property
    def n_artifact(self) -> int:
        return sum(1 for d in self.deposits if d.label == "artifact")
    
    @property
    def rod_fraction(self) -> float:
        valid = self.n_rod + self.n_normal
        return self.n_rod / valid if valid > 0 else 0.0
    
    def get_summary(self) -> Dict:
        # Single pass through deposits instead of multiple iterations
        normal_deposits = []
        rod_deposits = []
        artifact_count = 0
        
        for d in self.deposits:
            if d.label == "normal":
                normal_deposits.append(d)
            elif d.label == "rod":
                rod_deposits.append(d)
            elif d.label == "artifact":
                artifact_count += 1
        
        valid_deposits = normal_deposits + rod_deposits
        
        summary = {
            'filename': self.filename,
            'n_total': self.n_total,
            # Order: Normal → ROD → Artifact
            'n_normal': len(normal_deposits),
            'n_rod': len(rod_deposits),
            'n_artifact': artifact_count,
            'rod_fraction': self.rod_fraction,
        }
        
        # Normal statistics
        if normal_deposits:
            summary['normal_mean_area'] = np.mean([d.area for d in normal_deposits])
            summary['normal_std_area'] = np.std([d.area for d in normal_deposits])
            summary['normal_mean_iod'] = np.mean([d.iod for d in normal_deposits])
            summary['normal_total_iod'] = sum(d.iod for d in normal_deposits)
            summary['normal_mean_hue'] = np.mean([d.mean_hue for d in normal_deposits])
            summary['normal_mean_lightness'] = np.mean([d.mean_lightness for d in normal_deposits])
            summary['normal_mean_circularity'] = np.mean([d.circularity for d in normal_deposits])
        else:
            summary['normal_mean_area'] = np.nan
            summary['normal_std_area'] = np.nan
            summary['normal_mean_iod'] = np.nan
            summary['normal_total_iod'] = 0
            summary['normal_mean_hue'] = np.nan
            summary['normal_mean_lightness'] = np.nan
            summary['normal_mean_circularity'] = np.nan
        
        # ROD statistics
        if rod_deposits:
            summary['rod_mean_area'] = np.mean([d.area for d in rod_deposits])
            summary['rod_std_area'] = np.std([d.area for d in rod_deposits])
            summary['rod_mean_iod'] = np.mean([d.iod for d in rod_deposits])
            summary['rod_total_iod'] = sum(d.iod for d in rod_deposits)
            summary['rod_mean_hue'] = np.mean([d.mean_hue for d in rod_deposits])
            summary['rod_mean_lightness'] = np.mean([d.mean_lightness for d in rod_deposits])
            summary['rod_mean_circularity'] = np.mean([d.circularity for d in rod_deposits])
        else:
            summary['rod_mean_area'] = np.nan
            summary['rod_std_area'] = np.nan
            summary['rod_mean_iod'] = np.nan
            summary['rod_total_iod'] = 0
            summary['rod_mean_hue'] = np.nan
            summary['rod_mean_lightness'] = np.nan
            summary['rod_mean_circularity'] = np.nan
        
        # Total statistics
        if valid_deposits:
            summary['total_iod'] = sum(d.iod for d in valid_deposits)
            summary['mean_area'] = np.mean([d.area for d in valid_deposits])
            summary['mean_iod'] = np.mean([d.iod for d in valid_deposits])
        else:
            summary['total_iod'] = 0
            summary['mean_area'] = np.nan
            summary['mean_iod'] = np.nan
        
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        extractor = FeatureExtractor(dpi=self.dpi)
        records = [extractor.to_feature_dict(d) for d in self.deposits]
        df = pd.DataFrame(records)
        df['filename'] = self.filename
        return df


class Analyzer:
    """Main analysis pipeline."""
    
    def __init__(
        self,
        detector: Optional[DepositDetector] = None,
        classifier_config: Optional[ClassifierConfig] = None,
        dpi: float = 600.0
    ):
        self.detector = detector or DepositDetector()
        self.classifier_config = classifier_config or ClassifierConfig()
        self.classifier = get_classifier(self.classifier_config)
        self.dpi = dpi

    def analyze_image(self, image_path: Union[str, Path]) -> AnalysisResult:
        image_path = Path(image_path)
        img = Image.open(image_path)
        dpi = img.info.get('dpi', (self.dpi, self.dpi))[0]
        # Normalize to 3-channel RGB (compositing alpha over white) so grayscale/palette/RGBA/CMYK
        # inputs don't crash detection/feature extraction or misread transparency as deposits.
        image = np.array(to_rgb(img))
        # Use a local extractor (not self.extractor): analyze_batch runs this
        # method concurrently in a thread pool, so a shared instance attribute
        # would let one image's DPI clobber another's mid-extraction.
        extractor = FeatureExtractor(dpi=dpi)

        deposits = self.detector.detect(image)
        deposits = extractor.extract_features(image, deposits)
        
        # Call predict for each classifier
        from .classifier import ThresholdClassifier, RandomForestClassifier, CNNClassifier
        if isinstance(self.classifier, (ThresholdClassifier, RandomForestClassifier)):
            deposits = self.classifier.predict(deposits)
        elif isinstance(self.classifier, CNNClassifier):
            deposits = self.classifier.predict(deposits, image)
        else:
            # Fallback
            deposits = self.classifier.predict(deposits)
        
        return AnalysisResult(filename=image_path.name, deposits=deposits, dpi=dpi)
    
    def analyze_batch(
        self, image_paths: List[Union[str, Path]],
        metadata: Optional[pd.DataFrame] = None, progress_callback=None,
        parallel: bool = True, max_workers: int = 0
    ) -> List[AnalysisResult]:
        """
        Analyze multiple images.
        
        Args:
            image_paths: List of image paths to analyze
            metadata: Optional metadata DataFrame (unused here; applied later at save time)
            progress_callback: Callback function(current, total) for progress updates;
                may raise to cancel the batch cooperatively (checked per completed image)
            parallel: Enable parallel processing (default True)
            max_workers: Number of workers (0 = hardware-aware auto)

        Returns:
            List of AnalysisResult objects in same order as input paths
        """
        # Engine selection (fork process pool / capped thread pool / sequential) and
        # hardware-aware worker sizing live in scat.parallel. The per-image pipeline is
        # GIL-bound, so a fork pool whose workers inherit the parent-loaded model
        # copy-on-write is what actually uses multi-core hardware (~12x measured);
        # threads are a capped fallback. Progress, cooperative cancel, per-image failure
        # isolation, and input ordering are all preserved there.
        from . import parallel as _parallel

        n_images = len(image_paths)
        if not parallel:
            max_workers = 1  # forces the sequential engine
        results, engine = _parallel.run_batch(
            self, image_paths, progress_callback=progress_callback, max_workers=max_workers)
        if parallel and n_images > 1:
            print(f"[scat] analyzed {n_images} images | engine={engine}")
        return results

    @staticmethod
    def generate_annotated_image(
        image: np.ndarray, deposits: List,
        show_labels: bool = True, skip_artifacts: bool = False
    ) -> np.ndarray:
        # Uses no Analyzer state — deposits only need .label/.contour/.id/.centroid, so this
        # also accepts the lightweight objects from deposits_from_labels_json (regenerate path).
        result = image.copy()
        colors = {'rod': (255, 0, 0), 'normal': (0, 255, 0), 'artifact': (128, 128, 128), 'unknown': (255, 255, 0)}

        for d in deposits:
            if skip_artifacts and d.label == 'artifact':
                continue
            color = colors.get(d.label, colors['unknown'])
            cv2.drawContours(result, [d.contour], -1, color, 1)
            if show_labels:
                cv2.putText(result, f"{d.id}", (d.centroid[0] + 5, d.centroid[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return result


def deposits_from_labels_json(json_path) -> list:
    """Reconstruct lightweight, annotate-only deposit objects from a *.labels.json file.

    Used by the GUI 'regenerate report after editing' path: after a user corrects labels the
    edited JSON is the source of truth, so we rebuild just what generate_annotated_image reads
    (.id/.label/.contour/.centroid). Not full Deposit objects — geometry like perimeter/aspect
    ratio isn't stored and would have to be fabricated. Prefers the saved (x, y) centroid.
    """
    import json
    from types import SimpleNamespace
    with open(json_path) as f:
        data = json.load(f)
    out = []
    for d in data.get('deposits', []):
        contour = np.array(d.get('contour', []), dtype=np.int32).reshape((-1, 1, 2))
        if 'x' in d and 'y' in d:
            centroid = (int(d['x']), int(d['y']))
        elif contour.size:
            pts = contour.reshape(-1, 2)
            centroid = (int(pts[:, 0].mean()), int(pts[:, 1].mean()))
        else:
            centroid = (0, 0)
        out.append(SimpleNamespace(id=d.get('id', 0), label=d.get('label', 'unknown'),
                                   contour=contour, centroid=centroid))
    return out


class ReportGenerator:
    """Generate analysis reports."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.deposits_dir = self.output_dir / 'deposits'
        self.deposits_dir.mkdir(exist_ok=True)
    
    def generate_film_summary(self, results: List[AnalysisResult], metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = pd.DataFrame([r.get_summary() for r in results])
        if metadata is not None:
            df = df.merge(metadata, on='filename', how='left')
        return df
    
    def generate_condition_summary(self, film_summary: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
        numeric_cols = film_summary.select_dtypes(include=[np.number]).columns
        agg_funcs = {col: ['mean', 'std', 'count'] for col in numeric_cols if col not in group_by}
        condition_summary = film_summary.groupby(group_by).agg(agg_funcs)
        condition_summary.columns = ['_'.join(col).strip() for col in condition_summary.columns]
        return condition_summary.reset_index()
    
    def generate_deposit_data(
        self, results: List[AnalysisResult],
        metadata: Optional[pd.DataFrame] = None,
        exclude_artifacts: bool = True,
        frames: Optional[List[pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Generate combined deposit data from all results.

        Args:
            results: List of analysis results
            metadata: Optional metadata DataFrame
            exclude_artifacts: If True, exclude artifact deposits from output
            frames: Pre-computed per-result ``to_dataframe()`` frames (save_all builds
                these once and shares them with save_individual_deposits to avoid the
                per-deposit feature-dict work running twice). Read-only here (concat
                copies); falls back to computing them when not supplied.

        Returns:
            DataFrame with global IDs assigned
        """
        if frames is None:
            frames = [r.to_dataframe() for r in results]
        df = pd.concat(frames, ignore_index=True)
        
        # Exclude artifacts if requested
        if exclude_artifacts and 'label' in df.columns:
            df = df[df['label'] != 'artifact'].copy()
        
        # Rename original id to image_id and assign global id
        if 'id' in df.columns:
            df = df.rename(columns={'id': 'image_id'})
            df.insert(0, 'id', range(1, len(df) + 1))
        
        if metadata is not None:
            df = df.merge(metadata, on='filename', how='left')
        
        return df
    
    def save_individual_deposits(
        self, results: List[AnalysisResult],
        metadata: Optional[pd.DataFrame] = None,
        save_json: bool = True,
        exclude_artifacts: bool = True,
        frames: Optional[List[pd.DataFrame]] = None
    ):
        """Save individual CSV file for each image.

        Args:
            results: List of analysis results
            metadata: Optional metadata DataFrame
            save_json: Whether to save JSON files (always includes artifacts for training)
            exclude_artifacts: If True, exclude artifact deposits from CSV output
            frames: Pre-computed per-result ``to_dataframe()`` frames (shared with
                generate_deposit_data via save_all). Each is used read-only as the source
                and copied before any mutation; falls back to computing per result.
        """
        for i, result in enumerate(results):
            df = frames[i] if frames is not None else result.to_dataframe()

            # Exclude artifacts from CSV if requested
            if exclude_artifacts and 'label' in df.columns:
                df = df[df['label'] != 'artifact'].copy()
            
            if metadata is not None:
                df = df.merge(metadata, on='filename', how='left')
            
            # Reorder columns: id, position, Normal→ROD→Artifact related, then rest
            priority_cols = ['id', 'filename', 'x', 'y', 'width', 'height', 'label', 'confidence',
                           'area_px', 'area_um2', 'circularity', 'aspect_ratio',
                           'mean_hue', 'mean_saturation', 'mean_lightness', 'iod']
            
            existing_priority = [c for c in priority_cols if c in df.columns]
            other_cols = [c for c in df.columns if c not in priority_cols]
            df = df[existing_priority + other_cols]
            
            # Save with image name
            image_stem = Path(result.filename).stem
            filepath = self.deposits_dir / f'{image_stem}_deposits.csv'
            df.to_csv(filepath, index=False)
            
            # JSON always includes all deposits (including artifacts) for training
            if save_json:
                self._save_contour_json(result, image_stem)
    
    def _save_contour_json(self, result: AnalysisResult, image_stem: str):
        """Save deposit contours as JSON (unified format with labeling)."""
        import json
        
        deposits_data = []
        for d in result.deposits:
            deposit_dict = {
                'id': d.id,
                'contour': d.contour.squeeze().tolist() if d.contour is not None else [],
                'x': d.centroid[0],
                'y': d.centroid[1],
                'width': d.width,
                'height': d.height,
                'area': float(d.area),
                'circularity': float(d.circularity),
                'label': d.label,
                'confidence': float(d.confidence),
                'merged': getattr(d, 'merged', False),
                'group_id': getattr(d, 'group_id', None)
            }
            deposits_data.append(deposit_dict)
        
        # Use unified format: *.labels.json
        json_path = self.deposits_dir / f'{image_stem}.labels.json'
        with open(json_path, 'w') as f:
            json.dump({
                'image_file': result.filename,
                'next_group_id': 1,
                'deposits': deposits_data
            }, f, indent=2)
    
    def save_all(
        self, 
        results: List[AnalysisResult], 
        metadata: Optional[pd.DataFrame] = None, 
        group_by: Optional[List[str]] = None,
        save_individual: bool = True,
        save_json: bool = True
    ) -> Dict:
        """
        Save all reports.
        
        Args:
            results: List of analysis results
            metadata: Optional metadata DataFrame
            group_by: Columns for condition grouping
            save_individual: Whether to save individual deposit files per image
            save_json: Whether to save JSON files for retraining
        """
        # Image summary (formerly film_summary)
        image_summary = self.generate_film_summary(results, metadata)
        image_summary.to_csv(self.output_dir / IMAGE_SUMMARY, index=False)
        
        # Condition summary
        if group_by and metadata is not None:
            condition_summary = self.generate_condition_summary(image_summary, group_by)
            condition_summary.to_csv(self.output_dir / CONDITION_SUMMARY, index=False)
        
        # Build each result's deposit frame once and share it between the per-image
        # CSVs and the combined all_deposits.csv (to_dataframe rebuilds a FeatureExtractor
        # and every deposit's feature dict, so computing it twice is pure waste).
        frames = [r.to_dataframe() for r in results]

        # Individual deposit files per image
        if save_individual:
            self.save_individual_deposits(results, metadata, save_json=save_json, frames=frames)

        # Combined deposit data
        deposit_data = self.generate_deposit_data(results, metadata, frames=frames)
        deposit_data.to_csv(self.output_dir / ALL_DEPOSITS, index=False)
        
        # Group-specific deposit files
        if group_by and metadata is not None:
            group_col = group_by[0] if isinstance(group_by, list) else group_by
            if group_col in deposit_data.columns:
                groups_dir = self.output_dir / 'groups'
                groups_dir.mkdir(exist_ok=True)

                seen_stems = {}   # disambiguate distinct group names that sanitize alike
                for group_name in deposit_data[group_col].dropna().unique():
                    group_df = deposit_data[deposit_data[group_col] == group_name].copy()
                    # Re-assign IDs within group
                    group_df['id'] = range(1, len(group_df) + 1)
                    # Sanitize for Windows/cross-platform file names
                    safe_name = str(group_name)
                    # < > → +
                    for char in ['<', '>']:
                        safe_name = safe_name.replace(char, '+')
                    # / \ | → -
                    for char in ['/', '\\', '|']:
                        safe_name = safe_name.replace(char, '-')
                    # : * ? " → _
                    for char in [':', '*', '?', '"']:
                        safe_name = safe_name.replace(char, '_')
                    safe_name = safe_name.replace(' ', '_')
                    # Two distinct group names can sanitize to the same stem (e.g. 'A/B' and
                    # 'A|B' → 'A-B'); disambiguate so the second no longer overwrites the first.
                    if safe_name in seen_stems:
                        stem = safe_name
                        while safe_name in seen_stems:
                            seen_stems[stem] += 1
                            safe_name = f'{stem}_{seen_stems[stem]}'
                    seen_stems[safe_name] = seen_stems.get(safe_name, 0)
                    group_df.to_csv(groups_dir / f'{safe_name}_deposits.csv', index=False)
        
        return {
            'film_summary': image_summary,  # Keep 'film_summary' key for backwards compatibility
            'image_summary': image_summary, 
            'deposit_data': deposit_data,
            'deposits_dir': str(self.deposits_dir)
        }
