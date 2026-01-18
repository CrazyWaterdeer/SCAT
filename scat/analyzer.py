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

from .detector import DepositDetector, Deposit
from .features import FeatureExtractor
from .classifier import get_classifier, ClassifierConfig


class AnalysisResult:
    """Container for analysis results of a single image."""
    
    def __init__(self, filename: str, deposits: List[Deposit], dpi: float):
        self.filename = filename
        self.deposits = deposits
        self.dpi = dpi
        self.timestamp = datetime.now().isoformat()
    
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
        self.extractor = FeatureExtractor(dpi=dpi)
    
    def analyze_image(self, image_path: Union[str, Path], n_flies: int = 1) -> AnalysisResult:
        image_path = Path(image_path)
        img = Image.open(image_path)
        image = np.array(img)
        
        dpi = img.info.get('dpi', (self.dpi, self.dpi))[0]
        self.extractor = FeatureExtractor(dpi=dpi)
        
        deposits = self.detector.detect(image)
        deposits = self.extractor.extract_features(image, deposits)
        
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
            metadata: Optional metadata DataFrame
            progress_callback: Callback function(current, total) for progress updates
            parallel: Enable parallel processing (default True)
            max_workers: Number of worker threads (0 = auto)
        
        Returns:
            List of AnalysisResult objects in same order as input paths
        """
        n_images = len(image_paths)
        
        if not parallel or n_images <= 1:
            # Sequential processing
            results = []
            for i, path in enumerate(image_paths):
                if progress_callback:
                    progress_callback(i + 1, n_images)
                results.append(self.analyze_image(path))
            return results
        
        # Parallel processing with ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        # Determine worker count
        if max_workers <= 0:
            max_workers = self._get_safe_worker_count(n_images)
        
        results = [None] * n_images
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.analyze_image, path): i
                for i, path in enumerate(image_paths)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Create a placeholder result for failed images
                    path = image_paths[idx]
                    results[idx] = AnalysisResult(
                        filename=Path(path).name, deposits=[], dpi=self.dpi
                    )
                    print(f"Warning: Failed to analyze {path}: {e}")
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, n_images)
        
        return results
    
    def _get_safe_worker_count(self, n_images: int) -> int:
        """Determine safe number of workers based on system resources."""
        import os
        
        cpu_count = os.cpu_count() or 1
        
        # Try to check available memory
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            # ~200MB per image processing, be conservative
            memory_workers = max(1, int(available_gb / 0.3))
        except ImportError:
            memory_workers = 4  # Conservative default
        
        # Use half of CPUs to keep system responsive
        cpu_workers = max(1, cpu_count // 2)
        
        # Cap at 20 workers max for Auto mode and number of images
        optimal = min(cpu_workers, memory_workers, 20, n_images)
        
        return max(1, optimal)
    
    def generate_annotated_image(
        self, image: np.ndarray, deposits: List[Deposit], 
        show_labels: bool = True, skip_artifacts: bool = False
    ) -> np.ndarray:
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
        exclude_artifacts: bool = True
    ) -> pd.DataFrame:
        """
        Generate combined deposit data from all results.
        
        Args:
            results: List of analysis results
            metadata: Optional metadata DataFrame
            exclude_artifacts: If True, exclude artifact deposits from output
        
        Returns:
            DataFrame with global IDs assigned
        """
        df = pd.concat([r.to_dataframe() for r in results], ignore_index=True)
        
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
        exclude_artifacts: bool = True
    ):
        """Save individual CSV file for each image.
        
        Args:
            results: List of analysis results
            metadata: Optional metadata DataFrame
            save_json: Whether to save JSON files (always includes artifacts for training)
            exclude_artifacts: If True, exclude artifact deposits from CSV output
        """
        for result in results:
            df = result.to_dataframe()
            
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
        image_summary.to_csv(self.output_dir / 'image_summary.csv', index=False)
        
        # Condition summary
        if group_by and metadata is not None:
            condition_summary = self.generate_condition_summary(image_summary, group_by)
            condition_summary.to_csv(self.output_dir / 'condition_summary.csv', index=False)
        
        # Individual deposit files per image
        if save_individual:
            self.save_individual_deposits(results, metadata, save_json=save_json)
        
        # Combined deposit data
        deposit_data = self.generate_deposit_data(results, metadata)
        deposit_data.to_csv(self.output_dir / 'all_deposits.csv', index=False)
        
        # Group-specific deposit files
        if group_by and metadata is not None:
            group_col = group_by[0] if isinstance(group_by, list) else group_by
            if group_col in deposit_data.columns:
                groups_dir = self.output_dir / 'groups'
                groups_dir.mkdir(exist_ok=True)
                
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
                    group_df.to_csv(groups_dir / f'{safe_name}_deposits.csv', index=False)
        
        return {
            'film_summary': image_summary,  # Keep 'film_summary' key for backwards compatibility
            'image_summary': image_summary, 
            'deposit_data': deposit_data,
            'deposits_dir': str(self.deposits_dir)
        }
