"""Morphology (shape) analysis."""

import numpy as np
import pandas as pd
from typing import Dict

from .common import StatisticalAnalyzer, _compare_metric_between_groups, coefficient_of_variation


class MorphologyAnalyzer:
    """
    Analyze deposit morphology (shape characteristics).
    
    Provides:
    - Circularity distribution analysis
    - Aspect ratio analysis
    - Shape classification
    - Morphological abnormality detection
    """
    
    # Shape classification thresholds
    CIRCULARITY_ROUND = 0.8      # Above = round
    CIRCULARITY_OVAL = 0.5       # 0.5-0.8 = oval
    # Below 0.5 = irregular
    
    ASPECT_RATIO_SQUARE = 1.3    # Below = roughly square
    ASPECT_RATIO_ELONGATED = 2.0 # Above = elongated
    
    def __init__(self, alpha: float = 0.05):
        from scipy import stats
        self.stats = stats
        self.alpha = alpha
    
    def _classify_shape(self, circularity: float, aspect_ratio: float) -> str:
        """Classify deposit shape based on circularity and aspect ratio."""
        if circularity >= self.CIRCULARITY_ROUND:
            return 'round'
        elif circularity >= self.CIRCULARITY_OVAL:
            if aspect_ratio < self.ASPECT_RATIO_SQUARE:
                return 'oval'
            else:
                return 'elongated_oval'
        else:
            if aspect_ratio >= self.ASPECT_RATIO_ELONGATED:
                return 'elongated_irregular'
            else:
                return 'irregular'
    
    def analyze_morphology(self, deposits_df: pd.DataFrame) -> Dict:
        """
        Analyze morphological features of deposits.
        
        Args:
            deposits_df: DataFrame with 'circularity', 'aspect_ratio' columns
            
        Returns:
            Dict with morphology analysis
        """
        required_cols = ['circularity', 'aspect_ratio']
        if not all(col in deposits_df.columns for col in required_cols):
            return {'error': f'Required columns not found: {required_cols}'}
        
        # Filter valid deposits
        if 'label' in deposits_df.columns:
            valid_df = deposits_df[deposits_df['label'].isin(['normal', 'rod'])]
        else:
            valid_df = deposits_df
        
        circularity = valid_df['circularity'].dropna().values
        aspect_ratio = valid_df['aspect_ratio'].dropna().values
        
        if len(circularity) < 2:
            return {'error': 'Insufficient data'}
        
        # Shape classification
        shapes = []
        for c, ar in zip(circularity, aspect_ratio):
            shapes.append(self._classify_shape(c, ar))
        
        n_total = len(shapes)
        shape_counts = {
            'round': shapes.count('round'),
            'oval': shapes.count('oval'),
            'elongated_oval': shapes.count('elongated_oval'),
            'irregular': shapes.count('irregular'),
            'elongated_irregular': shapes.count('elongated_irregular')
        }
        
        result = {
            'n_deposits': n_total,
            # Circularity statistics
            'mean_circularity': float(np.mean(circularity)),
            'std_circularity': float(np.std(circularity)),
            'median_circularity': float(np.median(circularity)),
            'circularity_cv': coefficient_of_variation(circularity),
            # Aspect ratio statistics
            'mean_aspect_ratio': float(np.mean(aspect_ratio)),
            'std_aspect_ratio': float(np.std(aspect_ratio)),
            'median_aspect_ratio': float(np.median(aspect_ratio)),
            'aspect_ratio_cv': coefficient_of_variation(aspect_ratio),
            # Shape classification counts
            'shape_counts': shape_counts,
            # Shape classification fractions
            'shape_fractions': {k: v / n_total for k, v in shape_counts.items()},
            # Regularity metrics
            'fraction_regular': (shape_counts['round'] + shape_counts['oval']) / n_total,
            'fraction_irregular': (shape_counts['irregular'] + shape_counts['elongated_irregular']) / n_total,
            # Elongation metrics
            'fraction_elongated': (shape_counts['elongated_oval'] + shape_counts['elongated_irregular']) / n_total,
            # Distribution shape
            'circularity_skewness': float(self.stats.skew(circularity)),
            'aspect_ratio_skewness': float(self.stats.skew(aspect_ratio)),
            # Thresholds used
            'thresholds': {
                'circularity_round': self.CIRCULARITY_ROUND,
                'circularity_oval': self.CIRCULARITY_OVAL,
                'aspect_ratio_square': self.ASPECT_RATIO_SQUARE,
                'aspect_ratio_elongated': self.ASPECT_RATIO_ELONGATED
            }
        }
        
        # By deposit type if available
        if 'label' in deposits_df.columns:
            for label in ['normal', 'rod']:
                label_df = deposits_df[deposits_df['label'] == label]
                if len(label_df) >= 2:
                    label_circ = label_df['circularity'].dropna().values
                    label_ar = label_df['aspect_ratio'].dropna().values
                    
                    result[f'{label}_mean_circularity'] = float(np.mean(label_circ))
                    result[f'{label}_std_circularity'] = float(np.std(label_circ))
                    result[f'{label}_mean_aspect_ratio'] = float(np.mean(label_ar))
                    result[f'{label}_std_aspect_ratio'] = float(np.std(label_ar))
                    
                    # Shape distribution for this type
                    label_shapes = [self._classify_shape(c, ar) 
                                    for c, ar in zip(label_circ, label_ar)]
                    result[f'{label}_fraction_regular'] = (
                        label_shapes.count('round') + label_shapes.count('oval')
                    ) / len(label_shapes) if label_shapes else 0
        
        return result
    
    def compare_morphology_normal_vs_rod(self, deposits_df: pd.DataFrame) -> Dict:
        """
        Compare morphology between Normal deposits and RODs.
        
        Args:
            deposits_df: DataFrame with deposit features
            
        Returns:
            Dict with Normal vs ROD morphology comparison
        """
        if 'label' not in deposits_df.columns:
            return {'error': 'label column not found'}
        
        required_cols = ['circularity', 'aspect_ratio']
        if not all(col in deposits_df.columns for col in required_cols):
            return {'error': f'Required columns not found: {required_cols}'}
        
        normal_df = deposits_df[deposits_df['label'] == 'normal']
        rod_df = deposits_df[deposits_df['label'] == 'rod']
        
        results = {}
        stat_analyzer = StatisticalAnalyzer(alpha=self.alpha)
        
        for metric in ['circularity', 'aspect_ratio']:
            normal_vals = normal_df[metric].dropna().values
            rod_vals = rod_df[metric].dropna().values
            
            if len(normal_vals) >= 2 and len(rod_vals) >= 2:
                comparison = stat_analyzer.compare_two_groups(
                    normal_vals, rod_vals,
                    group1_name='Normal',
                    group2_name='ROD'
                )
                
                results[metric] = {
                    'normal_mean': float(np.mean(normal_vals)),
                    'normal_std': float(np.std(normal_vals)),
                    'rod_mean': float(np.mean(rod_vals)),
                    'rod_std': float(np.std(rod_vals)),
                    'comparison': comparison
                }
        
        return results
    
    def compare_morphology_between_groups(
        self,
        film_summary: pd.DataFrame,
        group_column: str,
        metric_column: str = 'normal_mean_circularity'
    ) -> Dict:
        """Compare morphology metrics between experimental groups."""
        return _compare_metric_between_groups(
            film_summary, group_column, metric_column, self.alpha, include_median=False)
    
    def detect_morphological_outliers(
        self,
        deposits_df: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict:
        """
        Detect morphologically abnormal deposits.
        
        Args:
            deposits_df: DataFrame with deposit features
            method: 'iqr' (Interquartile Range) or 'zscore'
            threshold: IQR multiplier or z-score threshold
            
        Returns:
            Dict with outlier detection results
        """
        required_cols = ['circularity', 'aspect_ratio']
        if not all(col in deposits_df.columns for col in required_cols):
            return {'error': f'Required columns not found: {required_cols}'}
        
        # Filter valid deposits
        if 'label' in deposits_df.columns:
            valid_df = deposits_df[deposits_df['label'].isin(['normal', 'rod'])].copy()
        else:
            valid_df = deposits_df.copy()
        
        outlier_flags = {
            'low_circularity': np.zeros(len(valid_df), dtype=bool),
            'high_aspect_ratio': np.zeros(len(valid_df), dtype=bool)
        }
        
        for metric, flag_name, direction in [
            ('circularity', 'low_circularity', 'low'),
            ('aspect_ratio', 'high_aspect_ratio', 'high')
        ]:
            values = valid_df[metric].values
            
            if method == 'iqr':
                q1, q3 = np.nanpercentile(values, [25, 75])
                iqr = q3 - q1
                
                if direction == 'low':
                    outlier_flags[flag_name] = values < (q1 - threshold * iqr)
                else:
                    outlier_flags[flag_name] = values > (q3 + threshold * iqr)
            else:  # zscore
                mean, std = np.nanmean(values), np.nanstd(values)
                z_scores = (values - mean) / std if std > 0 else np.zeros_like(values)
                
                if direction == 'low':
                    outlier_flags[flag_name] = z_scores < -threshold
                else:
                    outlier_flags[flag_name] = z_scores > threshold
        
        # Any morphological outlier
        any_outlier = outlier_flags['low_circularity'] | outlier_flags['high_aspect_ratio']
        
        n_total = len(valid_df)
        
        return {
            'method': method,
            'threshold': threshold,
            'n_total': n_total,
            'n_low_circularity_outliers': int(np.sum(outlier_flags['low_circularity'])),
            'n_high_aspect_ratio_outliers': int(np.sum(outlier_flags['high_aspect_ratio'])),
            'n_any_morphology_outlier': int(np.sum(any_outlier)),
            'fraction_outliers': float(np.sum(any_outlier) / n_total) if n_total > 0 else 0,
            # Outlier indices (if needed for visualization)
            'outlier_indices': np.where(any_outlier)[0].tolist()
        }


def analyze_morphology(
    deposits_df: pd.DataFrame = None,
    film_summary: pd.DataFrame = None,
    group_column: str = None
) -> Dict:
    """
    Convenience function for morphology analysis.
    
    Args:
        deposits_df: Optional DataFrame with individual deposit data
        film_summary: Optional DataFrame with film-level summary
        group_column: Optional column for group comparisons
        
    Returns:
        Dict with morphology analysis results
    """
    analyzer = MorphologyAnalyzer()
    results = {}
    
    if deposits_df is not None:
        results['distribution'] = analyzer.analyze_morphology(deposits_df)
        results['normal_vs_rod'] = analyzer.compare_morphology_normal_vs_rod(deposits_df)
        results['outliers'] = analyzer.detect_morphological_outliers(deposits_df)
    
    if film_summary is not None and group_column:
        results['group_comparison_circularity'] = analyzer.compare_morphology_between_groups(
            film_summary, group_column, 'normal_mean_circularity'
        )
    
    return results


# =============================================================================
# Comprehensive Analysis (All-in-one)
# =============================================================================
