"""Deposit size-distribution analysis."""

import numpy as np
import pandas as pd
from typing import Dict

from .common import StatisticalAnalyzer, _compare_metric_between_groups, coefficient_of_variation


class SizeDistributionAnalyzer:
    """
    Analyze deposit size distribution patterns.
    
    Provides:
    - Size class distribution (Small/Medium/Large)
    - Size heterogeneity metrics
    - Normal vs ROD size comparison
    - Bimodality detection
    """
    
    # Default size thresholds (in pixels, can be adjusted)
    SIZE_SMALL_MAX = 100      # pixels²
    SIZE_MEDIUM_MAX = 500     # pixels²
    # Above SIZE_MEDIUM_MAX = Large
    
    def __init__(self, alpha: float = 0.05, size_thresholds: tuple = None):
        from scipy import stats
        self.stats = stats
        self.alpha = alpha
        
        if size_thresholds:
            self.SIZE_SMALL_MAX, self.SIZE_MEDIUM_MAX = size_thresholds
    
    def _classify_size(self, area: float) -> str:
        """Classify deposit by size."""
        if area < self.SIZE_SMALL_MAX:
            return 'small'
        elif area < self.SIZE_MEDIUM_MAX:
            return 'medium'
        else:
            return 'large'
    
    def _gini_coefficient(self, values: np.ndarray) -> float:
        """
        Calculate Gini coefficient for size inequality.
        0 = perfect equality, 1 = perfect inequality
        """
        values = np.sort(values)
        n = len(values)
        if n < 2 or np.sum(values) == 0:
            return np.nan
        
        cumsum = np.cumsum(values)
        return (2 * np.sum((np.arange(1, n + 1) * values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
    
    def _bimodality_coefficient(self, values: np.ndarray) -> float:
        """
        Calculate bimodality coefficient.
        BC > 0.555 suggests bimodal distribution.
        BC = (skewness² + 1) / (kurtosis + 3 * (n-1)² / ((n-2)(n-3)))
        """
        n = len(values)
        if n < 4:
            return np.nan
        
        skewness = self.stats.skew(values)
        kurtosis = self.stats.kurtosis(values, fisher=True)  # excess kurtosis
        
        # Adjusted kurtosis for sample size
        adjusted_kurtosis = kurtosis + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        
        bc = (skewness ** 2 + 1) / adjusted_kurtosis
        return float(bc)
    
    def analyze_size_distribution(self, deposits_df: pd.DataFrame, area_column: str = 'area_px') -> Dict:
        """
        Analyze size distribution of deposits.
        
        Args:
            deposits_df: DataFrame with deposit data
            area_column: Column name for area values
            
        Returns:
            Dict with size distribution analysis
        """
        if area_column not in deposits_df.columns:
            return {'error': f'{area_column} column not found'}
        
        # Filter valid deposits
        if 'label' in deposits_df.columns:
            valid_df = deposits_df[deposits_df['label'].isin(['normal', 'rod'])]
        else:
            valid_df = deposits_df
        
        areas = valid_df[area_column].dropna().values
        
        if len(areas) < 2:
            return {'error': 'Insufficient data'}
        
        # Size class distribution
        size_classes = [self._classify_size(a) for a in areas]
        n_total = len(size_classes)
        n_small = size_classes.count('small')
        n_medium = size_classes.count('medium')
        n_large = size_classes.count('large')

        bc = self._bimodality_coefficient(areas)   # compute once (was called twice)
        result = {
            'n_deposits': n_total,
            # Basic statistics
            'mean_area': float(np.mean(areas)),
            'std_area': float(np.std(areas)),
            'median_area': float(np.median(areas)),
            'min_area': float(np.min(areas)),
            'max_area': float(np.max(areas)),
            'area_cv': coefficient_of_variation(areas),
            # Size class counts
            'n_small': n_small,
            'n_medium': n_medium,
            'n_large': n_large,
            # Size class fractions
            'fraction_small': n_small / n_total,
            'fraction_medium': n_medium / n_total,
            'fraction_large': n_large / n_total,
            # Heterogeneity metrics
            'gini_coefficient': self._gini_coefficient(areas),
            'bimodality_coefficient': bc,
            'is_bimodal': (len(areas) >= 4 and bc > 0.555),
            # Distribution shape
            'skewness': float(self.stats.skew(areas)),
            'kurtosis': float(self.stats.kurtosis(areas)),
            # Percentiles
            'percentile_25': float(np.percentile(areas, 25)),
            'percentile_75': float(np.percentile(areas, 75)),
            'iqr': float(np.percentile(areas, 75) - np.percentile(areas, 25)),
            # Thresholds used
            'size_threshold_small': self.SIZE_SMALL_MAX,
            'size_threshold_medium': self.SIZE_MEDIUM_MAX
        }
        
        # By deposit type if available
        if 'label' in deposits_df.columns:
            for label in ['normal', 'rod']:
                label_areas = deposits_df[deposits_df['label'] == label][area_column].dropna().values
                if len(label_areas) >= 2:
                    result[f'{label}_n'] = len(label_areas)
                    result[f'{label}_mean_area'] = float(np.mean(label_areas))
                    result[f'{label}_std_area'] = float(np.std(label_areas))
                    result[f'{label}_median_area'] = float(np.median(label_areas))
                    result[f'{label}_area_cv'] = coefficient_of_variation(label_areas)
        
        return result
    
    def compare_size_normal_vs_rod(self, deposits_df: pd.DataFrame, area_column: str = 'area_px') -> Dict:
        """
        Compare size distribution between Normal deposits and RODs.
        
        Args:
            deposits_df: DataFrame with deposit data
            area_column: Column name for area values
            
        Returns:
            Dict with Normal vs ROD size comparison
        """
        if area_column not in deposits_df.columns or 'label' not in deposits_df.columns:
            return {'error': 'Required columns not found'}
        
        normal_areas = deposits_df[deposits_df['label'] == 'normal'][area_column].dropna().values
        rod_areas = deposits_df[deposits_df['label'] == 'rod'][area_column].dropna().values
        
        if len(normal_areas) < 2 or len(rod_areas) < 2:
            return {'error': 'Insufficient data for comparison'}
        
        # Statistical comparison
        stat_analyzer = StatisticalAnalyzer(alpha=self.alpha)
        comparison = stat_analyzer.compare_two_groups(
            normal_areas, rod_areas,
            group1_name='Normal',
            group2_name='ROD'
        )
        
        # Size ratio
        size_ratio = np.mean(rod_areas) / np.mean(normal_areas) if np.mean(normal_areas) > 0 else np.nan
        
        return {
            'normal_statistics': {
                'n': len(normal_areas),
                'mean': float(np.mean(normal_areas)),
                'std': float(np.std(normal_areas)),
                'median': float(np.median(normal_areas))
            },
            'rod_statistics': {
                'n': len(rod_areas),
                'mean': float(np.mean(rod_areas)),
                'std': float(np.std(rod_areas)),
                'median': float(np.median(rod_areas))
            },
            'rod_to_normal_ratio': float(size_ratio),
            'comparison': comparison
        }
    
    def compare_size_between_groups(
        self,
        film_summary: pd.DataFrame,
        group_column: str,
        size_column: str = 'normal_mean_area'
    ) -> Dict:
        """Compare deposit size between experimental groups."""
        return _compare_metric_between_groups(
            film_summary, group_column, size_column, self.alpha, include_median=False)


def analyze_size_distribution(
    deposits_df: pd.DataFrame = None,
    film_summary: pd.DataFrame = None,
    group_column: str = None,
    area_column: str = 'area_px'
) -> Dict:
    """
    Convenience function for size distribution analysis.
    
    Args:
        deposits_df: Optional DataFrame with individual deposit data
        film_summary: Optional DataFrame with film-level summary
        group_column: Optional column for group comparisons
        area_column: Column name for area values
        
    Returns:
        Dict with size distribution analysis results
    """
    analyzer = SizeDistributionAnalyzer()
    results = {}
    
    if deposits_df is not None:
        results['distribution'] = analyzer.analyze_size_distribution(deposits_df, area_column)
        results['normal_vs_rod'] = analyzer.compare_size_normal_vs_rod(deposits_df, area_column)
    
    if film_summary is not None and group_column:
        results['group_comparison'] = analyzer.compare_size_between_groups(
            film_summary, group_column
        )
    
    return results


# =============================================================================
# Density Analysis (Deposit count and coverage normalization)
# =============================================================================
