"""Pigmentation (IOD / lightness) analysis."""

import numpy as np
import pandas as pd
from typing import Dict

from .common import _compare_metric_between_groups, coefficient_of_variation


class PigmentationAnalyzer:
    """
    Analyze pigmentation based on IOD (Integrated Optical Density).
    
    IOD = Area × (1 - Lightness) represents total pigment amount.
    Higher IOD = more pigment/darker deposit.
    """
    
    def __init__(self, alpha: float = 0.05):
        from scipy import stats
        self.stats = stats
        self.alpha = alpha
    
    def analyze_deposit_pigmentation(self, deposits_df: pd.DataFrame) -> Dict:
        """
        Analyze pigmentation at individual deposit level.
        
        Args:
            deposits_df: DataFrame with 'iod', 'area_px', 'mean_lightness' columns
            
        Returns:
            Dict with pigmentation analysis
        """
        required_cols = ['iod', 'area_px']
        if not all(col in deposits_df.columns for col in required_cols):
            return {'error': f'Required columns not found: {required_cols}'}
        
        # Filter valid deposits (exclude artifacts if label exists)
        if 'label' in deposits_df.columns:
            valid_df = deposits_df[deposits_df['label'].isin(['normal', 'rod'])]
        else:
            valid_df = deposits_df
        
        iod_values = valid_df['iod'].dropna().values

        if len(iod_values) < 2:
            return {'error': 'Insufficient data'}

        # Pigment density = IOD / Area. Drop NaNs jointly so iod and area stay
        # row-aligned (independent dropna() calls could misalign or mismatch length).
        pair = valid_df[['iod', 'area_px']].dropna()
        pair = pair[pair['area_px'] > 0]
        pigment_density = (pair['iod'] / pair['area_px']).values
        
        result = {
            'n_deposits': len(iod_values),
            # Total IOD statistics
            'total_iod': float(np.sum(iod_values)),
            'mean_iod': float(np.mean(iod_values)),
            'std_iod': float(np.std(iod_values)),
            'median_iod': float(np.median(iod_values)),
            'iod_cv': coefficient_of_variation(iod_values),
            # Pigment density (IOD per area)
            'mean_pigment_density': float(np.mean(pigment_density)),
            'std_pigment_density': float(np.std(pigment_density)),
            'pigment_density_cv': coefficient_of_variation(pigment_density),
        }
        
        # By deposit type if available
        if 'label' in deposits_df.columns:
            for label in ['normal', 'rod']:
                label_df = deposits_df[deposits_df['label'] == label]
                if len(label_df) >= 2:
                    label_iod = label_df['iod'].dropna().values
                    # Row-aligned pigment density (joint dropna, positive area only)
                    label_pair = label_df[['iod', 'area_px']].dropna()
                    label_pair = label_pair[label_pair['area_px'] > 0]
                    label_density = (label_pair['iod'] / label_pair['area_px']).values

                    result[f'{label}_total_iod'] = float(np.sum(label_iod))
                    result[f'{label}_mean_iod'] = float(np.mean(label_iod))
                    result[f'{label}_std_iod'] = float(np.std(label_iod))
                    result[f'{label}_mean_pigment_density'] = float(np.mean(label_density)) if len(label_density) > 0 else np.nan
        
        return result
    
    def analyze_film_pigmentation(
        self, 
        film_summary: pd.DataFrame,
        n_flies_column: str = None
    ) -> Dict:
        """
        Analyze pigmentation at film level.
        
        Args:
            film_summary: DataFrame with film-level IOD data
            n_flies_column: Optional column for per-fly normalization
            
        Returns:
            Dict with film-level pigmentation analysis
        """
        if 'total_iod' not in film_summary.columns:
            return {'error': 'total_iod column not found'}
        
        total_iod = film_summary['total_iod'].dropna().values
        
        if len(total_iod) < 1:
            return {'error': 'No valid IOD data'}
        
        result = {
            'n_films': len(total_iod),
            'mean_total_iod': float(np.mean(total_iod)),
            'std_total_iod': float(np.std(total_iod)),
            'median_total_iod': float(np.median(total_iod)),
            'total_iod_cv': coefficient_of_variation(total_iod)
        }
        
        # Per-fly normalization if n_flies available. Mask the FULL iod column jointly with
        # n_flies (the earlier dropna() shortened total_iod, so masking it with the full-length
        # n_flies>0 mask raised IndexError whenever total_iod contained any NaN).
        if n_flies_column and n_flies_column in film_summary.columns:
            iod_full = film_summary['total_iod'].values
            n_flies = film_summary[n_flies_column].values
            valid_mask = (n_flies > 0) & pd.notna(iod_full)
            if valid_mask.any():
                iod_per_fly = iod_full[valid_mask] / n_flies[valid_mask]
                result['mean_iod_per_fly'] = float(np.mean(iod_per_fly))
                result['std_iod_per_fly'] = float(np.std(iod_per_fly))
                result['iod_per_fly_cv'] = coefficient_of_variation(iod_per_fly)
        
        # Normal vs ROD IOD if available
        for col in ['normal_total_iod', 'rod_total_iod', 'normal_mean_iod', 'rod_mean_iod']:
            if col in film_summary.columns:
                values = film_summary[col].dropna().values
                if len(values) > 0:
                    result[f'mean_{col}'] = float(np.mean(values))
                    result[f'std_{col}'] = float(np.std(values))
        
        return result
    
    def compare_pigmentation_between_groups(
        self,
        film_summary: pd.DataFrame,
        group_column: str,
        iod_column: str = 'total_iod'
    ) -> Dict:
        """Compare pigmentation (IOD) between experimental groups."""
        return _compare_metric_between_groups(
            film_summary, group_column, iod_column, self.alpha, include_median=True)


def analyze_pigmentation(
    deposits_df: pd.DataFrame = None,
    film_summary: pd.DataFrame = None,
    group_column: str = None,
    n_flies_column: str = None
) -> Dict:
    """
    Convenience function for pigmentation analysis.
    
    Args:
        deposits_df: Optional DataFrame with individual deposit data
        film_summary: Optional DataFrame with film-level summary
        group_column: Optional column for group comparisons
        n_flies_column: Optional column for per-fly normalization
        
    Returns:
        Dict with pigmentation analysis results
    """
    analyzer = PigmentationAnalyzer()
    results = {}
    
    if deposits_df is not None:
        results['deposit_level'] = analyzer.analyze_deposit_pigmentation(deposits_df)
    
    if film_summary is not None:
        results['film_level'] = analyzer.analyze_film_pigmentation(film_summary, n_flies_column)
        
        if group_column:
            results['group_comparison'] = analyzer.compare_pigmentation_between_groups(
                film_summary, group_column
            )
    
    return results


# =============================================================================
# Size Distribution Analysis
# =============================================================================
