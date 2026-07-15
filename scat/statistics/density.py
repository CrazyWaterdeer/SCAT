"""Deposit density analysis."""

import numpy as np
import pandas as pd
from typing import Dict

from .common import _compare_metric_between_groups, coefficient_of_variation


class DensityAnalyzer:
    """
    Analyze deposit density and coverage.
    
    Provides:
    - Deposit density (count per area)
    - Coverage ratio (deposit area / image area)
    - Per-fly normalization
    - Activity indices
    """
    
    def __init__(self, alpha: float = 0.05):
        from scipy import stats
        self.stats = stats
        self.alpha = alpha
    
    def analyze_deposit_density(
        self, 
        deposits_df: pd.DataFrame,
        image_area_px: float = None,
        n_flies: int = None
    ) -> Dict:
        """
        Analyze deposit density from individual deposits.
        
        Args:
            deposits_df: DataFrame with deposit data
            image_area_px: Total image area in pixels (for density calculation)
            n_flies: Number of flies (for per-fly normalization)
            
        Returns:
            Dict with density analysis
        """
        # Filter valid deposits
        if 'label' in deposits_df.columns:
            valid_df = deposits_df[deposits_df['label'].isin(['normal', 'rod'])]
        else:
            valid_df = deposits_df
        
        n_total = len(valid_df)
        
        if n_total < 1:
            return {'error': 'No valid deposits'}
        
        result = {
            'n_total_deposits': n_total
        }
        
        # Count by type
        if 'label' in deposits_df.columns:
            result['n_normal'] = len(deposits_df[deposits_df['label'] == 'normal'])
            result['n_rod'] = len(deposits_df[deposits_df['label'] == 'rod'])
            result['n_artifact'] = len(deposits_df[deposits_df['label'] == 'artifact'])
            result['rod_fraction'] = result['n_rod'] / (result['n_normal'] + result['n_rod']) if (result['n_normal'] + result['n_rod']) > 0 else 0
        
        # Total deposit area
        if 'area_px' in deposits_df.columns:
            total_deposit_area = valid_df['area_px'].sum()
            result['total_deposit_area'] = float(total_deposit_area)
            result['mean_deposit_area'] = float(valid_df['area_px'].mean())
            
            # Coverage ratio if image area provided
            if image_area_px and image_area_px > 0:
                result['coverage_ratio'] = float(total_deposit_area / image_area_px)
                result['deposit_density'] = float(n_total / image_area_px * 1e6)  # per million pixels
        
        # Per-fly normalization
        if n_flies and n_flies > 0:
            result['deposits_per_fly'] = float(n_total / n_flies)
            if 'area_px' in deposits_df.columns:
                result['deposit_area_per_fly'] = float(total_deposit_area / n_flies)
            if 'iod' in valid_df.columns:
                result['iod_per_fly'] = float(valid_df['iod'].sum() / n_flies)
        
        return result
    
    def analyze_film_density(
        self,
        film_summary: pd.DataFrame,
        n_flies_column: str = None,
        image_width: int = None,
        image_height: int = None
    ) -> Dict:
        """
        Analyze density metrics at film level.
        
        Args:
            film_summary: DataFrame with film-level data
            n_flies_column: Column name for number of flies
            image_width: Image width in pixels (for density)
            image_height: Image height in pixels (for density)
            
        Returns:
            Dict with film-level density analysis
        """
        if 'n_total' not in film_summary.columns:
            return {'error': 'n_total column not found'}
        
        n_total = film_summary['n_total'].dropna().values
        
        if len(n_total) < 1:
            return {'error': 'No valid data'}
        
        result = {
            'n_films': len(n_total),
            'mean_deposits_per_film': float(np.mean(n_total)),
            'std_deposits_per_film': float(np.std(n_total)),
            'median_deposits_per_film': float(np.median(n_total)),
            'deposits_cv': coefficient_of_variation(n_total)
        }
        
        # Per-fly normalization
        if n_flies_column and n_flies_column in film_summary.columns:
            n_flies = film_summary[n_flies_column].values
            valid_mask = n_flies > 0
            
            if np.any(valid_mask):
                deposits_per_fly = n_total[valid_mask] / n_flies[valid_mask]
                result['mean_deposits_per_fly'] = float(np.mean(deposits_per_fly))
                result['std_deposits_per_fly'] = float(np.std(deposits_per_fly))
                result['deposits_per_fly_cv'] = coefficient_of_variation(deposits_per_fly)
        
        # ROD fraction statistics
        if 'rod_fraction' in film_summary.columns:
            rod_frac = film_summary['rod_fraction'].dropna().values
            if len(rod_frac) > 0:
                result['mean_rod_fraction'] = float(np.mean(rod_frac))
                result['std_rod_fraction'] = float(np.std(rod_frac))
                result['rod_fraction_cv'] = coefficient_of_variation(rod_frac)
        
        return result
    
    def compare_density_between_groups(
        self,
        film_summary: pd.DataFrame,
        group_column: str,
        density_metric: str = 'n_total'
    ) -> Dict:
        """Compare deposit density ('n_total', 'rod_fraction', ...) between experimental groups."""
        return _compare_metric_between_groups(
            film_summary, group_column, density_metric, self.alpha, include_median=True)
    
    def calculate_activity_index(
        self,
        film_summary: pd.DataFrame,
        n_flies_column: str = None
    ) -> Dict:
        """
        Calculate composite activity index combining multiple metrics.
        
        Activity Index = weighted combination of:
        - Deposit count (normalized)
        - Total IOD (normalized)
        - Coverage (if available)
        
        Args:
            film_summary: DataFrame with film-level data
            n_flies_column: Column for per-fly normalization
            
        Returns:
            Dict with activity indices
        """
        required_cols = ['n_total', 'total_iod']
        if not all(col in film_summary.columns for col in required_cols):
            return {'error': f'Required columns not found: {required_cols}'}
        
        # Get metrics
        n_total = film_summary['n_total'].values
        total_iod = film_summary['total_iod'].values
        
        # Normalize each metric to 0-1 range
        def normalize(arr):
            arr = np.array(arr, dtype=float)
            min_val, max_val = np.nanmin(arr), np.nanmax(arr)
            if max_val - min_val == 0:
                return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)
        
        norm_count = normalize(n_total)
        norm_iod = normalize(total_iod)
        
        # Simple average as activity index
        activity_index = (norm_count + norm_iod) / 2
        
        result = {
            'n_films': len(activity_index),
            'mean_activity_index': float(np.nanmean(activity_index)),
            'std_activity_index': float(np.nanstd(activity_index)),
            'activity_indices': activity_index.tolist()
        }
        
        # Per-fly if available
        if n_flies_column and n_flies_column in film_summary.columns:
            n_flies = film_summary[n_flies_column].values
            valid_mask = n_flies > 0
            
            if np.any(valid_mask):
                # Normalize per-fly metrics
                count_per_fly = n_total[valid_mask] / n_flies[valid_mask]
                iod_per_fly = total_iod[valid_mask] / n_flies[valid_mask]
                
                norm_count_pf = normalize(count_per_fly)
                norm_iod_pf = normalize(iod_per_fly)
                
                activity_per_fly = (norm_count_pf + norm_iod_pf) / 2
                
                result['mean_activity_index_per_fly'] = float(np.nanmean(activity_per_fly))
                result['std_activity_index_per_fly'] = float(np.nanstd(activity_per_fly))
        
        return result


def analyze_density(
    deposits_df: pd.DataFrame = None,
    film_summary: pd.DataFrame = None,
    group_column: str = None,
    n_flies_column: str = None,
    image_area_px: float = None
) -> Dict:
    """
    Convenience function for density analysis.
    
    Args:
        deposits_df: Optional DataFrame with individual deposit data
        film_summary: Optional DataFrame with film-level summary
        group_column: Optional column for group comparisons
        n_flies_column: Optional column for per-fly normalization
        image_area_px: Optional image area for density calculations
        
    Returns:
        Dict with density analysis results
    """
    analyzer = DensityAnalyzer()
    results = {}
    
    if deposits_df is not None:
        n_flies = None
        if film_summary is not None and n_flies_column and n_flies_column in film_summary.columns:
            # Use first film's n_flies as approximation
            n_flies = film_summary[n_flies_column].iloc[0] if len(film_summary) > 0 else None
        
        results['deposit_level'] = analyzer.analyze_deposit_density(
            deposits_df, image_area_px, n_flies
        )
    
    if film_summary is not None:
        results['film_level'] = analyzer.analyze_film_density(film_summary, n_flies_column)
        results['activity_index'] = analyzer.calculate_activity_index(film_summary, n_flies_column)
        
        if group_column:
            results['group_comparison_count'] = analyzer.compare_density_between_groups(
                film_summary, group_column, 'n_total'
            )
            results['group_comparison_rod_fraction'] = analyzer.compare_density_between_groups(
                film_summary, group_column, 'rod_fraction'
            )
    
    return results


# =============================================================================
# Correlation Analysis
# =============================================================================
