"""pH / Bromophenol-Blue indicator analysis (pHAnalyzer + analyze_ph)."""

import numpy as np
import pandas as pd
from typing import Dict

from .common import StatisticalAnalyzer, coefficient_of_variation, compare_group_values


class pHAnalyzer:
    """
    pH analysis based on BPB (Bromophenol Blue) color indicators.
    
    BPB color range:
        - pH < 3.0:    Yellow (Hue ~30-60°) → Acidic
        - pH 3.0-4.6:  Transition (Hue ~60-180°)
        - pH > 4.6:    Blue (Hue ~180-240°) → Basic
    """
    
    def __init__(self, alpha: float = 0.05):
        from scipy import stats
        from ..features import (
            estimate_ph_category, estimate_ph_value, calculate_acidity_index
        )
        self.stats = stats
        self.alpha = alpha
        # Store pH functions
        self.estimate_ph_category = estimate_ph_category
        self.estimate_ph_value = estimate_ph_value
        self.calculate_acidity_index = calculate_acidity_index
    
    def analyze_deposit_ph(self, deposits_df: pd.DataFrame) -> Dict:
        """
        Analyze pH distribution from individual deposit data.
        
        Args:
            deposits_df: DataFrame with 'mean_hue' column (individual deposits)
            
        Returns:
            Dict with pH analysis results
        """
        if 'mean_hue' not in deposits_df.columns:
            return {'error': 'mean_hue column not found'}
        
        hue_values = deposits_df['mean_hue'].dropna().values
        
        if len(hue_values) < 1:
            return {'error': 'No valid hue data'}
        
        # Calculate pH values and categories for each deposit
        ph_values = np.array([self.estimate_ph_value(h) for h in hue_values])
        acidity_indices = np.array([self.calculate_acidity_index(h) for h in hue_values])
        categories = [self.estimate_ph_category(h) for h in hue_values]
        
        # Category distribution
        n_total = len(categories)
        n_acidic = categories.count('acidic')
        n_transitional = categories.count('transitional')
        n_basic = categories.count('basic')
        
        return {
            'n_deposits': n_total,
            # pH estimation
            'mean_ph': float(np.mean(ph_values)),
            'std_ph': float(np.std(ph_values)),
            'median_ph': float(np.median(ph_values)),
            'min_ph': float(np.min(ph_values)),
            'max_ph': float(np.max(ph_values)),
            # Acidity index (0=basic, 1=acidic)
            'mean_acidity_index': float(np.mean(acidity_indices)),
            'std_acidity_index': float(np.std(acidity_indices)),
            # Category distribution
            'n_acidic': n_acidic,
            'n_transitional': n_transitional,
            'n_basic': n_basic,
            'fraction_acidic': n_acidic / n_total if n_total > 0 else 0,
            'fraction_transitional': n_transitional / n_total if n_total > 0 else 0,
            'fraction_basic': n_basic / n_total if n_total > 0 else 0,
            # Raw hue statistics
            'mean_hue': float(np.mean(hue_values)),
            'std_hue': float(np.std(hue_values)),
            # pH heterogeneity (CV of pH values)
            'ph_cv': coefficient_of_variation(ph_values)
        }
    
    def analyze_film_ph(self, film_summary: pd.DataFrame, hue_column: str = 'normal_mean_hue') -> Dict:
        """
        Analyze pH from film-level summary data.
        
        Args:
            film_summary: DataFrame with film-level hue averages
            hue_column: Column name containing mean hue values
            
        Returns:
            Dict with pH analysis results
        """
        if hue_column not in film_summary.columns:
            return {'error': f'{hue_column} column not found'}
        
        hue_values = film_summary[hue_column].dropna().values
        
        if len(hue_values) < 1:
            return {'error': 'No valid hue data'}
        
        ph_values = np.array([self.estimate_ph_value(h) for h in hue_values])
        acidity_indices = np.array([self.calculate_acidity_index(h) for h in hue_values])
        
        return {
            'n_films': len(hue_values),
            'mean_ph': float(np.mean(ph_values)),
            'std_ph': float(np.std(ph_values)),
            'median_ph': float(np.median(ph_values)),
            'mean_acidity_index': float(np.mean(acidity_indices)),
            'std_acidity_index': float(np.std(acidity_indices)),
            'mean_hue': float(np.mean(hue_values)),
            'std_hue': float(np.std(hue_values)),
            'ph_cv': coefficient_of_variation(ph_values)
        }
    
    def compare_ph_between_groups(
        self,
        film_summary: pd.DataFrame,
        group_column: str,
        hue_column: str = 'normal_mean_hue'
    ) -> Dict:
        """
        Compare pH between experimental groups.
        
        Args:
            film_summary: DataFrame with film-level data
            group_column: Column name for grouping
            hue_column: Column name containing mean hue values
            
        Returns:
            Dict with group comparison results
        """
        if hue_column not in film_summary.columns:
            return {'error': f'{hue_column} column not found'}
        
        if group_column not in film_summary.columns:
            return {'error': f'{group_column} column not found'}
        
        groups = [g for g in film_summary[group_column].unique() 
                  if g != 'ungrouped' and pd.notna(g)]
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for comparison'}
        
        # Calculate pH for each film
        film_summary = film_summary.copy()
        film_summary['_estimated_ph'] = film_summary[hue_column].apply(
            lambda h: self.estimate_ph_value(h) if pd.notna(h) else np.nan
        )
        film_summary['_acidity_index'] = film_summary[hue_column].apply(
            lambda h: self.calculate_acidity_index(h) if pd.notna(h) else np.nan
        )
        
        # Group statistics
        group_stats = {}
        for group in groups:
            group_data = film_summary[film_summary[group_column] == group]
            ph_values = group_data['_estimated_ph'].dropna().values
            acidity_values = group_data['_acidity_index'].dropna().values
            
            if len(ph_values) < 2:
                continue
            
            group_stats[group] = {
                'n': len(ph_values),
                'mean_ph': float(np.mean(ph_values)),
                'std_ph': float(np.std(ph_values)),
                'median_ph': float(np.median(ph_values)),
                'mean_acidity_index': float(np.mean(acidity_values)),
                'std_acidity_index': float(np.std(acidity_values))
            }
        
        valid_groups = list(group_stats.keys())
        
        if len(valid_groups) < 2:
            return {
                'error': 'Insufficient data in groups',
                'group_statistics': group_stats
            }
        
        # Statistical comparison on pH values — significance uses only groups with n>=3
        # (matches compare_two/multiple_groups); the descriptive group_statistics above
        # still lists smaller groups.
        group_ph_data = {}
        for g in valid_groups:
            v = film_summary[film_summary[group_column] == g]['_estimated_ph'].dropna().values
            if len(v) >= 3:
                group_ph_data[g] = v
        
        # Use StatisticalAnalyzer for comparison
        results = {
            'metric': 'estimated_ph',
            'group_statistics': group_stats,
            'n_groups': len(valid_groups)
        }
        results['comparison'] = compare_group_values(group_ph_data, self.alpha)
        
        return results
    
    def compare_ph_by_deposit_type(
        self,
        deposits_df: pd.DataFrame
    ) -> Dict:
        """
        Compare pH between Normal deposits and RODs.
        
        Args:
            deposits_df: DataFrame with individual deposit data
            
        Returns:
            Dict with Normal vs ROD pH comparison
        """
        if 'mean_hue' not in deposits_df.columns or 'label' not in deposits_df.columns:
            return {'error': 'Required columns not found'}
        
        normal_hue = deposits_df[deposits_df['label'] == 'normal']['mean_hue'].dropna().values
        rod_hue = deposits_df[deposits_df['label'] == 'rod']['mean_hue'].dropna().values
        
        if len(normal_hue) < 2 or len(rod_hue) < 2:
            return {'error': 'Insufficient data for comparison'}
        
        normal_ph = np.array([self.estimate_ph_value(h) for h in normal_hue])
        rod_ph = np.array([self.estimate_ph_value(h) for h in rod_hue])
        
        # Statistical comparison
        stat_analyzer = StatisticalAnalyzer(alpha=self.alpha)
        comparison = stat_analyzer.compare_two_groups(
            normal_ph, rod_ph,
            group1_name='Normal',
            group2_name='ROD'
        )
        
        return {
            'normal_statistics': {
                'n': len(normal_ph),
                'mean_ph': float(np.mean(normal_ph)),
                'std_ph': float(np.std(normal_ph)),
                'mean_hue': float(np.mean(normal_hue))
            },
            'rod_statistics': {
                'n': len(rod_ph),
                'mean_ph': float(np.mean(rod_ph)),
                'std_ph': float(np.std(rod_ph)),
                'mean_hue': float(np.mean(rod_hue))
            },
            'comparison': comparison
        }


def analyze_ph(
    deposits_df: pd.DataFrame = None,
    film_summary: pd.DataFrame = None,
    group_column: str = None
) -> Dict:
    """
    Convenience function for pH analysis.
    
    Args:
        deposits_df: Optional DataFrame with individual deposit data
        film_summary: Optional DataFrame with film-level summary
        group_column: Optional column for group comparisons
        
    Returns:
        Dict with pH analysis results
    """
    analyzer = pHAnalyzer()
    results = {}
    
    if deposits_df is not None:
        results['deposit_level'] = analyzer.analyze_deposit_ph(deposits_df)
        results['normal_vs_rod'] = analyzer.compare_ph_by_deposit_type(deposits_df)
    
    if film_summary is not None:
        results['film_level'] = analyzer.analyze_film_ph(film_summary)
        
        if group_column:
            results['group_comparison'] = analyzer.compare_ph_between_groups(
                film_summary, group_column
            )
    
    return results


# =============================================================================
# Pigmentation Analysis (IOD-based)
# =============================================================================
