"""Correlation analysis between deposit metrics."""

import numpy as np
import pandas as pd
from typing import List, Dict


class CorrelationAnalyzer:
    """
    Analyze correlations between deposit features.
    
    Provides:
    - Feature correlation matrix
    - Key correlations (size-IOD, size-hue, etc.)
    - Correlation significance testing
    """
    
    def __init__(self, alpha: float = 0.05):
        from scipy import stats
        self.stats = stats
        self.alpha = alpha
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation strength."""
        abs_r = abs(r)
        if abs_r < 0.1:
            return 'negligible'
        elif abs_r < 0.3:
            return 'weak'
        elif abs_r < 0.5:
            return 'moderate'
        elif abs_r < 0.7:
            return 'strong'
        else:
            return 'very_strong'
    
    def calculate_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = 'pearson'
    ) -> Dict:
        """
        Calculate correlation between two variables.
        
        Args:
            x, y: Arrays of values
            method: 'pearson' or 'spearman'
            
        Returns:
            Dict with correlation results
        """
        x = np.array(x)
        y = np.array(y)
        
        # Remove NaN pairs
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x = x[valid_mask]
        y = y[valid_mask]
        
        if len(x) < 3:
            return {'error': 'Insufficient data', 'n': len(x)}
        
        if method == 'spearman':
            r, p = self.stats.spearmanr(x, y)
        else:
            r, p = self.stats.pearsonr(x, y)
        
        return {
            'method': method,
            'r': float(r),
            'r_squared': float(r ** 2),
            'p_value': float(p),
            'significant': p < self.alpha,
            'interpretation': self._interpret_correlation(r),
            'direction': 'positive' if r > 0 else 'negative',
            'n': len(x)
        }
    
    def correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        method: str = 'pearson'
    ) -> Dict:
        """
        Calculate correlation matrix for multiple features.
        
        Args:
            df: DataFrame with features
            columns: List of columns to include (default: all numeric)
            method: 'pearson' or 'spearman'
            
        Returns:
            Dict with correlation matrix and significance
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to existing columns
        columns = [c for c in columns if c in df.columns]
        
        if len(columns) < 2:
            return {'error': 'Need at least 2 numeric columns'}
        
        n_cols = len(columns)
        corr_matrix = np.zeros((n_cols, n_cols))
        p_matrix = np.zeros((n_cols, n_cols))
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                elif i < j:
                    result = self.calculate_correlation(
                        df[col1].values, df[col2].values, method
                    )
                    if 'error' not in result:
                        corr_matrix[i, j] = result['r']
                        corr_matrix[j, i] = result['r']
                        p_matrix[i, j] = result['p_value']
                        p_matrix[j, i] = result['p_value']
                    else:
                        corr_matrix[i, j] = np.nan
                        corr_matrix[j, i] = np.nan
                        p_matrix[i, j] = np.nan
                        p_matrix[j, i] = np.nan
        
        # Find strongest correlations
        strong_correlations = []
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                if not np.isnan(corr_matrix[i, j]) and abs(corr_matrix[i, j]) >= 0.3:
                    strong_correlations.append({
                        'feature1': columns[i],
                        'feature2': columns[j],
                        'r': float(corr_matrix[i, j]),
                        'p_value': float(p_matrix[i, j]),
                        'significant': p_matrix[i, j] < self.alpha,
                        'interpretation': self._interpret_correlation(corr_matrix[i, j])
                    })
        
        # Sort by absolute correlation
        strong_correlations.sort(key=lambda x: abs(x['r']), reverse=True)
        
        return {
            'method': method,
            'columns': columns,
            'correlation_matrix': corr_matrix.tolist(),
            'p_value_matrix': p_matrix.tolist(),
            'strong_correlations': strong_correlations[:10],  # Top 10
            'n_features': n_cols
        }
    
    def analyze_key_correlations(self, deposits_df: pd.DataFrame) -> Dict:
        """
        Analyze biologically relevant correlations.
        
        Key correlations:
        - Size (area) vs IOD: pigment amount scales with size?
        - Size vs Hue: larger deposits more/less acidic?
        - Size vs Circularity: larger deposits less circular?
        - IOD vs Hue: pigment intensity relates to pH?
        
        Args:
            deposits_df: DataFrame with deposit features
            
        Returns:
            Dict with key correlation results
        """
        # Filter valid deposits
        if 'label' in deposits_df.columns:
            valid_df = deposits_df[deposits_df['label'].isin(['normal', 'rod'])]
        else:
            valid_df = deposits_df
        
        key_pairs = [
            ('area_px', 'iod', 'size_vs_iod', 'Does pigment scale with size?'),
            ('area_px', 'mean_hue', 'size_vs_hue', 'Are larger deposits more acidic?'),
            ('area_px', 'circularity', 'size_vs_circularity', 'Do larger deposits have different shape?'),
            ('iod', 'mean_hue', 'iod_vs_hue', 'Does pigment intensity relate to pH?'),
            ('area_px', 'mean_lightness', 'size_vs_lightness', 'Are larger deposits darker?'),
            ('circularity', 'aspect_ratio', 'circularity_vs_aspect', 'Shape consistency check'),
        ]
        
        results = {}
        
        for col1, col2, key, description in key_pairs:
            if col1 in valid_df.columns and col2 in valid_df.columns:
                # Use Spearman for robustness
                corr = self.calculate_correlation(
                    valid_df[col1].values,
                    valid_df[col2].values,
                    method='spearman'
                )
                corr['description'] = description
                results[key] = corr
        
        # Pigment density correlation (IOD/Area vs other features)
        if 'area_px' in valid_df.columns and 'iod' in valid_df.columns:
            valid_df = valid_df.copy()
            valid_df['pigment_density'] = valid_df['iod'] / valid_df['area_px'].replace(0, np.nan)
            
            if 'mean_hue' in valid_df.columns:
                results['pigment_density_vs_hue'] = self.calculate_correlation(
                    valid_df['pigment_density'].values,
                    valid_df['mean_hue'].values,
                    method='spearman'
                )
                results['pigment_density_vs_hue']['description'] = 'Does pigment concentration relate to pH?'
        
        return results
    
    def analyze_correlations_by_type(self, deposits_df: pd.DataFrame) -> Dict:
        """
        Compare correlations between Normal deposits and RODs.
        
        Args:
            deposits_df: DataFrame with deposit features
            
        Returns:
            Dict with correlations by deposit type
        """
        if 'label' not in deposits_df.columns:
            return {'error': 'label column not found'}
        
        results = {}
        
        for label in ['normal', 'rod']:
            label_df = deposits_df[deposits_df['label'] == label]
            if len(label_df) >= 10:  # Need sufficient data
                results[label] = self.analyze_key_correlations(label_df)
        
        # Compare correlation strengths between types
        if 'normal' in results and 'rod' in results:
            comparison = {}
            for key in results['normal']:
                if key in results['rod']:
                    normal_r = results['normal'][key].get('r', np.nan)
                    rod_r = results['rod'][key].get('r', np.nan)
                    
                    if not np.isnan(normal_r) and not np.isnan(rod_r):
                        comparison[key] = {
                            'normal_r': normal_r,
                            'rod_r': rod_r,
                            'difference': rod_r - normal_r,
                            'same_direction': (normal_r > 0) == (rod_r > 0)
                        }
            
            results['comparison'] = comparison
        
        return results


def analyze_correlations(
    deposits_df: pd.DataFrame,
    feature_columns: List[str] = None
) -> Dict:
    """
    Convenience function for correlation analysis.
    
    Args:
        deposits_df: DataFrame with deposit features
        feature_columns: Optional list of columns for correlation matrix
        
    Returns:
        Dict with correlation analysis results
    """
    analyzer = CorrelationAnalyzer()
    
    # Default feature columns for correlation matrix
    if feature_columns is None:
        feature_columns = [
            'area_px', 'iod', 'mean_hue', 'mean_lightness', 'mean_saturation',
            'circularity', 'aspect_ratio', 'perimeter'
        ]
    
    return {
        'key_correlations': analyzer.analyze_key_correlations(deposits_df),
        'by_deposit_type': analyzer.analyze_correlations_by_type(deposits_df),
        'correlation_matrix': analyzer.correlation_matrix(deposits_df, feature_columns)
    }


# =============================================================================
# Morphology Analysis
# =============================================================================
