"""
Statistical analysis module for SCAT.
Provides group comparisons, normality tests, and effect sizes.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from itertools import combinations

# scipy is imported lazily when needed


class StatisticalAnalyzer:
    """Statistical analysis for excreta data."""
    
    def __init__(self, alpha: float = 0.05):
        from scipy import stats
        self.stats = stats  # Store for use in methods
        self.alpha = alpha
    
    def normality_test(self, data: np.ndarray, method: str = 'shapiro') -> Dict:
        """
        Test for normality.
        
        Args:
            data: Array of values
            method: 'shapiro' or 'jarque_bera'
            
        Returns:
            Dict with statistic, p-value, and is_normal
        """
        data = np.array(data)
        data = data[~np.isnan(data)]
        
        if len(data) < 3:
            return {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False, 'method': method}
        
        if method == 'shapiro':
            # Shapiro-Wilk (better for small samples)
            stat, p = self.stats.shapiro(data[:5000])  # Limit for large samples
        else:
            # Jarque-Bera (better for large samples)
            stat, p = self.stats.jarque_bera(data)
        
        return {
            'statistic': float(stat),
            'p_value': float(p),
            'is_normal': p > self.alpha,
            'method': method
        }
    
    def compare_two_groups(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray,
        group1_name: str = 'Group1',
        group2_name: str = 'Group2',
        paired: bool = False
    ) -> Dict:
        """
        Compare two groups with appropriate test.
        
        Automatically selects t-test or Mann-Whitney U based on normality.
        """
        group1 = np.array(group1)
        group2 = np.array(group2)
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        
        if len(group1) < 3 or len(group2) < 3:
            return {'error': 'Insufficient samples', 'n1': len(group1), 'n2': len(group2)}
        
        # Test normality
        norm1 = self.normality_test(group1)
        norm2 = self.normality_test(group2)
        both_normal = norm1['is_normal'] and norm2['is_normal']
        
        # Select and run test
        if both_normal:
            if paired and len(group1) == len(group2):
                stat, p = self.stats.ttest_rel(group1, group2)
                test_name = 'Paired t-test'
            else:
                stat, p = self.stats.ttest_ind(group1, group2)
                test_name = 'Independent t-test'
        else:
            if paired and len(group1) == len(group2):
                stat, p = self.stats.wilcoxon(group1, group2)
                test_name = 'Wilcoxon signed-rank'
            else:
                stat, p = self.stats.mannwhitneyu(group1, group2, alternative='two-sided')
                test_name = 'Mann-Whitney U'
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        # Interpret effect size
        d_abs = abs(cohens_d)
        if d_abs < 0.2:
            effect_interpretation = 'negligible'
        elif d_abs < 0.5:
            effect_interpretation = 'small'
        elif d_abs < 0.8:
            effect_interpretation = 'medium'
        else:
            effect_interpretation = 'large'
        
        return {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'n1': len(group1),
            'n2': len(group2),
            'mean1': float(np.mean(group1)),
            'mean2': float(np.mean(group2)),
            'std1': float(np.std(group1)),
            'std2': float(np.std(group2)),
            'test_name': test_name,
            'statistic': float(stat),
            'p_value': float(p),
            'significant': p < self.alpha,
            'cohens_d': float(cohens_d),
            'effect_size': effect_interpretation,
            'normality_group1': norm1['is_normal'],
            'normality_group2': norm2['is_normal']
        }
    
    def compare_multiple_groups(
        self,
        groups: Dict[str, np.ndarray],
        correction: str = 'holm'
    ) -> Dict:
        """
        Compare multiple groups with correction for multiple comparisons.
        
        For exactly 2 groups: Uses direct two-sample test (t-test or Mann-Whitney)
        without ANOVA overhead or multiple comparison correction.
        
        For 3+ groups: Uses ANOVA/Kruskal-Wallis followed by pairwise comparisons
        with multiple comparison correction.
        
        Args:
            groups: Dict mapping group names to data arrays
            correction: 'holm', 'bonferroni', or 'none' (only applies for 3+ groups)
        """
        group_names = list(groups.keys())
        group_data = [np.array(groups[name]) for name in group_names]
        
        # Filter out empty groups and NaN values
        valid_groups = {}
        for name, data in zip(group_names, group_data):
            clean_data = data[~np.isnan(data)] if len(data) > 0 else np.array([])
            if len(clean_data) >= 2:  # Need at least 2 samples per group
                valid_groups[name] = clean_data
        
        if len(valid_groups) < 2:
            return {
                'error': 'Insufficient valid groups (need at least 2 groups with 2+ samples each)',
                'n_groups': len(valid_groups),
                'group_names': list(valid_groups.keys())
            }
        
        group_names = list(valid_groups.keys())
        group_data = list(valid_groups.values())
        
        # Special case: exactly 2 groups - use direct comparison without ANOVA
        if len(valid_groups) == 2:
            name1, name2 = group_names
            result = self.compare_two_groups(
                valid_groups[name1], valid_groups[name2],
                group1_name=name1, group2_name=name2
            )
            result['comparison_type'] = 'two_group_direct'
            
            # Build group statistics
            group_stats = {}
            for name in group_names:
                data = valid_groups[name]
                group_stats[name] = {
                    'n': len(data),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'cv': self._coefficient_of_variation(data),
                    'median': float(np.median(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'ci_95': self._confidence_interval(data)
                }
            
            return {
                'overall_test': result.get('test_name'),
                'overall_statistic': result.get('statistic'),
                'overall_p_value': result.get('p_value'),
                'overall_significant': result.get('significant'),
                'n_groups': 2,
                'group_names': group_names,
                'group_statistics': group_stats,
                'correction_method': 'none',  # No correction needed for 2 groups
                'pairwise_comparisons': [result]
            }
        
        # 3+ groups: Use ANOVA/Kruskal-Wallis with post-hoc pairwise comparisons
        normality_results = [self.normality_test(g) for g in group_data]
        all_normal = all(r['is_normal'] for r in normality_results)
        
        try:
            if all_normal:
                stat, p = self.stats.f_oneway(*group_data)
                overall_test = 'One-way ANOVA'
            else:
                stat, p = self.stats.kruskal(*group_data)
                overall_test = 'Kruskal-Wallis H'
        except Exception as e:
            return {
                'error': f'Statistical test failed: {str(e)}',
                'n_groups': len(valid_groups),
                'group_names': group_names
            }
        
        # Post-hoc pairwise comparisons (all pairs)
        pairwise = []
        pairs = list(combinations(group_names, 2))
        for name1, name2 in pairs:
            result = self.compare_two_groups(
                valid_groups[name1], valid_groups[name2],
                group1_name=name1, group2_name=name2
            )
            result['comparison_type'] = 'pairwise'
            pairwise.append(result)
        
        # Apply multiple comparison correction (required for 3+ groups)
        if correction != 'none' and pairwise:
            valid_results = [r for r in pairwise if 'p_value' in r]
            
            if valid_results:
                p_values = [r['p_value'] for r in valid_results]
                corrected_p = self._correct_pvalues(p_values, correction)
                
                for result, p_corr in zip(valid_results, corrected_p):
                    result['p_value_corrected'] = p_corr
                    result['significant_corrected'] = p_corr < self.alpha
        
        # Calculate group statistics summary
        group_stats = {}
        for name in group_names:
            data = valid_groups[name]
            group_stats[name] = {
                'n': len(data),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'cv': self._coefficient_of_variation(data),
                'median': float(np.median(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'ci_95': self._confidence_interval(data)
            }
        
        return {
            'overall_test': overall_test,
            'overall_statistic': float(stat),
            'overall_p_value': float(p),
            'overall_significant': p < self.alpha,
            'n_groups': len(valid_groups),
            'group_names': group_names,
            'group_statistics': group_stats,
            'correction_method': correction,
            'pairwise_comparisons': pairwise
        }
    
    def _confidence_interval(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean."""
        data = np.array(data)
        data = data[~np.isnan(data)]
        
        if len(data) < 2:
            return (np.nan, np.nan)
        
        n = len(data)
        mean = np.mean(data)
        se = self.stats.sem(data)
        
        # t-value for confidence interval
        t_val = self.stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_val * se
        
        return (float(mean - margin), float(mean + margin))
    
    def _coefficient_of_variation(self, data: np.ndarray) -> float:
        """Calculate coefficient of variation (CV) as percentage.
        
        CV = (std / mean) * 100
        Lower CV indicates more consistent/reproducible measurements.
        """
        data = np.array(data)
        data = data[~np.isnan(data)]
        
        if len(data) < 2:
            return np.nan
        
        mean = np.mean(data)
        if mean == 0:
            return np.nan
        
        return float((np.std(data) / abs(mean)) * 100)
    
    def _correct_pvalues(self, p_values: List[float], method: str) -> List[float]:
        """Apply multiple comparison correction."""
        p_values = np.array(p_values)
        n = len(p_values)
        
        if method == 'bonferroni':
            return list(np.minimum(p_values * n, 1.0))
        
        elif method == 'holm':
            # Holm-Bonferroni step-down
            sorted_indices = np.argsort(p_values)
            corrected = np.zeros(n)
            
            for rank, idx in enumerate(sorted_indices):
                corrected[idx] = p_values[idx] * (n - rank)
            
            # Enforce monotonicity
            corrected = np.minimum.accumulate(corrected[np.argsort(sorted_indices)][::-1])[::-1]
            corrected = np.minimum(corrected, 1.0)
            return list(corrected)
        
        return list(p_values)
    
    def run_all_tests(
        self, 
        film_summary: pd.DataFrame, 
        group_by: str = None,
        metrics: List[str] = None
    ) -> Dict:
        """
        Run comprehensive statistical analysis.
        
        For 0 or 1 group: Returns descriptive statistics only.
        For 2 groups: Direct two-sample comparison.
        For 3+ groups: ANOVA/Kruskal-Wallis with post-hoc comparisons.
        
        Args:
            film_summary: DataFrame with image-level data
            group_by: Column name for grouping
            metrics: List of metrics to analyze
            
        Returns:
            Dict with all statistical results
        """
        if metrics is None:
            metrics = [
                # Count & fraction
                'rod_fraction', 'n_total', 'n_rod', 'n_normal',
                # IOD (Integrated Optical Density)
                'total_iod', 'normal_total_iod', 'rod_total_iod',
                'normal_mean_iod', 'rod_mean_iod',
                # Area (size)
                'normal_mean_area', 'rod_mean_area',
                # Color (Hue for pH estimation, Lightness for pigment density)
                'normal_mean_hue', 'rod_mean_hue',
                'normal_mean_lightness', 'rod_mean_lightness',
                # Shape (morphology)
                'normal_mean_circularity', 'rod_mean_circularity',
            ]
        
        results = {
            'metrics': {},
            'summary': {}
        }
        
        # Handle case where no grouping is specified or column doesn't exist
        if group_by is None or group_by not in film_summary.columns:
            # Provide descriptive statistics for all data (no grouping)
            results['descriptive_only'] = True
            results['n_groups'] = 0
            
            for metric in metrics:
                if metric not in film_summary.columns:
                    continue
                data = film_summary[metric].dropna().values
                if len(data) >= 2:
                    results['metrics'][metric] = {
                        'descriptive_only': True,
                        'overall_statistics': {
                            'n': len(data),
                            'mean': float(np.mean(data)),
                            'std': float(np.std(data)),
                            'cv': self._coefficient_of_variation(data),
                            'median': float(np.median(data)),
                            'min': float(np.min(data)),
                            'max': float(np.max(data)),
                            'ci_95': self._confidence_interval(data),
                            'normality': self.normality_test(data)
                        }
                    }
            return results
        
        # Get unique groups
        groups = [g for g in film_summary[group_by].unique() if g != 'ungrouped' and pd.notna(g)]
        
        # Handle case with 0 or 1 group
        if len(groups) < 2:
            results['descriptive_only'] = True
            results['n_groups'] = len(groups)
            results['group_names'] = groups
            
            for metric in metrics:
                if metric not in film_summary.columns:
                    continue
                
                # Get all data for this metric
                data = film_summary[metric].dropna().values
                if len(data) < 2:
                    continue
                
                metric_result = {
                    'descriptive_only': True,
                    'overall_statistics': {
                        'n': len(data),
                        'mean': float(np.mean(data)),
                        'std': float(np.std(data)),
                        'cv': self._coefficient_of_variation(data),
                        'median': float(np.median(data)),
                        'min': float(np.min(data)),
                        'max': float(np.max(data)),
                        'ci_95': self._confidence_interval(data),
                        'normality': self.normality_test(data)
                    }
                }
                
                # If there's exactly 1 group, also provide group-level stats
                if len(groups) == 1:
                    group_name = groups[0]
                    group_data = film_summary[film_summary[group_by] == group_name][metric].dropna().values
                    if len(group_data) >= 2:
                        metric_result['group_statistics'] = {
                            group_name: {
                                'n': len(group_data),
                                'mean': float(np.mean(group_data)),
                                'std': float(np.std(group_data)),
                                'cv': self._coefficient_of_variation(group_data),
                                'median': float(np.median(group_data)),
                                'min': float(np.min(group_data)),
                                'max': float(np.max(group_data)),
                                'ci_95': self._confidence_interval(group_data),
                                'normality': self.normality_test(group_data)
                            }
                        }
                
                results['metrics'][metric] = metric_result
            
            return results
        
        # 2+ groups: proceed with group comparisons
        results['n_groups'] = len(groups)
        
        for metric in metrics:
            if metric not in film_summary.columns:
                continue
            
            group_data = {
                g: film_summary[film_summary[group_by] == g][metric].dropna().values
                for g in groups
            }
            
            # Filter groups with insufficient data
            group_data = {k: v for k, v in group_data.items() if len(v) >= 2}
            
            if len(group_data) < 2:
                continue
            
            try:
                # Multi-group comparison
                results['metrics'][metric] = self.compare_multiple_groups(
                    group_data, 
                    correction='holm'
                )
                    
            except Exception as e:
                results['metrics'][metric] = {'error': str(e)}
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate a human-readable summary of statistical results."""
        summary = {
            'significant_metrics': [],
            'large_effects': [],
            'recommendations': []
        }
        
        for metric, data in results.get('metrics', {}).items():
            if 'error' in data:
                continue
            
            if data.get('overall_significant'):
                summary['significant_metrics'].append({
                    'metric': metric,
                    'test': data.get('overall_test'),
                    'p_value': data.get('overall_p_value')
                })
            
            # Check for large effects in pairwise comparisons
            for pw in data.get('pairwise_comparisons', []):
                if pw.get('effect_size') == 'large':
                    summary['large_effects'].append({
                        'metric': metric,
                        'group1': pw.get('group1_name'),
                        'group2': pw.get('group2_name'),
                        'cohens_d': pw.get('cohens_d')
                    })
        
        # Add recommendations
        n_groups = results.get('n_groups', 0)
        if n_groups >= 3:
            summary['recommendations'].append(
                'Multiple groups detected. Use corrected p-values for pairwise comparisons.'
            )
        
        return summary


def analyze_groups(
    film_summary: pd.DataFrame,
    group_column: str,
    metrics: List[str] = None
) -> Dict:
    """
    Analyze groups in film summary data.
    
    Args:
        film_summary: DataFrame with image-level statistics
        group_column: Column name for grouping
        metrics: List of metric columns to analyze
        
    Returns:
        Dict with statistical results
    """
    analyzer = StatisticalAnalyzer()
    return analyzer.run_all_tests(film_summary, group_by=group_column, metrics=metrics)


def generate_statistics_report(
    film_summary: pd.DataFrame,
    group_column: str,
    metrics: List[str] = None
) -> Dict:
    """
    Generate comprehensive statistics report for grouped data.
    
    Args:
        film_summary: DataFrame with film-level summary
        group_column: Column name for grouping (e.g., 'condition', 'group')
        metrics: List of metric columns to analyze
        
    Returns:
        Dict with statistical results
    """
    if metrics is None:
        metrics = ['rod_fraction', 'n_total', 'n_rod', 'n_normal',
                   'total_iod', 'normal_mean_area', 'rod_mean_area']
    
    analyzer = StatisticalAnalyzer()
    results = {}
    
    # Get groups, excluding 'ungrouped'
    groups = [g for g in film_summary[group_column].unique() 
              if g != 'ungrouped' and pd.notna(g)]
    
    if len(groups) < 2:
        return {}  # Not enough groups for comparison
    
    for metric in metrics:
        if metric not in film_summary.columns:
            continue
        
        group_data = {
            g: film_summary[film_summary[group_column] == g][metric].dropna().values
            for g in groups
        }
        
        # Filter groups with insufficient data
        group_data = {k: v for k, v in group_data.items() if len(v) >= 2}
        
        if len(group_data) < 2:
            continue
        
        try:
            if len(group_data) == 2:
                names = list(group_data.keys())
                results[metric] = analyzer.compare_two_groups(
                    group_data[names[0]], group_data[names[1]],
                    group1_name=names[0], group2_name=names[1]
                )
            else:
                results[metric] = analyzer.compare_multiple_groups(group_data)
        except Exception as e:
            results[metric] = {'error': str(e)}
    
    return results


class SpatialStatistics:
    """Statistical analysis for spatial data."""
    
    def __init__(self, alpha: float = 0.05):
        from scipy import stats
        self.stats = stats
        self.alpha = alpha
    
    def compare_clustering(
        self,
        group1_r_values: List[float],
        group2_r_values: List[float],
        group1_name: str = "Group1",
        group2_name: str = "Group2"
    ) -> Dict:
        """
        Compare Clark-Evans R values between two groups.
        Tests whether spatial clustering differs between conditions.
        """
        g1 = np.array(group1_r_values)
        g2 = np.array(group2_r_values)
        
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]
        
        if len(g1) < 3 or len(g2) < 3:
            return {'error': 'Insufficient samples'}
        
        # Mann-Whitney U (non-parametric, R often not normal)
        stat, p = self.stats.mannwhitneyu(g1, g2, alternative='two-sided')
        
        return {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_mean_r': float(np.mean(g1)),
            'group2_mean_r': float(np.mean(g2)),
            'group1_n': len(g1),
            'group2_n': len(g2),
            'test_name': 'Mann-Whitney U',
            'statistic': float(stat),
            'p_value': float(p),
            'significant': p < self.alpha,
            'interpretation': self._interpret_r_difference(np.mean(g1), np.mean(g2))
        }
    
    def _interpret_r_difference(self, r1: float, r2: float) -> str:
        """Interpret difference in Clark-Evans R."""
        diff = r1 - r2
        if abs(diff) < 0.1:
            return 'similar_clustering'
        elif diff > 0:
            return 'group1_more_dispersed'
        else:
            return 'group2_more_dispersed'
    
    def compare_edge_preference(
        self,
        group1_edge_fractions: List[float],
        group2_edge_fractions: List[float],
        group1_name: str = "Group1",
        group2_name: str = "Group2"
    ) -> Dict:
        """
        Compare edge vs center preference between groups.
        """
        g1 = np.array(group1_edge_fractions)
        g2 = np.array(group2_edge_fractions)
        
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]
        
        if len(g1) < 3 or len(g2) < 3:
            return {'error': 'Insufficient samples'}
        
        stat, p = self.stats.mannwhitneyu(g1, g2, alternative='two-sided')
        
        return {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_mean_edge_frac': float(np.mean(g1)),
            'group2_mean_edge_frac': float(np.mean(g2)),
            'test_name': 'Mann-Whitney U',
            'statistic': float(stat),
            'p_value': float(p),
            'significant': p < self.alpha
        }
    
    def compare_nnd(
        self,
        group1_nnd: List[float],
        group2_nnd: List[float],
        group1_name: str = "Group1",
        group2_name: str = "Group2"
    ) -> Dict:
        """
        Compare Nearest Neighbor Distances between groups.
        """
        g1 = np.array(group1_nnd)
        g2 = np.array(group2_nnd)
        
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]
        
        if len(g1) < 3 or len(g2) < 3:
            return {'error': 'Insufficient samples'}
        
        # Test normality
        _, p1 = self.stats.shapiro(g1[:5000])
        _, p2 = self.stats.shapiro(g2[:5000])
        
        if p1 > 0.05 and p2 > 0.05:
            stat, p = self.stats.ttest_ind(g1, g2)
            test_name = 't-test'
        else:
            stat, p = self.stats.mannwhitneyu(g1, g2, alternative='two-sided')
            test_name = 'Mann-Whitney U'
        
        # Effect size
        pooled_std = np.sqrt((np.var(g1) + np.var(g2)) / 2)
        cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
        
        return {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_mean_nnd': float(np.mean(g1)),
            'group2_mean_nnd': float(np.mean(g2)),
            'group1_std_nnd': float(np.std(g1)),
            'group2_std_nnd': float(np.std(g2)),
            'test_name': test_name,
            'statistic': float(stat),
            'p_value': float(p),
            'significant': p < self.alpha,
            'cohens_d': float(cohens_d)
        }


def generate_spatial_statistics_report(
    spatial_results_by_group: Dict[str, List],  # group_name -> List[SpatialResult]
) -> Dict:
    """
    Generate statistical comparisons of spatial metrics between groups.
    
    Args:
        spatial_results_by_group: Dict mapping group names to lists of SpatialResult
        
    Returns:
        Dict with statistical comparison results
    """
    stats = SpatialStatistics()
    results = {}
    
    group_names = list(spatial_results_by_group.keys())
    
    if len(group_names) < 2:
        return {'error': 'Need at least 2 groups for comparison'}
    
    # For simplicity, compare first two groups
    # Could extend to multiple comparisons if needed
    g1_name, g2_name = group_names[0], group_names[1]
    g1_results = spatial_results_by_group[g1_name]
    g2_results = spatial_results_by_group[g2_name]
    
    # Clark-Evans R comparison
    g1_r = [r.clark_evans_r for r in g1_results if r.clark_evans_r > 0]
    g2_r = [r.clark_evans_r for r in g2_results if r.clark_evans_r > 0]
    results['clark_evans'] = stats.compare_clustering(g1_r, g2_r, g1_name, g2_name)
    
    # Edge fraction comparison
    g1_edge = [r.edge_fraction for r in g1_results]
    g2_edge = [r.edge_fraction for r in g2_results]
    results['edge_fraction'] = stats.compare_edge_preference(g1_edge, g2_edge, g1_name, g2_name)
    
    # NND comparison
    g1_nnd = [r.nnd_mean for r in g1_results if r.nnd_mean > 0]
    g2_nnd = [r.nnd_mean for r in g2_results if r.nnd_mean > 0]
    results['nnd'] = stats.compare_nnd(g1_nnd, g2_nnd, g1_name, g2_name)
    
    return results


# =============================================================================
# pH Analysis
# =============================================================================

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
        from .features import (
            estimate_ph_category, estimate_ph_value, calculate_acidity_index,
            PH_HUE_ACIDIC_MAX, PH_HUE_BASIC_MIN, PH_MIN, PH_MAX
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
            'ph_cv': float(np.std(ph_values) / np.mean(ph_values) * 100) if np.mean(ph_values) > 0 else np.nan
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
            'ph_cv': float(np.std(ph_values) / np.mean(ph_values) * 100) if np.mean(ph_values) > 0 else np.nan
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
        
        # Statistical comparison on pH values
        group_ph_data = {
            g: film_summary[film_summary[group_column] == g]['_estimated_ph'].dropna().values
            for g in valid_groups
        }
        
        # Use StatisticalAnalyzer for comparison
        stat_analyzer = StatisticalAnalyzer(alpha=self.alpha)
        
        results = {
            'metric': 'estimated_ph',
            'group_statistics': group_stats,
            'n_groups': len(valid_groups)
        }
        
        if len(valid_groups) == 2:
            # Two-group comparison
            names = valid_groups
            comparison = stat_analyzer.compare_two_groups(
                group_ph_data[names[0]], 
                group_ph_data[names[1]],
                group1_name=names[0],
                group2_name=names[1]
            )
            results['comparison'] = comparison
        else:
            # Multiple group comparison
            comparison = stat_analyzer.compare_multiple_groups(
                group_ph_data,
                correction='holm'
            )
            results['comparison'] = comparison
        
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
        area_values = valid_df['area_px'].dropna().values
        
        if len(iod_values) < 2:
            return {'error': 'Insufficient data'}
        
        # Pigment density = IOD / Area
        valid_mask = area_values > 0
        pigment_density = iod_values[valid_mask] / area_values[valid_mask]
        
        result = {
            'n_deposits': len(iod_values),
            # Total IOD statistics
            'total_iod': float(np.sum(iod_values)),
            'mean_iod': float(np.mean(iod_values)),
            'std_iod': float(np.std(iod_values)),
            'median_iod': float(np.median(iod_values)),
            'iod_cv': float(np.std(iod_values) / np.mean(iod_values) * 100) if np.mean(iod_values) > 0 else np.nan,
            # Pigment density (IOD per area)
            'mean_pigment_density': float(np.mean(pigment_density)),
            'std_pigment_density': float(np.std(pigment_density)),
            'pigment_density_cv': float(np.std(pigment_density) / np.mean(pigment_density) * 100) if np.mean(pigment_density) > 0 else np.nan,
        }
        
        # By deposit type if available
        if 'label' in deposits_df.columns:
            for label in ['normal', 'rod']:
                label_df = deposits_df[deposits_df['label'] == label]
                if len(label_df) >= 2:
                    label_iod = label_df['iod'].dropna().values
                    label_area = label_df['area_px'].dropna().values
                    valid_mask = label_area > 0
                    label_density = label_iod[valid_mask] / label_area[valid_mask] if len(label_iod[valid_mask]) > 0 else []
                    
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
            'total_iod_cv': float(np.std(total_iod) / np.mean(total_iod) * 100) if np.mean(total_iod) > 0 else np.nan
        }
        
        # Per-fly normalization if n_flies available
        if n_flies_column and n_flies_column in film_summary.columns:
            n_flies = film_summary[n_flies_column].values
            valid_mask = n_flies > 0
            iod_per_fly = total_iod[valid_mask] / n_flies[valid_mask]
            
            result['mean_iod_per_fly'] = float(np.mean(iod_per_fly))
            result['std_iod_per_fly'] = float(np.std(iod_per_fly))
            result['iod_per_fly_cv'] = float(np.std(iod_per_fly) / np.mean(iod_per_fly) * 100) if np.mean(iod_per_fly) > 0 else np.nan
        
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
        """
        Compare pigmentation between experimental groups.
        
        Args:
            film_summary: DataFrame with film-level data
            group_column: Column name for grouping
            iod_column: IOD column to compare
            
        Returns:
            Dict with group comparison results
        """
        if iod_column not in film_summary.columns:
            return {'error': f'{iod_column} column not found'}
        
        if group_column not in film_summary.columns:
            return {'error': f'{group_column} column not found'}
        
        groups = [g for g in film_summary[group_column].unique() 
                  if g != 'ungrouped' and pd.notna(g)]
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for comparison'}
        
        # Group statistics
        group_stats = {}
        group_data = {}
        
        for group in groups:
            values = film_summary[film_summary[group_column] == group][iod_column].dropna().values
            if len(values) >= 2:
                group_stats[group] = {
                    'n': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'cv': float(np.std(values) / np.mean(values) * 100) if np.mean(values) > 0 else np.nan
                }
                group_data[group] = values
        
        valid_groups = list(group_data.keys())
        
        if len(valid_groups) < 2:
            return {
                'error': 'Insufficient data in groups',
                'group_statistics': group_stats
            }
        
        # Statistical comparison
        stat_analyzer = StatisticalAnalyzer(alpha=self.alpha)
        
        if len(valid_groups) == 2:
            names = valid_groups
            comparison = stat_analyzer.compare_two_groups(
                group_data[names[0]], group_data[names[1]],
                group1_name=names[0], group2_name=names[1]
            )
        else:
            comparison = stat_analyzer.compare_multiple_groups(group_data, correction='holm')
        
        return {
            'metric': iod_column,
            'group_statistics': group_stats,
            'n_groups': len(valid_groups),
            'comparison': comparison
        }


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
        
        result = {
            'n_deposits': n_total,
            # Basic statistics
            'mean_area': float(np.mean(areas)),
            'std_area': float(np.std(areas)),
            'median_area': float(np.median(areas)),
            'min_area': float(np.min(areas)),
            'max_area': float(np.max(areas)),
            'area_cv': float(np.std(areas) / np.mean(areas) * 100) if np.mean(areas) > 0 else np.nan,
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
            'bimodality_coefficient': self._bimodality_coefficient(areas),
            'is_bimodal': self._bimodality_coefficient(areas) > 0.555 if len(areas) >= 4 else False,
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
                    result[f'{label}_area_cv'] = float(np.std(label_areas) / np.mean(label_areas) * 100) if np.mean(label_areas) > 0 else np.nan
        
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
        """
        Compare deposit size between experimental groups.
        
        Args:
            film_summary: DataFrame with film-level data
            group_column: Column name for grouping
            size_column: Size metric column to compare
            
        Returns:
            Dict with group comparison results
        """
        if size_column not in film_summary.columns:
            return {'error': f'{size_column} column not found'}
        
        if group_column not in film_summary.columns:
            return {'error': f'{group_column} column not found'}
        
        groups = [g for g in film_summary[group_column].unique() 
                  if g != 'ungrouped' and pd.notna(g)]
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for comparison'}
        
        # Group data
        group_data = {}
        group_stats = {}
        
        for group in groups:
            values = film_summary[film_summary[group_column] == group][size_column].dropna().values
            if len(values) >= 2:
                group_data[group] = values
                group_stats[group] = {
                    'n': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'cv': float(np.std(values) / np.mean(values) * 100) if np.mean(values) > 0 else np.nan
                }
        
        valid_groups = list(group_data.keys())
        
        if len(valid_groups) < 2:
            return {'error': 'Insufficient data in groups', 'group_statistics': group_stats}
        
        # Statistical comparison
        stat_analyzer = StatisticalAnalyzer(alpha=self.alpha)
        
        if len(valid_groups) == 2:
            names = valid_groups
            comparison = stat_analyzer.compare_two_groups(
                group_data[names[0]], group_data[names[1]],
                group1_name=names[0], group2_name=names[1]
            )
        else:
            comparison = stat_analyzer.compare_multiple_groups(group_data, correction='holm')
        
        return {
            'metric': size_column,
            'group_statistics': group_stats,
            'n_groups': len(valid_groups),
            'comparison': comparison
        }


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
            'deposits_cv': float(np.std(n_total) / np.mean(n_total) * 100) if np.mean(n_total) > 0 else np.nan
        }
        
        # Per-fly normalization
        if n_flies_column and n_flies_column in film_summary.columns:
            n_flies = film_summary[n_flies_column].values
            valid_mask = n_flies > 0
            
            if np.any(valid_mask):
                deposits_per_fly = n_total[valid_mask] / n_flies[valid_mask]
                result['mean_deposits_per_fly'] = float(np.mean(deposits_per_fly))
                result['std_deposits_per_fly'] = float(np.std(deposits_per_fly))
                result['deposits_per_fly_cv'] = float(np.std(deposits_per_fly) / np.mean(deposits_per_fly) * 100) if np.mean(deposits_per_fly) > 0 else np.nan
        
        # ROD fraction statistics
        if 'rod_fraction' in film_summary.columns:
            rod_frac = film_summary['rod_fraction'].dropna().values
            if len(rod_frac) > 0:
                result['mean_rod_fraction'] = float(np.mean(rod_frac))
                result['std_rod_fraction'] = float(np.std(rod_frac))
                result['rod_fraction_cv'] = float(np.std(rod_frac) / np.mean(rod_frac) * 100) if np.mean(rod_frac) > 0 else np.nan
        
        return result
    
    def compare_density_between_groups(
        self,
        film_summary: pd.DataFrame,
        group_column: str,
        density_metric: str = 'n_total'
    ) -> Dict:
        """
        Compare deposit density between experimental groups.
        
        Args:
            film_summary: DataFrame with film-level data
            group_column: Column name for grouping
            density_metric: Metric to compare ('n_total', 'rod_fraction', etc.)
            
        Returns:
            Dict with group comparison results
        """
        if density_metric not in film_summary.columns:
            return {'error': f'{density_metric} column not found'}
        
        if group_column not in film_summary.columns:
            return {'error': f'{group_column} column not found'}
        
        groups = [g for g in film_summary[group_column].unique() 
                  if g != 'ungrouped' and pd.notna(g)]
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for comparison'}
        
        # Group data
        group_data = {}
        group_stats = {}
        
        for group in groups:
            values = film_summary[film_summary[group_column] == group][density_metric].dropna().values
            if len(values) >= 2:
                group_data[group] = values
                group_stats[group] = {
                    'n': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'cv': float(np.std(values) / np.mean(values) * 100) if np.mean(values) > 0 else np.nan
                }
        
        valid_groups = list(group_data.keys())
        
        if len(valid_groups) < 2:
            return {'error': 'Insufficient data in groups', 'group_statistics': group_stats}
        
        # Statistical comparison
        stat_analyzer = StatisticalAnalyzer(alpha=self.alpha)
        
        if len(valid_groups) == 2:
            names = valid_groups
            comparison = stat_analyzer.compare_two_groups(
                group_data[names[0]], group_data[names[1]],
                group1_name=names[0], group2_name=names[1]
            )
        else:
            comparison = stat_analyzer.compare_multiple_groups(group_data, correction='holm')
        
        return {
            'metric': density_metric,
            'group_statistics': group_stats,
            'n_groups': len(valid_groups),
            'comparison': comparison
        }
    
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
            'circularity_cv': float(np.std(circularity) / np.mean(circularity) * 100) if np.mean(circularity) > 0 else np.nan,
            # Aspect ratio statistics
            'mean_aspect_ratio': float(np.mean(aspect_ratio)),
            'std_aspect_ratio': float(np.std(aspect_ratio)),
            'median_aspect_ratio': float(np.median(aspect_ratio)),
            'aspect_ratio_cv': float(np.std(aspect_ratio) / np.mean(aspect_ratio) * 100) if np.mean(aspect_ratio) > 0 else np.nan,
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
        """
        Compare morphology metrics between experimental groups.
        
        Args:
            film_summary: DataFrame with film-level data
            group_column: Column name for grouping
            metric_column: Morphology metric to compare
            
        Returns:
            Dict with group comparison results
        """
        if metric_column not in film_summary.columns:
            return {'error': f'{metric_column} column not found'}
        
        if group_column not in film_summary.columns:
            return {'error': f'{group_column} column not found'}
        
        groups = [g for g in film_summary[group_column].unique() 
                  if g != 'ungrouped' and pd.notna(g)]
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for comparison'}
        
        # Group data
        group_data = {}
        group_stats = {}
        
        for group in groups:
            values = film_summary[film_summary[group_column] == group][metric_column].dropna().values
            if len(values) >= 2:
                group_data[group] = values
                group_stats[group] = {
                    'n': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'cv': float(np.std(values) / np.mean(values) * 100) if np.mean(values) > 0 else np.nan
                }
        
        valid_groups = list(group_data.keys())
        
        if len(valid_groups) < 2:
            return {'error': 'Insufficient data in groups', 'group_statistics': group_stats}
        
        # Statistical comparison
        stat_analyzer = StatisticalAnalyzer(alpha=self.alpha)
        
        if len(valid_groups) == 2:
            names = valid_groups
            comparison = stat_analyzer.compare_two_groups(
                group_data[names[0]], group_data[names[1]],
                group1_name=names[0], group2_name=names[1]
            )
        else:
            comparison = stat_analyzer.compare_multiple_groups(group_data, correction='holm')
        
        return {
            'metric': metric_column,
            'group_statistics': group_stats,
            'n_groups': len(valid_groups),
            'comparison': comparison
        }
    
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

def run_comprehensive_analysis(
    film_summary: pd.DataFrame,
    deposits_df: pd.DataFrame = None,
    group_column: str = None,
    n_flies_column: str = None,
    include_analyses: List[str] = None
) -> Dict:
    """
    Run all available statistical analyses in one call.
    
    This is the main entry point for comprehensive statistical analysis.
    Results from this function can be used to generate reports.
    
    Args:
        film_summary: DataFrame with film-level summary data
        deposits_df: Optional DataFrame with individual deposit data
        group_column: Optional column name for group comparisons
        n_flies_column: Optional column name for number of flies (per-fly normalization)
        include_analyses: List of analyses to include. If None, all are included.
            Options: 'basic', 'ph', 'pigmentation', 'size', 'density', 
                     'correlation', 'morphology'
    
    Returns:
        Dict with all analysis results organized by category
    """
    if include_analyses is None:
        include_analyses = ['basic', 'ph', 'pigmentation', 'size', 'density', 
                           'correlation', 'morphology']
    
    results = {
        'metadata': {
            'n_films': len(film_summary),
            'n_deposits': len(deposits_df) if deposits_df is not None else 0,
            'group_column': group_column,
            'analyses_included': include_analyses
        }
    }
    
    # 1. Basic statistical comparison (original run_all_tests)
    if 'basic' in include_analyses:
        try:
            analyzer = StatisticalAnalyzer()
            results['basic'] = analyzer.run_all_tests(
                film_summary=film_summary,
                group_by=group_column
            )
        except Exception as e:
            results['basic'] = {'error': str(e)}
    
    # 2. pH Analysis
    if 'ph' in include_analyses:
        try:
            results['ph'] = analyze_ph(
                deposits_df=deposits_df,
                film_summary=film_summary,
                group_column=group_column
            )
        except Exception as e:
            results['ph'] = {'error': str(e)}
    
    # 3. Pigmentation Analysis
    if 'pigmentation' in include_analyses:
        try:
            results['pigmentation'] = analyze_pigmentation(
                deposits_df=deposits_df,
                film_summary=film_summary,
                group_column=group_column,
                n_flies_column=n_flies_column
            )
        except Exception as e:
            results['pigmentation'] = {'error': str(e)}
    
    # 4. Size Distribution Analysis
    if 'size' in include_analyses:
        try:
            results['size_distribution'] = analyze_size_distribution(
                deposits_df=deposits_df,
                film_summary=film_summary,
                group_column=group_column
            )
        except Exception as e:
            results['size_distribution'] = {'error': str(e)}
    
    # 5. Density Analysis
    if 'density' in include_analyses:
        try:
            results['density'] = analyze_density(
                deposits_df=deposits_df,
                film_summary=film_summary,
                group_column=group_column,
                n_flies_column=n_flies_column
            )
        except Exception as e:
            results['density'] = {'error': str(e)}
    
    # 6. Correlation Analysis (requires individual deposit data)
    if 'correlation' in include_analyses and deposits_df is not None:
        try:
            results['correlation'] = analyze_correlations(deposits_df)
        except Exception as e:
            results['correlation'] = {'error': str(e)}
    
    # 7. Morphology Analysis
    if 'morphology' in include_analyses:
        try:
            results['morphology'] = analyze_morphology(
                deposits_df=deposits_df,
                film_summary=film_summary,
                group_column=group_column
            )
        except Exception as e:
            results['morphology'] = {'error': str(e)}
    
    # Generate overall summary
    results['summary'] = _generate_comprehensive_summary(results)
    
    return results


def _generate_comprehensive_summary(results: Dict) -> Dict:
    """Generate human-readable summary of all analyses."""
    summary = {
        'significant_findings': [],
        'key_metrics': {},
        'recommendations': []
    }
    
    # Basic analysis summary
    if 'basic' in results and 'summary' in results['basic']:
        basic_summary = results['basic']['summary']
        if basic_summary.get('significant_metrics'):
            for item in basic_summary['significant_metrics']:
                summary['significant_findings'].append({
                    'category': 'basic',
                    'finding': f"{item['metric']} differs significantly between groups",
                    'p_value': item['p_value'],
                    'test': item['test']
                })
    
    # pH summary
    if 'ph' in results and 'error' not in results['ph']:
        ph_data = results['ph']
        if 'deposit_level' in ph_data:
            dl = ph_data['deposit_level']
            if 'mean_ph' in dl:
                summary['key_metrics']['mean_estimated_ph'] = dl['mean_ph']
                summary['key_metrics']['fraction_acidic'] = dl.get('fraction_acidic', 0)
        
        if 'group_comparison' in ph_data:
            gc = ph_data['group_comparison']
            if 'comparison' in gc and gc['comparison'].get('significant'):
                summary['significant_findings'].append({
                    'category': 'ph',
                    'finding': 'pH differs significantly between groups',
                    'p_value': gc['comparison'].get('p_value')
                })
    
    # Pigmentation summary
    if 'pigmentation' in results and 'error' not in results['pigmentation']:
        pig_data = results['pigmentation']
        if 'deposit_level' in pig_data:
            dl = pig_data['deposit_level']
            summary['key_metrics']['total_iod'] = dl.get('total_iod', 0)
            summary['key_metrics']['mean_pigment_density'] = dl.get('mean_pigment_density', 0)
    
    # Size summary
    if 'size_distribution' in results and 'error' not in results['size_distribution']:
        size_data = results['size_distribution']
        if 'distribution' in size_data:
            dist = size_data['distribution']
            summary['key_metrics']['mean_deposit_area'] = dist.get('mean_area', 0)
            summary['key_metrics']['is_bimodal'] = dist.get('is_bimodal', False)
            
            if dist.get('is_bimodal'):
                summary['recommendations'].append(
                    'Size distribution is bimodal - Normal and ROD may have distinct size ranges'
                )
    
    # Density summary
    if 'density' in results and 'error' not in results['density']:
        dens_data = results['density']
        if 'film_level' in dens_data:
            fl = dens_data['film_level']
            summary['key_metrics']['mean_deposits_per_film'] = fl.get('mean_deposits_per_film', 0)
            summary['key_metrics']['mean_rod_fraction'] = fl.get('mean_rod_fraction', 0)
    
    # Correlation summary
    if 'correlation' in results and 'error' not in results['correlation']:
        corr_data = results['correlation']
        if 'key_correlations' in corr_data:
            for key, corr in corr_data['key_correlations'].items():
                if isinstance(corr, dict) and corr.get('significant') and abs(corr.get('r', 0)) >= 0.5:
                    summary['significant_findings'].append({
                        'category': 'correlation',
                        'finding': f"Strong {corr.get('direction', '')} correlation: {key} (r={corr['r']:.2f})",
                        'p_value': corr.get('p_value')
                    })
    
    # Morphology summary
    if 'morphology' in results and 'error' not in results['morphology']:
        morph_data = results['morphology']
        if 'distribution' in morph_data:
            dist = morph_data['distribution']
            summary['key_metrics']['mean_circularity'] = dist.get('mean_circularity', 0)
            summary['key_metrics']['fraction_regular'] = dist.get('fraction_regular', 0)
            summary['key_metrics']['fraction_irregular'] = dist.get('fraction_irregular', 0)
        
        if 'outliers' in morph_data:
            outliers = morph_data['outliers']
            if outliers.get('fraction_outliers', 0) > 0.1:
                summary['recommendations'].append(
                    f"High proportion of morphological outliers ({outliers['fraction_outliers']:.1%}) - consider reviewing data quality"
                )
    
    return summary
