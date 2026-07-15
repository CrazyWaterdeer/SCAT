"""Core statistical primitives and the base StatisticalAnalyzer.

Shared building blocks used across the domain analyzers: coefficient of
variation, multiple-comparison correction, the group-comparison dispatch, and
the StatisticalAnalyzer class (two-group / multi-group tests, effect sizes)."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from itertools import combinations


def coefficient_of_variation(data) -> float:
    """Canonical coefficient of variation (%) = std / |mean| * 100, over the non-NaN values.

    The single correct implementation used everywhere (previously duplicated ~18x inline as
    ``std/mean*100 if mean>0 else nan``, which was wrong for two cases): CV of a single sample
    is undefined (return NaN, not the misleading 0 a lone value gives), and missing values are
    skipped (pandas-style) rather than propagating NaN through the whole group. CV is a
    descriptive consistency metric — it does not feed the significance tests, so this changes
    only degenerate groups (n<2 or with NaNs), to their honest values.
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return np.nan
    mean = np.mean(data)
    if mean == 0:
        return np.nan
    return float((np.std(data) / abs(mean)) * 100)


def correct_pvalues(p_values, method: str) -> list:
    """Canonical multiple-comparison correction ('bonferroni' | 'holm'), returning a list in the SAME
    order as the input. Single source of truth: visualization.py used to carry a drifted copy whose
    Holm branch mis-restored the original comparison order (a latent significance-bracket bug)."""
    p_values = np.array(p_values)
    n = len(p_values)
    if method == 'bonferroni':
        return list(np.minimum(p_values * n, 1.0))
    if method == 'holm':
        # Holm-Bonferroni step-down: sort ascending, the k-th smallest gets multiplier (n - rank),
        # enforce monotone non-decreasing via a forward cumulative max, then map back to input order.
        order = np.argsort(p_values)
        ranked = p_values[order]
        multipliers = n - np.arange(n)
        adjusted = np.minimum(np.maximum.accumulate(ranked * multipliers), 1.0)
        corrected = np.empty(n)
        corrected[order] = adjusted
        return list(corrected)
    return list(p_values)


def compare_group_values(group_values: Dict, alpha: float = 0.05) -> Dict:
    """Shared group-comparison dispatch for every compare_*_between_groups method: a two-group
    test (t / Mann-Whitney) for exactly two groups, else a multi-group test (ANOVA / Kruskal)
    with Holm correction, over a {group_name: values} mapping (insertion order preserved)."""
    analyzer = StatisticalAnalyzer(alpha=alpha)
    names = list(group_values.keys())
    if len(names) == 2:
        return analyzer.compare_two_groups(
            group_values[names[0]], group_values[names[1]],
            group1_name=names[0], group2_name=names[1])
    return analyzer.compare_multiple_groups(group_values, correction='holm')


def _compare_metric_between_groups(film_summary, group_column, metric_column, alpha, *,
                                   include_median: bool) -> Dict:
    """Shared body of the pigmentation/size/density/morphology group comparisons — they live on four
    different analyzer classes but differ only in the metric column and whether the per-group stats
    include a median (compare_ph is bespoke: pH transform + different keys, so it stays separate).
    Per-group descriptive stats (n/mean/std/[median]/cv) over the non-NaN metric values, then the
    shared significance dispatch. Key order/values match each metric's original output exactly."""
    if metric_column not in film_summary.columns:
        return {'error': f'{metric_column} column not found'}

    if group_column not in film_summary.columns:
        return {'error': f'{group_column} column not found'}

    groups = [g for g in film_summary[group_column].unique()
              if g != 'ungrouped' and pd.notna(g)]

    if len(groups) < 2:
        return {'error': 'Need at least 2 groups for comparison'}

    group_data = {}
    group_stats = {}
    for group in groups:
        values = film_summary[film_summary[group_column] == group][metric_column].dropna().values
        if len(values) >= 2:
            group_data[group] = values
            stats = {'n': len(values), 'mean': float(np.mean(values)), 'std': float(np.std(values))}
            if include_median:
                stats['median'] = float(np.median(values))
            stats['cv'] = coefficient_of_variation(values)
            group_stats[group] = stats

    valid_groups = list(group_data.keys())

    if len(valid_groups) < 2:
        return {'error': 'Insufficient data in groups', 'group_statistics': group_stats}

    comparison = compare_group_values(group_data, alpha)

    return {
        'metric': metric_column,
        'group_statistics': group_stats,
        'n_groups': len(valid_groups),
        'comparison': comparison
    }


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
        """CV (%) — delegates to the canonical module-level coefficient_of_variation."""
        return coefficient_of_variation(data)
    
    def _correct_pvalues(self, p_values: List[float], method: str) -> List[float]:
        """Apply multiple comparison correction (delegates to the module-level canonical impl)."""
        return correct_pvalues(p_values, method)
    
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


# =============================================================================
# pH Analysis
# =============================================================================
