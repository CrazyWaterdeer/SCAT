"""Top-level orchestration: run every analyzer over a dataset and summarize."""

import pandas as pd
from typing import List, Dict

from .common import StatisticalAnalyzer
from .ph import analyze_ph
from .pigmentation import analyze_pigmentation
from .size import analyze_size_distribution
from .density import analyze_density
from .correlation import analyze_correlations
from .morphology import analyze_morphology


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

    # Derive an artifact-EXCLUSIVE per-image deposit count (Normal+ROD) IN MEMORY so the
    # "Deposit Count" group comparison matches the rest of the report (Summary card, per-group
    # table, Methods all exclude artifacts). Never written to CSV → pipeline parity untouched.
    if ('n_normal' in film_summary.columns and 'n_rod' in film_summary.columns
            and 'n_deposits' not in film_summary.columns):
        film_summary = film_summary.copy()
        film_summary['n_deposits'] = film_summary['n_normal'] + film_summary['n_rod']

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
