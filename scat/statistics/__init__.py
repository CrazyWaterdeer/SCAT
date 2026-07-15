"""Statistical analysis for SCAT — group comparisons, normality tests, effect sizes.

Split into submodules for maintainability; this package re-exports the same public
surface the single ``scat.statistics`` module exposed, so existing imports
(``from scat.statistics import ...``) keep working unchanged.
"""

from .common import (
    coefficient_of_variation, correct_pvalues, compare_group_values,
    _compare_metric_between_groups, StatisticalAnalyzer, generate_statistics_report,
)
from .ph import pHAnalyzer, analyze_ph
from .pigmentation import PigmentationAnalyzer, analyze_pigmentation
from .size import SizeDistributionAnalyzer, analyze_size_distribution
from .density import DensityAnalyzer, analyze_density
from .correlation import CorrelationAnalyzer, analyze_correlations
from .morphology import MorphologyAnalyzer, analyze_morphology
from .comprehensive import run_comprehensive_analysis, _generate_comprehensive_summary

__all__ = [
    'coefficient_of_variation', 'correct_pvalues', 'compare_group_values',
    'StatisticalAnalyzer', 'generate_statistics_report', 'run_comprehensive_analysis',
    'pHAnalyzer', 'PigmentationAnalyzer', 'SizeDistributionAnalyzer',
    'DensityAnalyzer', 'CorrelationAnalyzer', 'MorphologyAnalyzer',
    'analyze_ph', 'analyze_pigmentation', 'analyze_size_distribution',
    'analyze_density', 'analyze_correlations', 'analyze_morphology',
]
