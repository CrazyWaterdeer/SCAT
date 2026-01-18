"""
SCAT - Spot Classification and Analysis Tool
ML-based Drosophila excreta analysis.
"""

__version__ = "1.0.0"

from .detector import DepositDetector, Deposit
from .features import FeatureExtractor
from .classifier import ThresholdClassifier, RandomForestClassifier, ClassifierConfig
from .analyzer import Analyzer, AnalysisResult, ReportGenerator
from .trainer import DataLoader, RFTrainer, CNNTrainer, train_from_labels
from .statistics import (
    StatisticalAnalyzer, generate_statistics_report, run_comprehensive_analysis,
    analyze_ph, analyze_pigmentation, analyze_size_distribution,
    analyze_density, analyze_correlations, analyze_morphology,
    pHAnalyzer, PigmentationAnalyzer, SizeDistributionAnalyzer,
    DensityAnalyzer, CorrelationAnalyzer, MorphologyAnalyzer
)
from .spatial import SpatialAnalyzer, SpatialResult, aggregate_spatial_stats
from .report import generate_report

# Optional segmentation (requires PyTorch)
try:
    from .segmentation import (
        UNet, UNetDetector, SegmentationTrainer,
        SegmentationDataLoader, train_segmentation_model
    )
except ImportError:
    UNet = None
    UNetDetector = None
    SegmentationTrainer = None
    SegmentationDataLoader = None
    train_segmentation_model = None

# Optional visualization (requires matplotlib)
try:
    from .visualization import (
        Visualizer, generate_all_visualizations,
        SpatialVisualizer, generate_spatial_visualizations
    )
except ImportError:
    Visualizer = None
    generate_all_visualizations = None
    SpatialVisualizer = None
    generate_spatial_visualizations = None

__all__ = [
    'DepositDetector', 'Deposit', 'FeatureExtractor',
    'ThresholdClassifier', 'RandomForestClassifier', 'ClassifierConfig',
    'Analyzer', 'AnalysisResult', 'ReportGenerator',
    'DataLoader', 'RFTrainer', 'CNNTrainer', 'train_from_labels',
    # Statistics - Core
    'StatisticalAnalyzer', 'generate_statistics_report', 'run_comprehensive_analysis',
    # Statistics - Analyzers
    'pHAnalyzer', 'PigmentationAnalyzer', 'SizeDistributionAnalyzer',
    'DensityAnalyzer', 'CorrelationAnalyzer', 'MorphologyAnalyzer',
    # Statistics - Convenience functions
    'analyze_ph', 'analyze_pigmentation', 'analyze_size_distribution',
    'analyze_density', 'analyze_correlations', 'analyze_morphology',
    # Spatial
    'SpatialAnalyzer', 'SpatialResult', 'aggregate_spatial_stats',
    'generate_report',
    'UNet', 'UNetDetector', 'SegmentationTrainer',
    'SegmentationDataLoader', 'train_segmentation_model',
    'Visualizer', 'generate_all_visualizations',
    'SpatialVisualizer', 'generate_spatial_visualizations'
]
