"""
Visualization module for SCAT.
Provides PCA, clustering, density plots, and comparison charts.
Publication-ready figures with GraphPad Prism-like styling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

# =============================================================================
# Color Palettes
# =============================================================================

# Default gray for ungrouped data (matches UI Theme.SECONDARY)
DEFAULT_GRAY = '#636867'

# Pastel palette for experiments (lighter, publication-friendly)
PASTEL_PALETTE = [
    '#a8d5ba',  # Light green
    '#f7b8b8',  # Light coral/pink
    '#b8d4e3',  # Light blue
    '#f5e6ab',  # Light yellow
    '#d4b8e3',  # Light purple
    '#ffd4a8',  # Light orange
    '#b8e3d4',  # Light teal
    '#e3b8d4',  # Light magenta
]

# Control group color (neutral gray)
CONTROL_COLOR = '#9E9E9E'

# Colors for deposit types
DEPOSIT_COLORS = {
    'normal': '#4CAF50',
    'rod': '#F44336', 
    'artifact': '#9E9E9E',
    'unknown': '#FFC107'
}

# =============================================================================
# Feature Labels
# =============================================================================

# Feature label mapping for proper display
# Labels should be scientifically accurate and publication-ready
FEATURE_LABELS = {
    # Counts
    'n_total': 'Total Deposit Count',
    'n_normal': 'Normal Deposit Count',
    'n_rod': 'ROD Deposit Count',
    'n_artifact': 'Artifact Count',
    # Fractions
    'rod_fraction': 'ROD Fraction',
    # Area (morphology/size)
    'normal_mean_area': 'Normal Deposit Size (px²)',
    'normal_std_area': 'Normal Size Variability (px²)',
    'rod_mean_area': 'ROD Deposit Size (px²)',
    'rod_std_area': 'ROD Size Variability (px²)',
    'area_px': 'Area (px²)',
    'area_um2': 'Area (μm²)',
    # IOD (Integrated Optical Density - pigment amount)
    'normal_mean_iod': 'Normal Mean Pigment (IOD)',
    'normal_total_iod': 'Normal Total Pigment (IOD)',
    'rod_mean_iod': 'ROD Mean Pigment (IOD)',
    'rod_total_iod': 'ROD Total Pigment (IOD)',
    'total_iod': 'Total Pigment Amount (IOD)',
    'iod': 'Pigment Amount (IOD)',
    # Color features - Hue reflects pH via Bromophenol Blue indicator
    'mean_hue': 'pH Indicator Hue (°)',
    'normal_mean_hue': 'Normal pH Indicator (Hue °)',
    'rod_mean_hue': 'ROD pH Indicator (Hue °)',
    # Saturation
    'mean_saturation': 'Color Saturation',
    # Lightness - reflects pigment concentration/density
    'mean_lightness': 'Pigment Density (Lightness)',
    'normal_mean_lightness': 'Normal Pigment Density (Lightness)',
    'rod_mean_lightness': 'ROD Pigment Density (Lightness)',
    # Shape features (morphology)
    'circularity': 'Circularity',
    'normal_mean_circularity': 'Normal Circularity',
    'rod_mean_circularity': 'ROD Circularity',
    'aspect_ratio': 'Aspect Ratio',
    # Position
    'x': 'X Position (px)',
    'y': 'Y Position (px)',
    'width': 'Width (px)',
    'height': 'Height (px)',
}

def get_feature_label(feature: str) -> str:
    """Get display label for a feature name."""
    return FEATURE_LABELS.get(feature, feature.replace('_', ' ').title())


def hue_to_rgb(hue_degrees: float, saturation: float = 0.7, lightness: float = 0.6) -> str:
    """
    Convert Hue value (0-360 degrees) to RGB hex color.
    
    Uses HSL color space for intuitive color mapping.
    Saturation and lightness are set to produce pleasant, visible colors.
    
    Args:
        hue_degrees: Hue value in degrees (0-360)
        saturation: Color saturation (0-1), default 0.7 for visible colors
        lightness: Color lightness (0-1), default 0.6 for balanced visibility
        
    Returns:
        RGB hex color string (e.g., '#FF5733')
    """
    import colorsys
    
    # Normalize hue to 0-1 range
    h = (hue_degrees % 360) / 360.0
    
    # Convert HSL to RGB
    # colorsys uses HLS (Hue, Lightness, Saturation) order
    r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
    
    # Convert to hex
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))


def is_hue_metric(metric: str) -> bool:
    """Check if metric is a hue-related metric that should use hue-based colors."""
    return 'hue' in metric.lower()


def get_palette(groups: List[str], control_group: str = None) -> Dict[str, str]:
    """
    Get color palette for groups.
    
    Args:
        groups: List of group names
        control_group: Name of control group (will be gray)
    
    Returns:
        Dict mapping group names to colors
    """
    palette = {}
    color_idx = 0
    
    for group in groups:
        if control_group and group == control_group:
            palette[group] = CONTROL_COLOR
        else:
            palette[group] = PASTEL_PALETTE[color_idx % len(PASTEL_PALETTE)]
            color_idx += 1
    
    return palette


def apply_publication_style(ax, despine: bool = True):
    """
    Apply publication-ready styling to axes.
    
    Args:
        ax: Matplotlib axes
        despine: Remove top and right spines
    """
    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)  # Grid behind data

# Lazy loading flags
_viz_libs_loaded = False
_plt = None
_mpatches = None
_to_rgba = None
_sns = None
_PCA = None
_StandardScaler = None
_KMeans = None
_AgglomerativeClustering = None

HAS_MATPLOTLIB = False
HAS_SEABORN = False
HAS_SKLEARN = False


def _load_viz_libs():
    """Lazy load visualization libraries."""
    global _viz_libs_loaded, _plt, _mpatches, _to_rgba, _sns
    global _PCA, _StandardScaler, _KMeans, _AgglomerativeClustering
    global HAS_MATPLOTLIB, HAS_SEABORN, HAS_SKLEARN
    
    if _viz_libs_loaded:
        return
    
    # Load matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import to_rgba
        _plt = plt
        _mpatches = mpatches
        _to_rgba = to_rgba
        HAS_MATPLOTLIB = True
    except ImportError:
        warnings.warn("matplotlib not installed. Visualization features disabled.")
    
    # Load seaborn
    try:
        import seaborn as sns
        _sns = sns
        HAS_SEABORN = True
    except ImportError:
        pass
    
    # Load sklearn
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, AgglomerativeClustering
        _PCA = PCA
        _StandardScaler = StandardScaler
        _KMeans = KMeans
        _AgglomerativeClustering = AgglomerativeClustering
        HAS_SKLEARN = True
    except ImportError:
        pass
    
    _viz_libs_loaded = True


class Visualizer:
    """Generate publication-ready visualizations for excreta analysis."""
    
    def __init__(self, output_dir: Path, style: str = 'whitegrid'):
        _load_viz_libs()  # Lazy load visualization libraries
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB and HAS_SEABORN:
            _sns.set_style(style)
            # Publication-ready defaults
            _plt.rcParams.update({
                'figure.figsize': (10, 8),
                'figure.dpi': 150,
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'axes.spines.top': False,
                'axes.spines.right': False,
            })
    
    def pca_plot(
        self,
        film_summary: pd.DataFrame,
        features: List[str] = None,
        color_by: str = None,
        control_group: str = None,
        show_loadings: bool = True,
        loading_threshold: float = 0.1,
        title: str = "PCA of Image Samples",
        filename: str = "pca_plot.png"
    ) -> Optional[str]:
        """
        Generate PCA plot of film samples.
        
        Args:
            film_summary: DataFrame with film-level data
            features: Columns to use for PCA (default: numeric columns)
            color_by: Column for color coding (e.g., 'condition')
            control_group: Name of control group (shown in gray)
            show_loadings: Whether to show loading vectors (arrows)
            loading_threshold: Minimum loading magnitude to display
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB or not HAS_SKLEARN:
            warnings.warn("PCA plot requires matplotlib and scikit-learn")
            return None
        
        # Select features
        if features is None:
            features = [
                'n_normal', 'n_rod', 'rod_fraction',
                'normal_mean_area', 'rod_mean_area',
                'normal_mean_iod', 'total_iod',
                'normal_mean_hue', 'normal_mean_lightness'
            ]
        
        # Filter available features
        features = [f for f in features if f in film_summary.columns]
        
        if len(features) < 2:
            warnings.warn("Not enough features for PCA")
            return None
        
        # Prepare data
        df = film_summary[features].dropna()
        if len(df) < 3:
            warnings.warn("Not enough samples for PCA")
            return None
        
        # Standardize and fit PCA
        scaler = _StandardScaler()
        X_scaled = scaler.fit_transform(df)
        
        pca = _PCA(n_components=min(2, len(features)))
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate total variance explained
        total_var = sum(pca.explained_variance_ratio_) * 100
        
        # Plot
        fig, ax = _plt.subplots(figsize=(10, 8))
        
        if color_by and color_by in film_summary.columns:
            # Get matching indices
            valid_idx = df.index
            groups = film_summary.loc[valid_idx, color_by]
            unique_groups = sorted(groups.unique())
            
            # Get palette with control group support
            palette = get_palette(unique_groups, control_group)
            
            for group in unique_groups:
                mask = groups == group
                ax.scatter(
                    X_pca[mask, 0], X_pca[mask, 1],
                    c=[palette[group]], label=group, s=100, alpha=0.7, 
                    edgecolors='black', linewidth=0.5
                )
            ax.legend(title=color_by.replace('_', ' ').title(), framealpha=0.9)
        else:
            # No grouping - use default gray
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=DEFAULT_GRAY, 
                      s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'{title}\n(Total variance explained: {total_var:.1f}%)')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        
        # Add loading vectors (optional)
        if show_loadings:
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            scale = 2
            for i, feature in enumerate(features):
                # Skip small loadings
                magnitude = np.sqrt(loadings[i, 0]**2 + loadings[i, 1]**2)
                if magnitude < loading_threshold:
                    continue
                    
                ax.annotate(
                    '', xy=(loadings[i, 0]*scale, loadings[i, 1]*scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#555555', alpha=0.6, lw=1.2)
                )
                ax.text(
                    loadings[i, 0]*scale*1.1, loadings[i, 1]*scale*1.1,
                    get_feature_label(feature), fontsize=8, color='#333333', alpha=0.8
                )
        
        apply_publication_style(ax)
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath, dpi=300, bbox_inches='tight')
        _plt.close()
        
        return str(filepath)
    
    def hue_density_plot(
        self,
        deposits_data: Dict[str, pd.DataFrame],
        title: str = "Hue Distribution by Condition",
        filename: str = "hue_density.png",
        bandwidth: float = 7
    ) -> Optional[str]:
        """
        Generate hue (pH proxy) density plot.
        
        Args:
            deposits_data: Dict mapping condition names to deposit DataFrames
            title: Plot title
            filename: Output filename
            bandwidth: KDE bandwidth
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
        
        fig, ax = _plt.subplots(figsize=(12, 6))
        
        colors = _plt.cm.Set1(np.linspace(0, 1, len(deposits_data)))
        
        for (name, df), color in zip(deposits_data.items(), colors):
            if 'mean_hue' in df.columns:
                hue_values = df['mean_hue'].dropna()
                if len(hue_values) > 10:
                    _sns.kdeplot(
                        data=hue_values, ax=ax, label=f"{name} (n={len(hue_values)})",
                        color=color, linewidth=2, bw_adjust=bandwidth/10
                    )
        
        ax.set_xlabel('Hue (degrees)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, 360)
        
        # Add pH reference zones for Bromophenol Blue
        ax.axvspan(0, 60, alpha=0.1, color='yellow', label='Acidic')
        ax.axvspan(60, 150, alpha=0.1, color='green', label='Transitional')
        ax.axvspan(150, 360, alpha=0.1, color='blue', label='Basic')
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def violin_comparison(
        self,
        film_summary: pd.DataFrame,
        metric: str,
        group_by: str,
        control_group: str = None,
        show_significance: bool = False,
        title: str = None,
        filename: str = None,
        ylabel: str = None
    ) -> Optional[str]:
        """
        Generate violin plot comparing groups.
        
        Args:
            film_summary: DataFrame with film-level data
            metric: Column to plot (e.g., 'rod_fraction')
            group_by: Column for grouping
            control_group: Name of control group (shown in gray for color only)
            show_significance: Whether to show statistical significance
            title: Plot title
            filename: Output filename
            ylabel: Y-axis label
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
        
        if metric not in film_summary.columns or group_by not in film_summary.columns:
            return None
        
        # Get unique groups and create palette
        unique_groups = sorted(film_summary[group_by].dropna().unique())
        
        # For hue metrics, use actual hue values as colors
        if is_hue_metric(metric):
            palette = {}
            for group in unique_groups:
                mean_hue = film_summary[film_summary[group_by] == group][metric].mean()
                if not np.isnan(mean_hue):
                    palette[group] = hue_to_rgb(mean_hue)
                else:
                    palette[group] = DEFAULT_GRAY
        else:
            palette = get_palette(unique_groups, control_group)
        
        fig, ax = _plt.subplots(figsize=(max(8, len(unique_groups) * 1.5), 6))
        
        # Violin with lighter alpha
        violin = _sns.violinplot(
            data=film_summary, x=group_by, y=metric, 
            hue=group_by, order=unique_groups, hue_order=unique_groups,
            ax=ax, inner='box', palette=palette, alpha=0.6, linewidth=1, legend=False
        )
        
        # Individual points
        _sns.stripplot(
            data=film_summary, x=group_by, y=metric, order=unique_groups,
            ax=ax, color='#333333', alpha=0.6, size=4, jitter=True
        )
        
        # Statistical significance - compare adjacent groups
        if show_significance and len(unique_groups) >= 2:
            self._add_significance_annotations(
                ax, film_summary, metric, group_by, unique_groups
            )
        
        metric_label = get_feature_label(metric)
        ax.set_title(title or f'{metric_label} by {group_by.replace("_", " ").title()}')
        ax.set_ylabel(ylabel or metric_label)
        ax.set_xlabel(group_by.replace("_", " ").title())
        
        apply_publication_style(ax)
        _plt.tight_layout()
        filepath = self.output_dir / (filename or f'violin_{metric}_by_{group_by}.png')
        _plt.savefig(filepath, dpi=300, bbox_inches='tight')
        _plt.close()
        
        return str(filepath)
    
    def _add_significance_annotations(
        self, ax, data: pd.DataFrame, metric: str, group_by: str,
        groups: List[str], max_pairs: int = 6, correction: str = 'holm'
    ):
        """
        Add statistical significance annotations (*, **, ***, ns).
        
        For 2 groups: Direct comparison without correction.
        For 3+ groups: All pairwise comparisons with multiple comparison correction.
        
        Args:
            ax: Matplotlib axes
            data: DataFrame with data
            metric: Column to compare
            group_by: Grouping column
            groups: List of group names in order
            max_pairs: Maximum number of comparisons to show
            correction: Multiple comparison correction method ('holm', 'bonferroni', 'none')
        """
        from scipy import stats
        from itertools import combinations
        
        if len(groups) < 2:
            return
        
        y_max = data[metric].max()
        y_range = data[metric].max() - data[metric].min()
        if y_range == 0:
            y_range = y_max * 0.1 if y_max > 0 else 1
        
        # For 2 groups: direct comparison, no correction needed
        # For 3+ groups: all pairwise comparisons with correction
        if len(groups) == 2:
            pairs = [(0, 1)]
            apply_correction = False
        else:
            # All pairwise comparisons for 3+ groups
            pairs = [(groups.index(g1), groups.index(g2)) 
                     for g1, g2 in combinations(groups, 2)]
            apply_correction = correction != 'none'
        
        # Limit number of comparisons for visual clarity
        if len(pairs) > max_pairs:
            # Prioritize adjacent comparisons
            adjacent = [(i, i+1) for i in range(len(groups) - 1)]
            pairs = adjacent[:max_pairs]
            apply_correction = apply_correction and len(groups) > 2
        
        # Calculate p-values for all pairs
        p_values = []
        valid_pairs = []
        for idx1, idx2 in pairs:
            group1, group2 = groups[idx1], groups[idx2]
            
            data1 = data[data[group_by] == group1][metric].dropna()
            data2 = data[data[group_by] == group2][metric].dropna()
            
            if len(data1) < 2 or len(data2) < 2:
                continue
            
            # Mann-Whitney U test (non-parametric)
            try:
                _, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                p_values.append(p_value)
                valid_pairs.append((idx1, idx2))
            except:
                continue
        
        if not p_values:
            return
        
        # Apply multiple comparison correction for 3+ groups
        if apply_correction and len(p_values) > 1:
            p_values = self._correct_pvalues(p_values, correction)
        
        # Convert p-values to stars
        annotations = []
        for (idx1, idx2), p_value in zip(valid_pairs, p_values):
            if p_value > 0.05:
                sig_text = 'ns'
            elif p_value > 0.01:
                sig_text = '*'
            elif p_value > 0.001:
                sig_text = '**'
            elif p_value > 0.0001:
                sig_text = '***'
            else:
                sig_text = '****'
            
            annotations.append((idx1, idx2, sig_text, p_value))
        
        # Draw brackets and annotations
        bracket_height = y_range * 0.03
        y_offset = y_range * 0.08
        
        for level, (idx1, idx2, sig_text, p_value) in enumerate(annotations):
            y_bar = y_max + y_offset * (level + 1)
            
            # Draw bracket
            ax.plot([idx1, idx1, idx2, idx2], 
                   [y_bar - bracket_height, y_bar, y_bar, y_bar - bracket_height],
                   color='black', linewidth=1)
            
            # Add significance text
            x_mid = (idx1 + idx2) / 2
            ax.text(x_mid, y_bar + bracket_height * 0.5, sig_text, 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Adjust y-axis limit to accommodate annotations
        if annotations:
            new_ylim = y_max + y_offset * (len(annotations) + 1)
            ax.set_ylim(top=new_ylim)
    
    def _correct_pvalues(self, p_values: List[float], method: str) -> List[float]:
        """Apply multiple comparison correction to p-values."""
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
    
    def box_comparison(
        self,
        film_summary: pd.DataFrame,
        metrics: List[str],
        group_by: str,
        title: str = "Metrics Comparison",
        filename: str = "box_comparison.png"
    ) -> Optional[str]:
        """
        Generate grouped box plots for multiple metrics.
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
        
        metrics = [m for m in metrics if m in film_summary.columns]
        if not metrics:
            return None
        
        # Rename metrics for display
        rename_map = {m: get_feature_label(m) for m in metrics}
        plot_df = film_summary[[group_by] + metrics].copy()
        plot_df = plot_df.rename(columns=rename_map)
        
        # Melt data for grouped plotting
        df_melted = plot_df.melt(
            id_vars=[group_by],
            value_vars=[rename_map[m] for m in metrics],
            var_name='Metric',
            value_name='Value'
        )
        
        fig, ax = _plt.subplots(figsize=(12, 6))
        
        _sns.boxplot(
            data=df_melted, x='Metric', y='Value', hue=group_by,
            ax=ax, palette='Set2'
        )
        
        ax.set_title(title)
        ax.set_ylabel('Value')
        _plt.xticks(rotation=45, ha='right')
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def mean_ci_plot(
        self,
        film_summary: pd.DataFrame,
        metric: str,
        group_by: str,
        control_group: str = None,
        confidence: float = 0.95,
        title: str = None,
        filename: str = None,
        ylabel: str = None
    ) -> Optional[str]:
        """
        Generate Mean ± 95% CI plot (publication-standard format).
        
        Shows mean as point with confidence interval error bars.
        More suitable for publication than box plots in some contexts.
        
        Args:
            film_summary: DataFrame with film-level data
            metric: Column to plot
            group_by: Column for grouping
            control_group: Name of control group (shown in gray)
            confidence: Confidence level (default 0.95 for 95% CI)
            title: Plot title
            filename: Output filename
            ylabel: Y-axis label
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if metric not in film_summary.columns or group_by not in film_summary.columns:
            return None
        
        from scipy import stats
        
        unique_groups = sorted(film_summary[group_by].dropna().unique())
        
        # Calculate mean and CI for each group
        means = []
        ci_lower = []
        ci_upper = []
        group_sizes = []
        
        for group in unique_groups:
            data = film_summary[film_summary[group_by] == group][metric].dropna()
            if len(data) < 2:
                means.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
                group_sizes.append(len(data))
                continue
            
            n = len(data)
            mean = np.mean(data)
            se = stats.sem(data)
            t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin = t_val * se
            
            means.append(mean)
            ci_lower.append(mean - margin)
            ci_upper.append(mean + margin)
            group_sizes.append(n)
        
        # For hue metrics, use actual hue values as colors
        if is_hue_metric(metric):
            colors = []
            for group, mean in zip(unique_groups, means):
                if not np.isnan(mean):
                    colors.append(hue_to_rgb(mean))
                else:
                    colors.append(DEFAULT_GRAY)
        else:
            palette = get_palette(unique_groups, control_group)
            colors = [palette[g] for g in unique_groups]
        
        fig, ax = _plt.subplots(figsize=(max(8, len(unique_groups) * 1.5), 6))
        
        x_pos = np.arange(len(unique_groups))
        
        # Plot means with CI error bars
        for i, (group, mean, ci_lo, ci_hi, color) in enumerate(
            zip(unique_groups, means, ci_lower, ci_upper, colors)
        ):
            if np.isnan(mean):
                continue
            
            # Error bar
            ax.errorbar(
                i, mean, yerr=[[mean - ci_lo], [ci_hi - mean]],
                fmt='o', markersize=10, capsize=5, capthick=2,
                color=color, ecolor=color, elinewidth=2,
                markeredgecolor='black', markeredgewidth=1
            )
        
        # Add individual data points (jittered)
        for i, group in enumerate(unique_groups):
            data = film_summary[film_summary[group_by] == group][metric].dropna()
            jitter = np.random.uniform(-0.15, 0.15, len(data))
            ax.scatter(
                i + jitter, data, 
                c='#333333', alpha=0.4, s=20, zorder=1
            )
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(unique_groups)
        
        metric_label = get_feature_label(metric)
        ax.set_title(title or f'{metric_label} by {group_by.replace("_", " ").title()}')
        ax.set_ylabel(ylabel or f'{metric_label} (Mean ± {int(confidence*100)}% CI)')
        ax.set_xlabel(group_by.replace("_", " ").title())
        
        # Add sample sizes below x-axis labels
        for i, (group, n) in enumerate(zip(unique_groups, group_sizes)):
            ax.text(i, ax.get_ylim()[0], f'n={n}', ha='center', va='top', 
                   fontsize=8, color='#666666')
        
        apply_publication_style(ax)
        _plt.tight_layout()
        filepath = self.output_dir / (filename or f'mean_ci_{metric}_by_{group_by}.png')
        _plt.savefig(filepath, dpi=300, bbox_inches='tight')
        _plt.close()
        
        return str(filepath)
    
    def effect_size_forest_plot(
        self,
        statistical_results: Dict,
        metric: str = None,
        title: str = "Effect Size (Cohen's d) Forest Plot",
        filename: str = "effect_size_forest.png"
    ) -> Optional[str]:
        """
        Generate forest plot showing effect sizes (Cohen's d) for pairwise comparisons.
        
        Useful for visualizing the magnitude and direction of group differences.
        
        Args:
            statistical_results: Results from StatisticalAnalyzer.compare_multiple_groups()
            metric: Metric name for labeling (optional)
            title: Plot title
            filename: Output filename
        """
        if not HAS_MATPLOTLIB:
            return None
        
        pairwise = statistical_results.get('pairwise_comparisons', [])
        if not pairwise:
            return None
        
        # Extract effect sizes
        comparisons = []
        effect_sizes = []
        significances = []
        
        for pw in pairwise:
            if 'error' in pw:
                continue
            
            g1 = pw.get('group1_name', '?')
            g2 = pw.get('group2_name', '?')
            d = pw.get('cohens_d', 0)
            
            # Check significance (use corrected if available)
            if 'significant_corrected' in pw:
                is_sig = pw['significant_corrected']
            else:
                is_sig = pw.get('significant', False)
            
            comparisons.append(f'{g1} vs {g2}')
            effect_sizes.append(d)
            significances.append(is_sig)
        
        if not comparisons:
            return None
        
        fig, ax = _plt.subplots(figsize=(10, max(4, len(comparisons) * 0.6)))
        
        y_pos = np.arange(len(comparisons))
        
        # Color based on significance and direction
        # Use DEPOSIT_COLORS for consistency: green=normal(positive), red=rod(negative), gray=artifact(ns)
        colors = []
        for d, is_sig in zip(effect_sizes, significances):
            if not is_sig:
                colors.append(DEPOSIT_COLORS['artifact'])  # Gray for non-significant
            elif d > 0:
                colors.append(DEPOSIT_COLORS['normal'])  # Green for positive
            else:
                colors.append(DEPOSIT_COLORS['rod'])  # Red for negative
        
        # Horizontal bar chart
        bars = ax.barh(y_pos, effect_sizes, color=colors, alpha=0.7, 
                       edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (d, is_sig) in enumerate(zip(effect_sizes, significances)):
            label = f'{d:.2f}'
            if is_sig:
                label += ' *'
            x_pos = d + 0.05 if d >= 0 else d - 0.05
            ha = 'left' if d >= 0 else 'right'
            ax.text(x_pos, i, label, ha=ha, va='center', fontsize=9)
        
        # Reference lines for effect size interpretation
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(-0.2, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0.2, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(-0.8, color='gray', linestyle='-', alpha=0.5)
        ax.axvline(0.8, color='gray', linestyle='-', alpha=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(comparisons)
        ax.set_xlabel("Cohen's d")
        ax.set_title(title)
        
        # Add effect size interpretation legend
        legend_text = 'Effect size: |d|<0.2=negligible, 0.2-0.5=small, 0.5-0.8=medium, >0.8=large'
        ax.text(0.5, -0.12, legend_text, transform=ax.transAxes, 
               ha='center', fontsize=8, color='#666666')
        
        apply_publication_style(ax)
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath, dpi=300, bbox_inches='tight')
        _plt.close()
        
        return str(filepath)
    
    def heatmap(
        self,
        film_summary: pd.DataFrame,
        features: List[str] = None,
        row_label: str = 'filename',
        title: str = "Feature Heatmap",
        filename: str = "heatmap.png",
        sort_by: str = 'first_column'
    ) -> Optional[str]:
        """
        Generate heatmap of features across samples.
        
        Args:
            film_summary: DataFrame with film-level data
            features: Columns to include in heatmap
            row_label: Column for row labels
            title: Plot title
            filename: Output filename
            sort_by: Row sorting method - 'first_column', 'cluster', or 'original'
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
        
        if features is None:
            features = [
                'n_normal', 'n_rod', 'rod_fraction',
                'normal_mean_area', 'normal_mean_iod', 'total_iod'
            ]
        
        features = [f for f in features if f in film_summary.columns]
        if not features:
            return None
        
        # Prepare data
        df = film_summary[features].copy()
        
        # Standardize for visualization
        df_scaled = (df - df.mean()) / df.std()
        
        # Rename columns for display
        display_cols = [get_feature_label(f) for f in features]
        df_scaled.columns = display_cols
        
        # Set index
        if row_label in film_summary.columns:
            df_scaled.index = film_summary[row_label]
        
        # Sort rows
        if sort_by == 'first_column':
            # Sort by first column (descending for visual appeal)
            df_scaled = df_scaled.sort_values(by=display_cols[0], ascending=False)
        elif sort_by == 'cluster' and HAS_SKLEARN and len(df_scaled) > 2:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import pdist
            
            linkage_matrix = linkage(pdist(df_scaled.fillna(0)), method='ward')
            dendro = dendrogram(linkage_matrix, no_plot=True)
            df_scaled = df_scaled.iloc[dendro['leaves']]
        # else: 'original' - keep original order
        
        # Dynamic figure height based on row count
        fig_height = max(8, min(20, len(df_scaled) * 0.35))
        fig, ax = _plt.subplots(figsize=(12, fig_height))
        
        _sns.heatmap(
            df_scaled, ax=ax, cmap='RdBu_r', center=0,
            xticklabels=True, yticklabels=True,
            cbar_kws={'label': 'Z-score', 'shrink': 0.8},
            linewidths=0.5, linecolor='white'
        )
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        _plt.xticks(rotation=30, ha='right', fontsize=9)
        _plt.yticks(fontsize=8)
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath, dpi=300, bbox_inches='tight')
        _plt.close()
        
        return str(filepath)
    
    def scatter_matrix(
        self,
        film_summary: pd.DataFrame,
        features: List[str] = None,
        color_by: str = None,
        control_group: str = None,
        filename: str = "scatter_matrix.png"
    ) -> Optional[str]:
        """
        Generate scatter matrix (pair plot) of features.
        
        Args:
            film_summary: DataFrame with film-level data
            features: Columns to include in scatter matrix
            color_by: Column for color coding
            control_group: Name of control group (shown in gray)
            filename: Output filename
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
        
        if features is None:
            features = ['n_total', 'rod_fraction', 'total_iod', 'normal_mean_area']
        
        features = [f for f in features if f in film_summary.columns]
        if len(features) < 2:
            return None
        
        plot_df = film_summary[features].copy()
        
        # Rename columns for display
        rename_map = {f: get_feature_label(f) for f in features}
        plot_df = plot_df.rename(columns=rename_map)
        
        if color_by and color_by in film_summary.columns:
            plot_df[color_by] = film_summary[color_by]
            unique_groups = sorted(plot_df[color_by].dropna().unique())
            palette = get_palette(unique_groups, control_group)
            g = _sns.pairplot(
                plot_df, hue=color_by, palette=palette, 
                diag_kind='kde', plot_kws={'alpha': 0.7, 'edgecolor': 'white', 's': 50}
            )
        else:
            g = _sns.pairplot(
                plot_df, diag_kind='kde',
                plot_kws={'color': DEFAULT_GRAY, 'alpha': 0.7, 'edgecolor': 'white', 's': 50}
            )
        
        g.fig.suptitle('Feature Relationships', y=1.02, fontweight='bold')
        
        filepath = self.output_dir / filename
        g.savefig(filepath, dpi=300, bbox_inches='tight')
        _plt.close()
        
        return str(filepath)
    
    def area_iod_scatter(
        self,
        deposits_df: pd.DataFrame,
        color_by_label: bool = True,
        title: str = "Area vs IOD by Deposit Type",
        filename: str = "area_iod_scatter.png"
    ) -> Optional[str]:
        """
        Scatter plot of area vs IOD for individual deposits.
        
        Args:
            deposits_df: DataFrame with individual deposit data
            color_by_label: Whether to color by deposit label
            title: Plot title
            filename: Output filename
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = _plt.subplots(figsize=(10, 8))
        
        if color_by_label and 'label' in deposits_df.columns:
            for label in ['normal', 'rod', 'artifact']:
                mask = deposits_df['label'] == label
                if mask.any():
                    ax.scatter(
                        deposits_df.loc[mask, 'area_px'],
                        deposits_df.loc[mask, 'iod'],
                        c=DEPOSIT_COLORS.get(label, DEFAULT_GRAY),
                        label=label.capitalize(),
                        alpha=0.6, s=25, edgecolors='white', linewidth=0.3
                    )
            ax.legend(framealpha=0.9)
        else:
            ax.scatter(
                deposits_df['area_px'], deposits_df['iod'], 
                c=DEFAULT_GRAY, alpha=0.6, s=25, edgecolors='white', linewidth=0.3
            )
        
        ax.set_xlabel(get_feature_label('area_px'))
        ax.set_ylabel(get_feature_label('iod'))
        ax.set_title(title, fontweight='bold')
        
        apply_publication_style(ax)
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath, dpi=300, bbox_inches='tight')
        _plt.close()
        
        return str(filepath)
    
    def summary_dashboard(
        self,
        film_summary: pd.DataFrame,
        group_by: str = None,
        control_group: str = None,
        filename: str = "dashboard.png"
    ) -> Optional[str]:
        """
        Generate summary dashboard with multiple plots.
        
        Args:
            film_summary: DataFrame with film-level data
            group_by: Column for grouping
            control_group: Name of control group (shown in gray)
            filename: Output filename
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
        
        # Get unique groups and calculate optimal width
        if group_by and group_by in film_summary.columns:
            unique_groups = sorted(film_summary[group_by].dropna().unique())
            n_groups = len(unique_groups)
            palette = get_palette(unique_groups, control_group)
            
            # Box width: max 0.6, scales down with more groups
            box_width = min(0.6, 0.8 / max(1, n_groups / 4))
        else:
            unique_groups = []
            n_groups = 0
            palette = [DEFAULT_GRAY]
            box_width = 0.5
        
        # Dynamic figure size based on number of groups
        fig_width = max(12, min(16, n_groups * 1.2 + 8))
        fig, axes = _plt.subplots(2, 2, figsize=(fig_width, 10))
        
        # 1. ROD fraction distribution (Box plot)
        ax = axes[0, 0]
        if group_by and group_by in film_summary.columns:
            _sns.boxplot(
                data=film_summary, x=group_by, y='rod_fraction', 
                hue=group_by, order=unique_groups, hue_order=unique_groups,
                ax=ax, palette=palette, width=box_width,
                linewidth=1, fliersize=4, legend=False
            )
            _sns.stripplot(
                data=film_summary, x=group_by, y='rod_fraction',
                order=unique_groups, ax=ax, color='#333333', alpha=0.6, size=4
            )
        else:
            _sns.histplot(film_summary['rod_fraction'], ax=ax, kde=True, color=DEFAULT_GRAY)
        ax.set_title('ROD Fraction Distribution', fontweight='bold')
        ax.set_ylabel('ROD Fraction')
        ax.set_xlabel('')
        apply_publication_style(ax)
        
        # 2. Total deposits (Bar plot)
        ax = axes[0, 1]
        if group_by and group_by in film_summary.columns:
            _sns.barplot(
                data=film_summary, x=group_by, y='n_total',
                hue=group_by, order=unique_groups, hue_order=unique_groups,
                ax=ax, palette=palette, 
                errorbar='sd', width=box_width, edgecolor='black', linewidth=0.5, legend=False
            )
        else:
            _sns.histplot(film_summary['n_total'], ax=ax, kde=True, color=DEFAULT_GRAY)
        ax.set_title('Total Deposits per Image', fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_xlabel('')
        apply_publication_style(ax)
        
        # 3. Normal vs ROD counts (Grouped bar)
        ax = axes[1, 0]
        if 'n_normal' in film_summary.columns and 'n_rod' in film_summary.columns:
            plot_df = film_summary.melt(
                id_vars=[group_by] if group_by else [],
                value_vars=['n_normal', 'n_rod'],
                var_name='Type', value_name='Count'
            )
            plot_df['Type'] = plot_df['Type'].map({'n_normal': 'Normal', 'n_rod': 'ROD'})
            
            if group_by:
                _sns.barplot(
                    data=plot_df, x='Type', y='Count', hue=group_by,
                    hue_order=unique_groups, ax=ax, palette=palette, 
                    errorbar='sd', edgecolor='black', linewidth=0.5
                )
                ax.legend(title=group_by.replace('_', ' ').title(), 
                         fontsize=8, title_fontsize=9, framealpha=0.9)
            else:
                _sns.barplot(
                    data=plot_df, x='Type', y='Count', ax=ax, 
                    color=DEFAULT_GRAY, errorbar='sd', edgecolor='black', linewidth=0.5
                )
        ax.set_title('Deposit Counts by Type', fontweight='bold')
        ax.set_xlabel('')
        apply_publication_style(ax)
        
        # 4. Total IOD (Box plot)
        ax = axes[1, 1]
        if 'total_iod' in film_summary.columns:
            if group_by and group_by in film_summary.columns:
                _sns.boxplot(
                    data=film_summary, x=group_by, y='total_iod',
                    hue=group_by, order=unique_groups, hue_order=unique_groups,
                    ax=ax, palette=palette, width=box_width,
                    linewidth=1, fliersize=4, legend=False
                )
                _sns.stripplot(
                    data=film_summary, x=group_by, y='total_iod',
                    order=unique_groups, ax=ax, color='#333333', alpha=0.6, size=4
                )
            else:
                _sns.histplot(film_summary['total_iod'], ax=ax, kde=True, color=DEFAULT_GRAY)
        ax.set_title('Total IOD Distribution', fontweight='bold')
        ax.set_ylabel('Total IOD')
        ax.set_xlabel('')
        apply_publication_style(ax)
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath, dpi=300, bbox_inches='tight')
        _plt.close()
        
        return str(filepath)


def generate_all_visualizations(
    film_summary: pd.DataFrame,
    deposits_df: pd.DataFrame,
    output_dir: Path,
    group_by: str = None,
    control_group: str = None,
    show_significance: bool = True,
    statistical_results: Dict = None
) -> Dict[str, str]:
    """
    Generate all available visualizations.
    
    Args:
        film_summary: DataFrame with film-level data
        deposits_df: DataFrame with individual deposit data
        output_dir: Output directory for figures
        group_by: Column for grouping
        control_group: Name of control group (shown in gray)
        show_significance: Whether to show statistical significance on violin plots
        statistical_results: Results from StatisticalAnalyzer (for effect size plots)
    
    Returns:
        Dict mapping visualization name to filepath
    """
    viz = Visualizer(output_dir)
    results = {}
    
    # Dashboard
    path = viz.summary_dashboard(film_summary, group_by, control_group)
    if path:
        results['dashboard'] = path
    
    # PCA
    path = viz.pca_plot(film_summary, color_by=group_by, control_group=control_group)
    if path:
        results['pca'] = path
    
    # Heatmap
    path = viz.heatmap(film_summary)
    if path:
        results['heatmap'] = path
    
    # Violin plots for key metrics
    # Primary metrics (always generated)
    primary_metrics = ['rod_fraction', 'total_iod', 'n_total']
    # Secondary metrics (morphology, pH, pigment - important for biological interpretation)
    secondary_metrics = [
        'normal_mean_area', 'rod_mean_area',      # Size/morphology
        'normal_mean_hue', 'rod_mean_hue',        # pH proxy (Bromophenol Blue)
        'normal_mean_lightness', 'rod_mean_lightness',  # Pigment density
    ]
    
    for metric in primary_metrics + secondary_metrics:
        if metric in film_summary.columns and group_by:
            path = viz.violin_comparison(
                film_summary, metric, group_by, 
                control_group=control_group,
                show_significance=show_significance
            )
            if path:
                results[f'violin_{metric}'] = path
    
    # Mean + CI plots for publication-standard visualization
    # Include area and hue for biological significance
    mean_ci_metrics = [
        'rod_fraction', 'total_iod',
        'normal_mean_area', 'rod_mean_area',
        'normal_mean_hue', 'rod_mean_hue',
    ]
    for metric in mean_ci_metrics:
        if metric in film_summary.columns and group_by:
            path = viz.mean_ci_plot(
                film_summary, metric, group_by,
                control_group=control_group
            )
            if path:
                results[f'mean_ci_{metric}'] = path
    
    # Scatter matrix
    path = viz.scatter_matrix(film_summary, color_by=group_by, control_group=control_group)
    if path:
        results['scatter_matrix'] = path
    
    # Area vs IOD scatter
    if deposits_df is not None and len(deposits_df) > 0:
        path = viz.area_iod_scatter(deposits_df)
        if path:
            results['area_iod'] = path
    
    # Effect size forest plot (if statistical results provided)
    if statistical_results:
        # Generate forest plot for rod_fraction if available
        rod_stats = statistical_results.get('metrics', {}).get('rod_fraction', {})
        if rod_stats and 'pairwise_comparisons' in rod_stats:
            path = viz.effect_size_forest_plot(
                rod_stats,
                metric='rod_fraction',
                title="Effect Size (Cohen's d) - ROD Fraction"
            )
            if path:
                results['effect_size_rod_fraction'] = path
    
    return results


class SpatialVisualizer:
    """Generate spatial analysis visualizations."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def density_heatmap(
        self,
        density_map: np.ndarray,
        title: str = "Deposit Density Map",
        filename: str = "density_map.png"
    ) -> Optional[str]:
        """Generate density heatmap from spatial analysis."""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = _plt.subplots(figsize=(8, 8))
        
        im = ax.imshow(density_map, cmap='YlOrRd', interpolation='bilinear')
        _plt.colorbar(im, ax=ax, label='Deposit Density')
        
        ax.set_title(title)
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Y (grid)')
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def nnd_histogram(
        self,
        nnd_values: np.ndarray,
        mean_nnd: float = None,
        title: str = "Nearest Neighbor Distance Distribution",
        filename: str = "nnd_histogram.png"
    ) -> Optional[str]:
        """Generate NND histogram."""
        if not HAS_MATPLOTLIB or len(nnd_values) == 0:
            return None
        
        fig, ax = _plt.subplots(figsize=(10, 5))
        
        ax.hist(nnd_values, bins=30, color=DEFAULT_GRAY, edgecolor='white', alpha=0.8)
        
        if mean_nnd is not None:
            ax.axvline(mean_nnd, color='#E53935', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_nnd:.1f}')
            ax.legend()
        
        ax.set_xlabel('Nearest Neighbor Distance (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        
        apply_publication_style(ax)
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath, dpi=300, bbox_inches='tight')
        _plt.close()
        
        return str(filepath)
    
    def quadrant_plot(
        self,
        quadrant_counts: Dict[str, int],
        title: str = "Quadrant Distribution",
        filename: str = "quadrant_plot.png"
    ) -> Optional[str]:
        """Generate quadrant distribution plot."""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = _plt.subplots(figsize=(8, 8))
        
        # Create 2x2 grid
        data = np.array([
            [quadrant_counts['Q1'], quadrant_counts['Q2']],
            [quadrant_counts['Q3'], quadrant_counts['Q4']]
        ])
        
        im = ax.imshow(data, cmap='Blues')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{data[i, j]}', 
                              ha='center', va='center', fontsize=24, fontweight='bold')
        
        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Left', 'Right'])
        ax.set_yticklabels(['Top', 'Bottom'])
        ax.set_title(title)
        
        _plt.colorbar(im, ax=ax, label='Deposit Count')
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def spatial_scatter(
        self,
        centroids: np.ndarray,
        labels: List[str] = None,
        image_shape: Tuple[int, int] = None,
        title: str = "Deposit Spatial Distribution",
        filename: str = "spatial_scatter.png"
    ) -> Optional[str]:
        """Generate scatter plot of deposit locations."""
        if not HAS_MATPLOTLIB or len(centroids) == 0:
            return None
        
        fig, ax = _plt.subplots(figsize=(10, 10))
        
        if labels is not None:
            colors = {'normal': 'green', 'rod': 'red', 'artifact': 'gray', 'unknown': 'yellow'}
            for label in set(labels):
                mask = np.array(labels) == label
                if mask.any():
                    ax.scatter(
                        centroids[mask, 0], centroids[mask, 1],
                        c=colors.get(label, 'blue'),
                        label=label.capitalize(),
                        alpha=0.6, s=50
                    )
            ax.legend()
        else:
            ax.scatter(centroids[:, 0], centroids[:, 1], alpha=0.6, s=50)
        
        if image_shape:
            ax.set_xlim(0, image_shape[1])
            ax.set_ylim(image_shape[0], 0)  # Invert Y for image coordinates
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)
    
    def clark_evans_summary(
        self,
        r_values: List[float],
        interpretations: List[str],
        title: str = "Clark-Evans Clustering Index",
        filename: str = "clark_evans_summary.png"
    ) -> Optional[str]:
        """Generate summary of Clark-Evans R values."""
        if not HAS_MATPLOTLIB or len(r_values) == 0:
            return None
        
        fig, axes = _plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram of R values
        ax = axes[0]
        ax.hist(r_values, bins=20, color='#9C27B0', edgecolor='white', alpha=0.8)
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Random (R=1)')
        ax.axvline(np.mean(r_values), color='blue', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(r_values):.2f}')
        ax.set_xlabel('Clark-Evans R')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of R Values')
        ax.legend()
        
        # Interpretation pie chart
        ax = axes[1]
        interp_counts = {}
        for interp in interpretations:
            if 'cluster' in interp:
                key = 'Clustered'
            elif interp == 'random':
                key = 'Random'
            elif 'dispers' in interp:
                key = 'Dispersed'
            else:
                key = 'Other'
            interp_counts[key] = interp_counts.get(key, 0) + 1
        
        if interp_counts:
            colors = {'Clustered': '#F44336', 'Random': '#4CAF50', 
                     'Dispersed': '#2196F3', 'Other': '#9E9E9E'}
            ax.pie(
                interp_counts.values(), 
                labels=interp_counts.keys(),
                colors=[colors.get(k, 'gray') for k in interp_counts.keys()],
                autopct='%1.1f%%',
                startangle=90
            )
            ax.set_title('Clustering Interpretation')
        
        _plt.suptitle(title)
        _plt.tight_layout()
        filepath = self.output_dir / filename
        _plt.savefig(filepath)
        _plt.close()
        
        return str(filepath)


def generate_spatial_visualizations(
    spatial_results: List,  # List of SpatialResult
    output_dir: Path,
    deposits_by_image: Dict[str, Tuple[np.ndarray, List[str], Tuple[int, int]]] = None
) -> Dict[str, str]:
    """
    Generate all spatial visualizations.
    
    Args:
        spatial_results: List of SpatialResult objects
        output_dir: Output directory
        deposits_by_image: Dict mapping filename to (centroids, labels, image_shape)
    """
    viz = SpatialVisualizer(output_dir)
    results = {}
    
    if not spatial_results:
        return results
    
    # Aggregate NND values
    all_nnd = np.concatenate([r.nnd_values for r in spatial_results if len(r.nnd_values) > 0])
    if len(all_nnd) > 0:
        mean_nnd = np.mean([r.nnd_mean for r in spatial_results if r.nnd_mean > 0])
        path = viz.nnd_histogram(all_nnd, mean_nnd)
        if path:
            results['nnd_histogram'] = path
    
    # Clark-Evans summary
    r_values = [r.clark_evans_r for r in spatial_results if r.clark_evans_r > 0]
    interpretations = [r.clustering_interpretation for r in spatial_results 
                       if r.clustering_interpretation != 'insufficient_data']
    if r_values:
        path = viz.clark_evans_summary(r_values, interpretations)
        if path:
            results['clark_evans'] = path
    
    # Aggregate density map
    if spatial_results:
        agg_density = np.mean([r.density_map for r in spatial_results], axis=0)
        path = viz.density_heatmap(agg_density, title="Average Deposit Density")
        if path:
            results['density_map'] = path
    
    # Aggregate quadrant counts
    total_quadrants = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
    for r in spatial_results:
        for q, count in r.quadrant_counts.items():
            total_quadrants[q] += count
    
    path = viz.quadrant_plot(total_quadrants, title="Total Quadrant Distribution")
    if path:
        results['quadrant_plot'] = path
    
    return results
