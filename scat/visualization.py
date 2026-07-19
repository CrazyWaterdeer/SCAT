"""
Visualization module for SCAT.
Provides PCA, clustering, density plots, and comparison charts.
Publication-ready figures with GraphPad Prism-like styling.
"""

import warnings
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# =============================================================================
# Color Palettes
# =============================================================================

# -----------------------------------------------------------------------------
# Design system — the categorical palette + chart styling follow SCAT's sibling
# tool Imajin (colorblind-safe, min ΔE >= 17.7 across deuteranopia/protanopia;
# control = de-emphasised slate grey, conditions get colour). The one exception
# is pH: Bromophenol-Blue is a pH indicator (yellow acidic -> blue basic), so
# pH-specific views keep that acidic->basic gradient — voiced in the same
# palette's hues (gold -> teal-green -> steel-blue). Report CSS reuses these tokens.
# -----------------------------------------------------------------------------

# Neutrals (Imajin chart scheme: clean white ground, cool grey grid/text)
INK = '#1A1A1A'          # near-black — titles
MUTED = '#333333'        # dark grey — axis labels
FAINT = '#DDDDDD'        # grid / hairlines
PAPER = '#FFFFFF'        # figure ground

# Categorical palette for experimental groups — Imajin's colorblind-safe set,
# assigned in order (control/first = slate grey, conditions get colour). Last two
# extend it past 6 groups while staying distinct.
PASTEL_PALETTE = [
    '#636867',  # slate grey  (control / first)
    '#DA4E42',  # coral red
    '#2F6B9E',  # steel blue
    '#1F9E77',  # teal green
    '#DDA43A',  # gold
    '#C77BA9',  # mauve
    '#8C564B',  # brown       (extension)
    '#5C6BC0',  # indigo      (extension)
]

# Neutral slate grey for the control / ungrouped baseline and single-series fills
CONTROL_COLOR = '#636867'
DEFAULT_GRAY = '#636867'

# Deposit types — mapped onto the Imajin palette, kept semantically intuitive
# (normal = teal green, ROD = coral red/alert, artifact = neutral slate, unknown = gold)
DEPOSIT_COLORS = {
    'normal': '#1F9E77',
    'rod': '#DA4E42',
    'artifact': '#636867',
    'unknown': '#DDA43A'
}

_fonts_registered = False


def _register_fonts() -> None:
    """Register the bundled Noto Sans/Serif with matplotlib (same faces Imajin uses).
    Idempotent; silently degrades to DejaVu if the files are absent."""
    global _fonts_registered
    if _fonts_registered:
        return
    _fonts_registered = True
    try:
        from matplotlib import font_manager as fm
        font_dir = Path(__file__).resolve().parent / "assets" / "fonts"
        # Register every face — regular AND the Bold static instances — so bold titles render in Noto.
        for fp in sorted(font_dir.glob("*.ttf")):
            fm.fontManager.addfont(str(fp))
    except Exception:
        pass

# =============================================================================
# Feature Labels
# =============================================================================

# Feature label mapping for proper display
# Labels should be scientifically accurate and publication-ready
FEATURE_LABELS = {
    # Counts
    'n_total': 'Total Deposit Count',
    'n_deposits': 'Deposit Count',            # Normal + ROD (artifact-exclusive)
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
    'pigment_density': 'Pigment Density',
    'mean_pigment_density': 'Mean Pigment Density',
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


def normalize_palette_override(override, groups) -> Dict[str, str]:
    """Turn a caller color override into a {group_value: color} dict for `groups`.

    `override` may be a dict {group -> color} (keys matched to the raw group value OR its display
    label, so a caller may pass "WT" for "WT (driverless ctrl)"), or a LIST/tuple of colors applied
    in the given group order. Unknown keys / extra colors are ignored; empty/falsy colors skip.
    Colors are whatever matplotlib accepts (hex, name, rgb tuple) — validate upstream if you want to
    warn on typos."""
    if not override:
        return {}
    # Only keep colors matplotlib can actually paint, so a bad value persisted in a manifest or typed
    # by hand can never raise inside set_facecolor / seaborn — the group falls back to the default.
    try:
        from matplotlib.colors import is_color_like as _ok
    except Exception:
        _ok = lambda c: True
    if isinstance(override, (list, tuple)):
        return {g: c for g, c in zip(groups, override) if c and _ok(c)}
    if isinstance(override, dict):
        keyed = {str(k): c for k, c in override.items() if c and _ok(c)}
        out = {}
        for g in groups:
            if str(g) in keyed:
                out[g] = keyed[str(g)]
            elif _display_label(str(g)) in keyed:
                out[g] = keyed[_display_label(str(g))]
        return out
    return {}


def get_palette(groups: List[str], control_group: str = None, override=None) -> Dict[str, str]:
    """
    Get color palette for groups.

    Args:
        groups: List of group names
        control_group: Name of control group (will be gray)
        override: Optional caller color override (dict {group->color} or list in group order);
            an overridden group takes the given color, everything else keeps the default cycle.

    Returns:
        Dict mapping group names to colors
    """
    ov = normalize_palette_override(override, groups)
    palette = {}
    has_control = bool(control_group and control_group in groups)
    # When a control takes the slate grey (PASTEL_PALETTE[0]), cycle the conditions over the REST
    # of the palette so a large group set never wraps back onto the control's own colour.
    pool = PASTEL_PALETTE[1:] if has_control else PASTEL_PALETTE
    color_idx = 0

    for group in groups:
        if group in ov:
            palette[group] = ov[group]                      # explicit caller color wins
        elif has_control and group == control_group:
            palette[group] = CONTROL_COLOR
        else:
            palette[group] = pool[color_idx % len(pool)]
            color_idx += 1

    return palette


def guess_control_group(groups: List[str]) -> Optional[str]:
    """Best-effort: identify a control/reference group by name (control, ctrl, vehicle, WT, ...).
    Delegates to _is_control so significance-bracket control detection and group ORDERING can
    never disagree (previously two token sets had drifted — 'sham' matched one but not the other).
    Returns the first matching group or None — pass one explicitly to override."""
    for g in groups:
        if _is_control(g):
            return g
    return None


# Ordinal level words -> rank, for logical (not alphabetical) group ordering. Longest match wins.
_LEVEL_RANK = {
    'control': 0, 'ctrl': 0, 'vehicle': 0, 'untreated': 0, 'baseline': 0, 'mock': 0, 'sham': 0,
    'wildtype': 0, 'wt': 0,
    'minimum': 1, 'lowest': 1, 'verylow': 1, 'min': 1,
    'low': 2, 'mild': 2, 'lo': 2,
    'medium': 3, 'moderate': 3, 'intermediate': 3, 'mid': 3, 'med': 3,
    'high': 4, 'strong': 4, 'severe': 4, 'hi': 4,
    'maximum': 5, 'highest': 5, 'veryhigh': 5, 'max': 5,
}


def _ordinal_rank(name: str):
    """Rank of the longest ordinal level word contained in `name` (normalized), else None."""
    norm = re.sub(r'[^a-z]', '', str(name).lower())
    for word in sorted(_LEVEL_RANK, key=len, reverse=True):
        if word in norm:
            return _LEVEL_RANK[word]
    return None


def _numeric_key(name: str):
    """First number embedded in `name` (e.g. '10uM'->10, '6h'->6, '0.5'->0.5), else None. A leading
    '-' counts as a sign only when NOT preceded by an alnum, so a hyphen SEPARATOR ('Day-1','sample-10')
    is not read as a minus (which had reversed hyphen-numbered group order)."""
    m = re.search(r'(?:(?<![A-Za-z0-9])-)?\d+\.?\d*', str(name))
    return float(m.group()) if m else None


# Control detection for ordering. Distinctive substrings are safe to match anywhere; 'wt' is too
# short (growth/network), so it is matched as a whole word / prefix. Any group carrying one of these
# is treated as a control/reference and ordered before the non-control (treated/experimental) groups.
# These are generic biology control terms — nothing about specific genotypes/drugs/conditions is here.
_CONTROL_SUBSTR = ('control', 'ctrl', 'untreated', 'vehicle', 'baseline', 'mock', 'sham', 'wildtype')


def _is_control(name) -> bool:
    n = re.sub(r'[^a-z]', '', str(name).lower())
    if any(t in n for t in _CONTROL_SUBSTR):
        return True
    return any(w == 'wt' or w.startswith('wt') for w in re.split(r'[^a-z0-9]+', str(name).lower()))


def _control_role_rank(name) -> int:
    """Order WITHIN the control block by genetic role when the label declares one: the DRIVER control
    (Gal4/+) before the EFFECTOR control (UAS/+) — the standard Drosophila figure order (driver ctrl,
    effector ctrl, then the experimental cross). Generic controls keep their appearance order in
    between. These are universal Gal4/UAS binary-system roles, not experiment-specific genotypes;
    override with group_order / control_group when a run wants a different order."""
    n = str(name).lower()
    if 'driver' in n:
        return 0
    if 'effector' in n:
        return 2
    return 1


def _ordered_within(groups) -> List[str]:
    """Order same-kind groups by ordinal level word (low<mid<high) if they ALL carry one, else by an
    embedded number (dose/temperature/time) if they ALL carry one, else keep the order they were
    defined (appearance). No experiment-specific assumptions."""
    gs = list(groups)
    if len(gs) > 1:
        if all(_ordinal_rank(g) is not None for g in gs):
            gs = sorted(gs, key=lambda g: (_ordinal_rank(g), g))
        elif all(_numeric_key(g) is not None for g in gs):
            gs = sorted(gs, key=_numeric_key)
    return gs


def order_groups(values, control_group: str = None, explicit_order=None) -> List[str]:
    """Order groups for display by LOGICAL structure, never plain alphabetical (which scrambles
    Low/Mid/High and 2/10/100 and puts 'treated' before 'untreated'). Control/reference groups come
    first (an explicit control_group leads; the remaining controls keep the order they were defined),
    then the non-control groups by ordinal level word (low<mid<high), else an embedded number
    (dose/temperature/time), else the order they were defined.

    Group names are arbitrary: NOTHING about specific genotypes, drugs, temperatures or conditions is
    hard-coded — only two generic signals are inferred, 'looks like a control' and 'carries a level
    word or a number'. Everything else keeps the experimenter's defined order. Pass control_group to
    pin a reference explicitly.

    ``explicit_order`` fully overrides the logic: the caller-supplied order wins for every name it
    lists (in that exact order), and any present value it omits is appended afterwards via the normal
    logical ordering. It is matched leniently — a caller may pass the display label WITHOUT the
    trailing '(…)' note (see _display_label), and it still matches the full group value. Unknown names
    in the list are ignored, so a stale/typo'd order never drops real groups."""
    seen = list(dict.fromkeys(str(v) for v in values))          # appearance (defined) order, de-duped
    if explicit_order:
        # Map both the raw value and its display-stripped form to the actual group value, so the
        # caller can order by either. First match wins; later duplicates are ignored.
        lookup = {}
        for g in seen:
            lookup.setdefault(g, g)
            lookup.setdefault(_display_label(g), g)
        lead_list, used = [], set()
        for name in explicit_order:
            g = lookup.get(str(name)) or lookup.get(_display_label(str(name)))
            if g is not None and g not in used:
                lead_list.append(g)
                used.add(g)
        rest = [g for g in seen if g not in used]
        return lead_list + order_groups(rest, control_group)     # leftovers keep logical order
    lead = str(control_group) if control_group is not None and str(control_group) in seen else None
    # Controls first; within them, driver control before effector control (stable sort keeps generic
    # controls in appearance order between them).
    controls = sorted((g for g in seen if g != lead and _is_control(g)), key=_control_role_rank)
    rest = [g for g in seen if g != lead and g not in controls]
    return ([lead] if lead else []) + controls + _ordered_within(rest)


def _display_label(name) -> str:
    """A clean axis/legend label: drop a trailing parenthetical note the agent may append to a group
    for its OWN recognition (e.g. '21_+trpA1 (driverless ctrl)' -> '21_+trpA1'). The full label is
    still used for grouping/ordering/detection; only what a graph prints is trimmed."""
    cleaned = re.sub(r'\s*\([^)]*\)\s*$', '', str(name)).strip()
    return cleaned or str(name)


def set_group_xticklabels(ax, groups, positions=None) -> None:
    """Label the x-axis with group names, cleaned for display (_display_label) and ROTATED ~30 deg
    when the labels are many or long, so long condition names don't overflow / overlap horizontally."""
    labels = [_display_label(g) for g in groups]
    if positions is not None:
        ax.set_xticks(list(positions))
    longest = max((len(s) for s in labels), default=0)
    crowded = len(labels) >= 6 or longest >= 12 or sum(len(s) for s in labels) >= 42
    if crowded:
        ax.set_xticklabels(labels, rotation=30, ha='right')
    else:
        ax.set_xticklabels(labels, rotation=0, ha='center')


def draw_condition_matrix(ax, x_positions, matrix, groups, color: str = None) -> int:
    """Draw an open/closed-circle CONDITION MATRIX beneath a categorical plot's x-axis — the
    molecular-biology design table (as in Western-blot lane annotations). One row per experimental
    factor, one column per group; a **filled ● = the factor is present** for that group, an
    **open ○ = absent**. Replaces the x tick labels (the factor rows identify the columns).

    matrix: {factor_name: {group: truthy}}  (or {factor_name: [truthy per group, in `groups` order]}).
    Returns the number of factor rows drawn (0 if no matrix). `groups` gives the column order.
    """
    if not matrix:
        return 0
    import matplotlib.transforms as mtransforms
    color = color or MUTED
    xs = [float(x) for x in x_positions]
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    row_h, y0 = 0.095, -0.14
    label_x = (min(xs) - 0.7) if xs else -0.7
    for r, (factor, levels) in enumerate(matrix.items()):
        y = y0 - r * row_h
        for c, g in enumerate(groups):
            if isinstance(levels, dict):
                on = bool(levels.get(g, False))
            else:
                on = bool(levels[c]) if c < len(levels) else False
            ax.plot(xs[c], y, marker='o', markersize=9, transform=trans, clip_on=False, zorder=6,
                    markerfacecolor=(color if on else 'white'),
                    markeredgecolor=color, markeredgewidth=1.3)
        ax.text(label_x, y, str(factor), transform=trans, ha='right', va='center',
                fontsize=10, color=color, clip_on=False)
    ax.set_xticklabels([])       # the factor rows label the columns now
    ax.tick_params(axis='x', length=0)
    return len(matrix)


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
    # Thin spines + ticks, horizontal-only grey grid behind the data (after Imajin's _style_axes)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color(MUTED)
    ax.tick_params(width=0.8, length=3, colors=MUTED)
    ax.grid(True, axis='y', color=FAINT, alpha=0.8, linewidth=0.5)
    ax.grid(False, axis='x')
    ax.set_axisbelow(True)

# Lazy loading flags
_viz_libs_loaded = False
_plt = None
_mpatches = None
_to_rgba = None
_sns = None
_PCA = None
_StandardScaler = None

HAS_MATPLOTLIB = False
HAS_SEABORN = False
HAS_SKLEARN = False


def _load_viz_libs():
    """Lazy load visualization libraries."""
    global _viz_libs_loaded, _plt, _mpatches, _to_rgba, _sns
    global _PCA, _StandardScaler
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
        _PCA = PCA
        _StandardScaler = StandardScaler
        HAS_SKLEARN = True
    except ImportError:
        pass
    
    _viz_libs_loaded = True


class Visualizer:
    """Generate publication-ready visualizations for excreta analysis."""
    
    def __init__(self, output_dir: Path, style: str = 'whitegrid', group_order=None, palette=None):
        _load_viz_libs()  # Lazy load visualization libraries

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Optional caller-supplied group display order (overrides the logical order_groups()
        # heuristic); None keeps the automatic control-first / low<mid<high ordering.
        self.group_order = list(group_order) if group_order else None
        # Optional caller color override (dict {group->color} or list in group order); None keeps
        # the default Imajin categorical palette. Normalized per-plot against the actual groups.
        self.palette = palette
        
        if HAS_MATPLOTLIB and HAS_SEABORN:
            _register_fonts()                  # bundled Noto Sans/Serif (fallback: DejaVu)
            _sns.set_style('white')            # clean white ground; grid supplied via rcParams
            _sns.set_palette(PASTEL_PALETTE)   # Imajin categorical set as the seaborn default
            # Chart theme after Imajin: white ground, cool-grey y-grid, thin spines, Noto type.
            _plt.rcParams.update({
                'figure.figsize': (10, 8),
                'figure.dpi': 150,
                'figure.facecolor': PAPER,
                'axes.facecolor': PAPER,
                'savefig.facecolor': PAPER,
                'savefig.bbox': 'tight',
                'font.family': 'sans-serif',
                'font.sans-serif': ['Noto Sans', 'DejaVu Sans'],
                'font.serif': ['Noto Serif', 'DejaVu Serif'],
                'font.size': 11,
                'text.color': MUTED,
                'axes.titlesize': 13,
                'axes.titleweight': 'bold',
                'axes.titlecolor': INK,
                'axes.titlepad': 12,
                'axes.titlelocation': 'left',
                'axes.labelsize': 11,
                'axes.labelcolor': MUTED,
                'axes.edgecolor': MUTED,
                'axes.linewidth': 0.8,
                'xtick.color': MUTED,
                'ytick.color': MUTED,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'xtick.major.width': 0.8,
                'ytick.major.width': 0.8,
                'legend.fontsize': 10,
                'legend.frameon': False,
                'legend.title_fontsize': 10,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.grid': True,
                'axes.grid.axis': 'y',         # horizontal reference lines only (Imajin)
                'grid.color': FAINT,
                'grid.linewidth': 0.5,
                'grid.alpha': 0.8,
                'axes.axisbelow': True,
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
            unique_groups = order_groups(groups.unique(), explicit_order=self.group_order)
            
            # Get palette with control group support
            palette = get_palette(unique_groups, control_group, override=self.palette)
            
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
    
    def violin_comparison(
        self,
        film_summary: pd.DataFrame,
        metric: str,
        group_by: str,
        control_group: str = None,
        show_significance: bool = False,
        significance_mode: str = 'auto',
        show_ns: bool = False,
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
        unique_groups = order_groups(film_summary[group_by].dropna().unique(), explicit_order=self.group_order)
        
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
            palette = get_palette(unique_groups, control_group, override=self.palette)
        
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
            ax=ax, color='#333333', alpha=0.65, size=7, edgecolor='white', linewidth=0.6, jitter=True
        )
        
        # Statistical significance - compare adjacent groups
        if show_significance and len(unique_groups) >= 2:
            self._add_significance_annotations(
                ax, film_summary, metric, group_by, unique_groups,
                mode=significance_mode, control_group=control_group, show_ns=show_ns
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

    def condition_comparison(
        self,
        film_summary: pd.DataFrame,
        metric: str,
        group_by: str,
        condition_matrix: dict,
        control_group: str = None,
        show_significance: bool = False,
        significance_mode: str = 'auto',
        show_ns: bool = False,
        title: str = None,
        filename: str = None,
        ylabel: str = None
    ) -> Optional[str]:
        """Bar (mean ± SEM) + individual points per group, with an open/closed-circle CONDITION
        MATRIX beneath the axis — the factorial-design table (filled ● = factor present for that
        group, open ○ = absent). condition_matrix: {factor_name: {group: truthy}}. Groups are placed
        in logical order (control first, then low<mid<high / numeric)."""
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None
        if metric not in film_summary.columns or group_by not in film_summary.columns:
            return None

        groups = order_groups(film_summary[group_by].dropna().unique(), control_group, explicit_order=self.group_order)
        if not groups:
            return None
        ctrl = control_group if control_group in groups else guess_control_group(groups)

        if is_hue_metric(metric):
            palette = {}
            for g in groups:
                mh = film_summary[film_summary[group_by] == g][metric].mean()
                palette[g] = hue_to_rgb(mh) if not np.isnan(mh) else DEFAULT_GRAY
        else:
            palette = get_palette(groups, ctrl, override=self.palette)

        x = np.arange(len(groups))
        means, sems, per_group = [], [], []
        for g in groups:
            vals = film_summary[film_summary[group_by] == g][metric].dropna()
            per_group.append(vals.values)
            means.append(float(vals.mean()) if len(vals) else 0.0)
            sems.append(float(vals.sem()) if len(vals) > 1 else 0.0)

        n_factors = len(condition_matrix) if condition_matrix else 0
        fig, ax = _plt.subplots(figsize=(max(6, len(groups) * 1.4), 6 + 0.55 * n_factors))
        ax.bar(x, means, yerr=sems, color=[palette[g] for g in groups], alpha=0.9,
               edgecolor='#333333', linewidth=0.8, width=0.62, capsize=4,
               error_kw={'elinewidth': 1.0, 'ecolor': '#333333'}, zorder=2)
        rng = np.random.RandomState(0)
        for i, vals in enumerate(per_group):
            if len(vals):
                ax.scatter(i + rng.uniform(-0.13, 0.13, len(vals)), vals, s=34, color='#333333',
                           alpha=0.6, edgecolors='white', linewidth=0.5, zorder=3)

        set_group_xticklabels(ax, groups, positions=x)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        metric_label = get_feature_label(metric)
        ax.set_title(title or f'{metric_label} by {group_by.replace("_", " ").title()}', fontweight='bold')
        ax.set_ylabel(ylabel or metric_label)
        apply_publication_style(ax)

        if show_significance and len(groups) >= 2:
            self._add_significance_annotations(ax, film_summary, metric, group_by, groups,
                                               mode=significance_mode, control_group=ctrl, show_ns=show_ns)
        # Condition matrix beneath the axis (open/closed circles); replaces the x tick labels.
        draw_condition_matrix(ax, x, condition_matrix, groups)

        _plt.tight_layout()
        filepath = self.output_dir / (filename or f'condition_{metric}_by_{group_by}.png')
        _plt.savefig(filepath, dpi=300, bbox_inches='tight')
        _plt.close()
        return str(filepath)

    def _add_significance_annotations(
        self, ax, data: pd.DataFrame, metric: str, group_by: str,
        groups: List[str], mode: str = 'auto', control_group: str = None,
        show_ns: bool = False, max_pairs: int = 8, correction: str = 'holm'
    ):
        """
        Draw significance brackets for the comparisons that `mode` selects — NOT every pair by default.
        Which comparisons to bracket is an experimental-design choice; see the agent guidance in
        prompts.py. This method just executes the chosen policy.

        mode:
          'none'       — no brackets (rely on the omnibus test reported elsewhere).
          'vs_control' — each non-control group vs the control (Dunnett-style, k-1 comparisons).
          'adjacent'   — consecutive groups only (ordered designs: dose / time series).
          'pairwise'   — every pair (only sensible for a few groups; capped at max_pairs).
          'auto'       — 2 groups: the one pair; 3+ with a resolvable control: 'vs_control';
                         3+ without a control: 'none' (the omnibus result carries the message).
        show_ns: also draw non-significant ('ns') brackets (default False — show only significant).
        """
        from scipy import stats
        from itertools import combinations

        if len(groups) < 2 or mode == 'none':
            return

        y_max = data[metric].max()
        y_range = data[metric].max() - data[metric].min()
        if y_range == 0:
            y_range = y_max * 0.1 if y_max > 0 else 1

        ctrl = control_group if control_group in groups else guess_control_group(groups)
        resolved = mode
        if mode == 'auto':
            resolved = 'pairwise' if len(groups) == 2 else ('vs_control' if ctrl else 'none')
        if resolved == 'none':
            return
        if resolved == 'vs_control' and not ctrl:
            resolved = 'pairwise'  # asked for vs-control but none found -> fall back rather than nothing

        idx = {g: i for i, g in enumerate(groups)}
        if resolved == 'vs_control':
            pairs = [(idx[ctrl], idx[g]) for g in groups if g != ctrl]
        elif resolved == 'adjacent':
            pairs = [(i, i + 1) for i in range(len(groups) - 1)]
        else:  # pairwise
            pairs = [(idx[g1], idx[g2]) for g1, g2 in combinations(groups, 2)]

        if len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]
        apply_correction = correction != 'none' and len(pairs) > 1
        
        # Calculate p-values for all pairs
        p_values = []
        valid_pairs = []
        for idx1, idx2 in pairs:
            group1, group2 = groups[idx1], groups[idx2]
            
            data1 = data[data[group_by] == group1][metric].dropna()
            data2 = data[data[group_by] == group2][metric].dropna()

            if len(data1) < 3 or len(data2) < 3:   # match the stats' n>=3 significance gate
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
        
        # Apply multiple comparison correction for 3+ groups (canonical impl in statistics — this file
        # used to carry a drifted Holm copy that mis-restored the comparison order).
        if apply_correction and len(p_values) > 1:
            from .statistics import correct_pvalues
            p_values = correct_pvalues(p_values, correction)
        
        # Convert p-values to stars; drop non-significant brackets unless show_ns
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

            if sig_text == 'ns' and not show_ns:
                continue
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
        
        unique_groups = order_groups(film_summary[group_by].dropna().unique(), explicit_order=self.group_order)
        
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
            palette = get_palette(unique_groups, control_group, override=self.palette)
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
        
        # Add individual data points (jittered). Seed a LOCAL RNG so the saved mean_ci_* PNGs are
        # reproducible run-to-run (mirrors condition_comparison) without perturbing global RNG state.
        rng = np.random.RandomState(0)
        for i, group in enumerate(unique_groups):
            data = film_summary[film_summary[group_by] == group][metric].dropna()
            jitter = rng.uniform(-0.15, 0.15, len(data))
            ax.scatter(
                i + jitter, data, 
                c='#333333', alpha=0.5, s=45, zorder=1, edgecolors='white', linewidth=0.5
            )
        
        set_group_xticklabels(ax, unique_groups, positions=x_pos)

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
            unique_groups = order_groups(plot_df[color_by].dropna().unique(), explicit_order=self.group_order)
            palette = get_palette(unique_groups, control_group, override=self.palette)
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
                        alpha=0.6, s=34, edgecolors='white', linewidth=0.4
                    )
            ax.legend(framealpha=0.9)
        else:
            ax.scatter(
                deposits_df['area_px'], deposits_df['iod'], 
                c=DEFAULT_GRAY, alpha=0.6, s=34, edgecolors='white', linewidth=0.4
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

        # Deposit count = Normal + ROD (artifact-exclusive), matching the report/stats — and per fly
        # when every image has a valid n_flies (fly_normalize divides the count/sum columns).
        from scat.metrics import fly_normalize
        film_summary, _dash_per_fly = fly_normalize(film_summary)
        count_col = 'n_deposits' if 'n_deposits' in film_summary.columns else 'n_total'
        _cnt_word = 'per fly' if _dash_per_fly else 'per Image'
        _iod_word = 'IOD / fly' if _dash_per_fly else 'Total IOD'

        # Get unique groups and calculate optimal width
        if group_by and group_by in film_summary.columns:
            unique_groups = order_groups(film_summary[group_by].dropna().unique(), explicit_order=self.group_order)
            n_groups = len(unique_groups)
            palette = get_palette(unique_groups, control_group, override=self.palette)
            
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
                order=unique_groups, ax=ax, color='#333333', alpha=0.65, size=7, edgecolor='white', linewidth=0.6
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
                data=film_summary, x=group_by, y=count_col,
                hue=group_by, order=unique_groups, hue_order=unique_groups,
                ax=ax, palette=palette,
                errorbar='sd', width=box_width, edgecolor='black', linewidth=0.5, legend=False
            )
        else:
            _sns.histplot(film_summary[count_col], ax=ax, kde=True, color=DEFAULT_GRAY)
        ax.set_title(f'Deposits {_cnt_word}', fontweight='bold')
        ax.set_ylabel('Deposits / fly' if _dash_per_fly else 'Count')
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
                    order=unique_groups, ax=ax, color='#333333', alpha=0.65, size=7, edgecolor='white', linewidth=0.6
                )
            else:
                _sns.histplot(film_summary['total_iod'], ax=ax, kde=True, color=DEFAULT_GRAY)
        ax.set_title(f'{_iod_word} Distribution', fontweight='bold')
        ax.set_ylabel(_iod_word)
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
    significance_mode: str = 'auto',
    show_ns: bool = False,
    condition_matrix: dict = None,
    group_order=None,
    palette=None
) -> Dict[str, str]:
    """
    Generate all available visualizations.

    Args:
        film_summary: DataFrame with film-level data
        deposits_df: DataFrame with individual deposit data
        output_dir: Output directory for figures
        group_by: Column for grouping
        control_group: Name of control group (shown in gray)
        show_significance: Whether to draw significance brackets on violin plots at all
        significance_mode: Which comparisons to bracket — 'auto'|'vs_control'|'adjacent'|'pairwise'|'none'
            (see the agent guidance in prompts.py; 'auto' avoids all-pairwise clutter)
        show_ns: Also draw non-significant brackets (default False)
        group_order: Optional explicit group display order (overrides the logical auto-order)
        palette: Optional caller color override (dict {group->color} or list in group order)

    Returns:
        Dict mapping visualization name to filepath

    Each plot is generated under its own guard: a single plot that fails (e.g. a seaborn/matplotlib
    quirk on one metric) is skipped with a warning and never aborts the rest of the figures.
    """
    viz = Visualizer(output_dir, group_order=group_order, palette=palette)
    results = {}

    def _safe(name, fn):
        """Run one plot builder; on any failure log + skip so the other figures still render."""
        try:
            path = fn()
        except Exception as e:
            warnings.warn(f"visualization '{name}' skipped: {type(e).__name__}: {e}")
            return
        if path:
            results[name] = path

    # Derive the artifact-exclusive deposit count (Normal+ROD) and, when every image has a valid
    # n_flies, divide the count/sum columns PER FLY — so the standalone plots agree with the
    # report/stats. Never written to disk. per_fly drives the count/IOD axis labels below.
    from scat.metrics import fly_normalize
    film_summary, per_fly = fly_normalize(film_summary)
    _count_metric = 'n_deposits' if 'n_deposits' in film_summary.columns else 'n_total'
    # Per-fly axis labels for the count/sum metrics (values are already divided); others unchanged.
    _pf_label = {_count_metric: 'Deposits / fly', 'total_iod': 'IOD / fly'} if per_fly else {}
    def _ylabel(metric):
        return _pf_label.get(metric)   # None -> the plotter's default get_feature_label

    # Dashboard
    _safe('dashboard', lambda: viz.summary_dashboard(film_summary, group_by, control_group))
    # PCA
    _safe('pca', lambda: viz.pca_plot(film_summary, color_by=group_by, control_group=control_group))
    # Heatmap
    _safe('heatmap', lambda: viz.heatmap(film_summary))

    # Violin plots for key metrics
    # Primary metrics (always generated); deposit count is artifact-exclusive (Normal+ROD).
    primary_metrics = ['rod_fraction', 'total_iod', _count_metric]
    # Secondary metrics (morphology, pH, pigment - important for biological interpretation)
    secondary_metrics = [
        'normal_mean_area', 'rod_mean_area',      # Size/morphology
        'normal_mean_hue', 'rod_mean_hue',        # pH proxy (Bromophenol Blue)
        'normal_mean_lightness', 'rod_mean_lightness',  # Pigment density
    ]

    for metric in primary_metrics + secondary_metrics:
        if metric in film_summary.columns and group_by:
            _safe(f'violin_{metric}', lambda m=metric: viz.violin_comparison(
                film_summary, m, group_by,
                control_group=control_group,
                show_significance=show_significance,
                significance_mode=significance_mode,
                show_ns=show_ns, ylabel=_ylabel(m)
            ))

    # Condition-matrix bar charts (open/closed circles) for factorial designs, when the caller
    # supplies the design table. One per primary metric.
    if condition_matrix and group_by:
        for metric in primary_metrics:
            if metric in film_summary.columns:
                _safe(f'condition_{metric}', lambda m=metric: viz.condition_comparison(
                    film_summary, m, group_by, condition_matrix,
                    control_group=control_group, show_significance=show_significance,
                    significance_mode=significance_mode, show_ns=show_ns, ylabel=_ylabel(m)))

    # Mean + CI plots for publication-standard visualization
    # Include area and hue for biological significance
    mean_ci_metrics = [
        'rod_fraction', 'total_iod',
        'normal_mean_area', 'rod_mean_area',
        'normal_mean_hue', 'rod_mean_hue',
    ]
    for metric in mean_ci_metrics:
        if metric in film_summary.columns and group_by:
            _safe(f'mean_ci_{metric}', lambda m=metric: viz.mean_ci_plot(
                film_summary, m, group_by,
                control_group=control_group, ylabel=_ylabel(m)
            ))

    # Scatter matrix
    _safe('scatter_matrix', lambda: viz.scatter_matrix(film_summary, color_by=group_by, control_group=control_group))

    # Area vs IOD scatter
    if deposits_df is not None and len(deposits_df) > 0:
        _safe('area_iod', lambda: viz.area_iod_scatter(deposits_df))

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
