"""
Report generation module for SCAT.
Generates HTML and PDF reports from analysis results.
"""

import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from io import BytesIO

# Check for optional dependencies
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import color constants from visualization for consistency
from .visualization import (
    DEFAULT_GRAY, PASTEL_PALETTE, DEPOSIT_COLORS,
    apply_publication_style, hue_to_rgb, order_groups
)
from . import __version__


# CSS design tokens shared with the plots (Bromophenol-Blue pH axis, warm
# neutrals, teal accent). Kept as a module constant so _build_html stays legible.
_REPORT_CSS = """\
        /* Design tokens — shared story with the plots: Bromophenol-Blue pH axis
           (acidic amber -> basic teal) as the signature, warm neutrals, teal accent. */
        :root {
            --paper: #FAFAF9;
            --surface: #FFFFFF;
            --ink: #1A1A1A;
            --muted: #5A5A5A;
            --hair: #E4E3E0;
            --rule: #C9C7C3;            /* neutral structural rule (plot-caption left-borders) */
            --accent: #2F6B9E;          /* focus affordance only — chrome is neutral, blue is the pH-basic DATA axis */
            --wash: #F1F0EC;            /* neutral table-header / row-hover wash (was a steel-blue tint) */
            --ph-acidic: #DDA43A;       /* Bromophenol-Blue axis (pH views only) */
            --ph-mid: #1F9E77;
            --ph-basic: #2F6B9E;
            --normal: #1F9E77;
            --rod: #DA4E42;
            --artifact: #636867;
            --warn-bg: #FBF1DC;
            --warn-line: #DDA43A;
            --serif: 'Noto Serif', Georgia, 'Times New Roman', serif;
            --sans: 'Noto Sans', ui-sans-serif, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;
            /* promoted from scattered inline styles so the report speaks one palette */
            --fill: #F1F0ED;            /* warm inset panel */
            --ok-bg: #E7F2EC;           /* significance callout background */
            --ok-line: #1F9E77;         /* = --normal */
            --ok-ink: #176B4E;          /* callout heading */
            --rod-text: #C4453B;        /* rod hue at small text — 4.7:1 on paper */
            --normal-text: #17805F;     /* normal hue at small text — 4.7:1 on paper */
            --track-caps: 0.06em;       /* single uppercase-caption tracking */
            /* motion — strong custom curves, all durations < 400ms */
            --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
            --ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
            --dur-hover: 150ms;
            --dur-reveal: 360ms;
            --reveal-stagger: 60ms;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: var(--sans);
            line-height: 1.65;
            color: var(--ink);
            background: var(--paper);
            max-width: 940px;
            margin: 0 auto;
            padding: 48px 28px 72px;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        .header { margin-bottom: 40px; }
        .header::before {
            content: "";
            display: block;
            height: 4px;
            border-radius: 2px;
            background: linear-gradient(90deg, var(--ph-acidic), var(--ph-mid), var(--ph-basic));
            margin-bottom: 24px;
        }
        .header h1 {
            font-family: var(--serif);
            font-size: 2.1rem;
            font-weight: 600;
            letter-spacing: -0.01em;
            text-wrap: balance;
        }
        .header .subtitle {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 10px;
            text-transform: uppercase;
            letter-spacing: var(--track-caps);
        }
        .section {
            background: var(--surface);
            border: 1px solid var(--hair);
            border-radius: 10px;
            padding: 28px 30px;
            margin-bottom: 22px;
        }
        .section h2 {
            font-family: var(--serif);
            font-size: 1.35rem;
            font-weight: 600;
            letter-spacing: -0.008em;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--hair);
        }
        .section h3 {
            font-family: var(--serif);
            font-size: 1.05rem;
            font-weight: 600;
            letter-spacing: -0.005em;
            color: var(--ink);             /* structure, not color — matches the app's neutral section headings */
            margin-top: 28px;
            margin-bottom: 12px;
        }
        .section-intro { color: var(--muted); margin-bottom: 20px; }
        .appendix-ref { color: var(--muted); font-size: 0.82rem; margin-left: 10px; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
            gap: 16px;
            margin-bottom: 8px;
        }
        .stat-card {
            background: var(--paper);
            padding: 20px;
            border-radius: 8px;
            border: 1px solid var(--hair);
            text-align: left;
        }
        .stat-card .value {
            font-size: 1.9rem;
            font-weight: 700;
            color: var(--ink);              /* numbers are ink; color is reserved for meaning */
            font-variant-numeric: tabular-nums;
            letter-spacing: -0.01em;        /* size-specific tightening, like the serif headings */
            line-height: 1.1;
        }
        .stat-card .label {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 6px;
            min-height: 2.1em;              /* reserve two lines so every card's value shares a baseline */
            text-transform: uppercase;
            letter-spacing: var(--track-caps);
        }
        .stat-card .label .lc { text-transform: none; }   /* keep domain casing (pH, not PH) */
        .stat-card.normal .value { color: var(--normal); }
        .stat-card.rod .value { color: var(--rod); }
        .stat-card.artifact .value { color: var(--artifact); }
        .lede{border-left:4px solid var(--rod);background:var(--surface);border:1px solid var(--hair);border-radius:8px;padding:22px 24px;margin:20px 0}
        .finding{font-family:var(--serif);font-size:1.4rem;font-weight:600;line-height:1.35}
        .lede-trio{display:flex;gap:32px;margin-top:14px;font-size:0.95rem}
        .lede-trio b{color:var(--muted);font-weight:600;text-transform:uppercase;font-size:0.7rem;letter-spacing:var(--track-caps);display:block;margin-bottom:2px}
        .lede-trust{color:var(--muted);font-size:0.8rem;margin-top:12px;border-top:1px solid var(--hair);padding-top:10px}
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 14px;
            font-size: 0.9rem;
            font-variant-numeric: tabular-nums;
        }
        th, td {
            padding: 11px 12px;
            text-align: left;
            border-bottom: 1px solid var(--hair);
        }
        th {
            font-weight: 600;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: var(--track-caps);
            font-size: 0.76rem;
        }
        tbody tr { transition: background-color var(--dur-hover) ease; }
        .plot-container {
            text-align: center;
            margin: 22px 0;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid var(--hair);
            border-radius: 8px;
        }
        .plot-description {
            font-size: 0.88rem;
            color: var(--muted);
            margin-top: 12px;
            text-align: left;
            padding: 10px 14px;
            background: var(--paper);
            border-radius: 6px;
            border-left: 3px solid var(--rule);
        }
        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 22px;
        }
        @media (max-width: 760px) {
            .two-column { grid-template-columns: 1fr; }
            body { padding: 32px 18px 56px; }
        }
        .highlight {
            background: var(--warn-bg);
            padding: 15px 18px;
            border-radius: 8px;
            border-left: 4px solid var(--warn-line);
            margin: 16px 0;
        }
        .footer {
            text-align: center;
            padding: 28px 20px 0;
            color: var(--muted);
            font-size: 0.82rem;
            border-top: 1px solid var(--hair);
            margin-top: 32px;
        }

        /* ---- Token-driven components (promoted from scattered inline styles) ---- */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
            font-variant-numeric: tabular-nums;
            margin: 12px 0;
        }
        .data-table th, .data-table td {
            border: 1px solid var(--hair);
            padding: 8px 10px;
            text-align: left;
        }
        .data-table th { background: var(--wash); color: var(--ink); text-transform: none; letter-spacing: 0; font-size: 0.84rem; }
        .data-table td.num, .data-table th.num { text-align: center; }
        .result-box {
            background: var(--fill);
            border: 1px solid var(--hair);
            border-radius: 8px;
            padding: 12px 14px;
            margin-top: 12px;
        }
        .callout {
            padding: 15px 18px;
            border-radius: 8px;
            border-left: 4px solid var(--hair);
            margin: 16px 0;
        }
        .callout h4 { margin-top: 0; }
        .callout--warn { background: var(--warn-bg); border-left-color: var(--warn-line); }
        .callout--ok { background: var(--ok-bg); border-left-color: var(--ok-line); }
        .callout--ok h4 { color: var(--ok-ink); }
        /* Significance verdict — words carry it; a small neutral dot marks the significant case.
           A null result is a valid finding, never a red "failure". */
        .verdict { font-weight: 600; white-space: nowrap; }
        .verdict--sig { color: var(--ink); }
        .verdict--sig::before {
            content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%;
            background: var(--ink); margin-right: 7px; vertical-align: 0.06em;
        }
        .verdict--ns { color: var(--muted); font-weight: 500; }
        .omnibus-line { color: var(--muted); font-size: 0.85rem; margin-top: 8px; }
        .omnibus-line .appendix-ref { margin-left: 6px; }
        .plot-container--sep {
            margin-bottom: 44px;
            padding-bottom: 28px;
            border-bottom: 1px solid var(--hair);
        }

        /* ---- Hover polish (pointer-only, transform/opacity only) ---- */
        @media (hover: hover) and (pointer: fine) {
            tbody tr:hover { background: var(--wash); }
            .stat-card { transition: transform var(--dur-hover) var(--ease-out), box-shadow var(--dur-hover) var(--ease-out); }
            .stat-card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(26, 26, 26, 0.08); }
            .plot-container img { transition: box-shadow var(--dur-hover) var(--ease-out); }
            .plot-container img:hover { box-shadow: 0 4px 14px rgba(26, 26, 26, 0.10); }
        }

        /* ---- Scroll reveal — hidden ONLY when JS confirms it can reveal (.js-reveal on the root html element) ---- */
        .js-reveal .section, .js-reveal .plot-container {
            opacity: 0;
            transform: translateY(8px);
            transition: opacity var(--dur-reveal) var(--ease-out), transform var(--dur-reveal) var(--ease-out);
        }
        .js-reveal .section.is-visible, .js-reveal .plot-container.is-visible {
            opacity: 1;
            transform: none;
        }

        /* ---- Accessibility ---- */
        :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 3px; }
        @media (prefers-reduced-motion: reduce) {
            .js-reveal .section, .js-reveal .plot-container { opacity: 1; transform: none; transition: none; }
            .stat-card, .plot-container img, tbody tr { transition: none; }
            .stat-card:hover { transform: none; box-shadow: none; }
            .plot-container img:hover { box-shadow: none; }
        }
        @media (prefers-contrast: more) {
            :root { --hair: #9A9A97; --muted: #3A3A3A; }
            .section, .stat-card, .plot-container img { border-color: var(--hair); }
            tbody tr:hover { background: var(--wash); outline: 1px solid var(--accent); }
        }

        /* ---- Print: no motion, nothing splits across a page ---- */
        @media print {
            * { animation: none !important; transition: none !important; }
            .js-reveal .section, .js-reveal .plot-container { opacity: 1 !important; transform: none !important; }
            body { max-width: none; padding: 0; }
            .section, .stat-card, .plot-container, .highlight, .callout, .result-box, .data-table, table, tr {
                page-break-inside: avoid; break-inside: avoid;
            }
            .section h2, .section h3 { break-after: avoid; }
            .plot-container img { break-inside: avoid; }
            .section { border: none; }
            tbody tr:hover { background: transparent; }
        }"""

# Static document footer.
_REPORT_FOOTER = """
    <div class="footer">
        <p>Generated by SCAT """ + __version__ + """ &middot; Spot Classification and Analysis Tool</p>
    </div>
    <script>
    (function () {
      try {
        var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (reduce || !('IntersectionObserver' in window)) return;   // leave everything visible
        var targets = document.querySelectorAll('.section, .plot-container');
        if (!targets.length) return;
        var io = new IntersectionObserver(function (entries, obs) {
          var shown = 0;
          entries.forEach(function (e) {
            if (e.isIntersecting) {
              e.target.style.transitionDelay = (Math.min(shown, 4) * 60) + 'ms';
              e.target.classList.add('is-visible');
              obs.unobserve(e.target);
              shown++;
            }
          });
        }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
        // Hide-then-reveal is armed ONLY now that the observer exists — if setup had thrown,
        // .js-reveal is never added and content stays visible.
        document.documentElement.classList.add('js-reveal');
        targets.forEach(function (t) { io.observe(t); });
        // Safety net: reveal all after 3s in case anything wedges.
        setTimeout(function () {
          document.querySelectorAll('.section, .plot-container').forEach(function (t) {
            t.classList.add('is-visible');
          });
        }, 3000);
      } catch (e) {
        document.documentElement.classList.remove('js-reveal');
      }
    })();
    </script>
</body>
</html>"""


# One neutral reference/mean line for every histogram — a mean marker is structure, not a data
# category, so it is never a semantic color (was a stray Material red on the grey histograms).
_MEAN_LINE = "#333333"


class ReportGenerator:
    """Generate HTML/PDF reports from analysis results."""
    
    # Human-readable metric names for reports
    # Should match FEATURE_LABELS in visualization.py for consistency
    METRIC_LABELS = {
        # Count metrics
        'n_total': 'Total Deposit Count',
        'n_normal': 'Normal Deposit Count',
        'n_rod': 'ROD Deposit Count',
        'n_artifact': 'Artifact Count',
        
        # Fraction/ratio metrics
        'rod_fraction': 'ROD Fraction',
        'normal_fraction': 'Normal Fraction',
        'artifact_fraction': 'Artifact Fraction',
        
        # Area metrics (morphology/size)
        'total_area': 'Total Deposit Area (px²)',
        'mean_area': 'Mean Deposit Size (px²)',
        'normal_mean_area': 'Normal Deposit Size (px²)',
        'rod_mean_area': 'ROD Deposit Size (px²)',
        'normal_std_area': 'Normal Size Variability (px²)',
        'rod_std_area': 'ROD Size Variability (px²)',
        'area_px': 'Area (px²)',
        
        # IOD metrics (Integrated Optical Density - pigment amount)
        'total_iod': 'Total Pigment Amount (IOD)',
        'mean_iod': 'Mean Pigment Amount (IOD)',
        'normal_total_iod': 'Normal Total Pigment (IOD)',
        'normal_mean_iod': 'Normal Mean Pigment (IOD)',
        'rod_total_iod': 'ROD Total Pigment (IOD)',
        'rod_mean_iod': 'ROD Mean Pigment (IOD)',
        'iod': 'Pigment Amount (IOD)',
        
        # Color metrics - Hue reflects pH via Bromophenol Blue indicator
        'mean_hue': 'pH Indicator Hue (°)',
        'normal_mean_hue': 'Normal pH Indicator (Hue °)',
        'rod_mean_hue': 'ROD pH Indicator (Hue °)',
        'mean_saturation': 'Color Saturation',
        'hue_cv': 'pH Indicator Variability (CV)',
        
        # Lightness - reflects pigment concentration/density
        'mean_lightness': 'Pigment Density (Lightness)',
        'normal_mean_lightness': 'Normal Pigment Density (Lightness)',
        'rod_mean_lightness': 'ROD Pigment Density (Lightness)',
        'pigment_density': 'Pigment Density',
        'mean_pigment_density': 'Mean Pigment Density',
        'mean_brightness': 'Mean Brightness',
        'mean_red': 'Mean Red Channel',
        'mean_green': 'Mean Green Channel',
        'mean_blue': 'Mean Blue Channel',
        
        # Shape metrics (morphology)
        'mean_circularity': 'Mean Circularity',
        'normal_mean_circularity': 'Normal Circularity',
        'rod_mean_circularity': 'ROD Circularity',
        'mean_eccentricity': 'Mean Eccentricity',
        'mean_solidity': 'Mean Solidity',
        'mean_compactness': 'Mean Compactness',
        'mean_aspect_ratio': 'Mean Aspect Ratio',
        'circularity': 'Circularity',
        'eccentricity': 'Eccentricity',
        'solidity': 'Solidity',
        'aspect_ratio': 'Aspect Ratio',
        
        # Density metrics
        'deposit_density': 'Deposit Density (per unit area)',
        'coverage_ratio': 'Film Coverage Ratio',
        'deposits_per_fly': 'Deposits per Fly',
        
        # pH metrics (if directly calculated)
        'estimated_ph': 'Estimated pH Value',
        'acidity_index': 'Acidity Index',
        
        # Spatial metrics
        'mean_nnd': 'Mean Nearest Neighbor Distance',
        'clark_evans_r': 'Clark-Evans R Index',
        'edge_fraction': 'Edge Fraction',
        
        # Texture metrics
        'contrast': 'Texture Contrast',
        'homogeneity': 'Texture Homogeneity',
        'energy': 'Texture Energy',
        'correlation': 'Texture Correlation',
    }
    
    # Proper display names for metrics (without biological interpretation)
    # Used in significant findings, table headers, etc.
    METRIC_DISPLAY_NAMES = {
        # Count metrics
        'n_total': 'Total Count',
        'n_normal': 'Normal Count',
        'n_rod': 'ROD Count',
        'n_artifact': 'Artifact Count',
        
        # Fraction/ratio metrics
        'rod_fraction': 'ROD Fraction',
        'normal_fraction': 'Normal Fraction',
        'artifact_fraction': 'Artifact Fraction',
        
        # Area metrics
        'total_area': 'Total Area',
        'mean_area': 'Mean Area',
        'normal_mean_area': 'Mean Area of Normal Deposits',
        'rod_mean_area': 'Mean Area of ROD Deposits',
        'normal_std_area': 'Area Std Dev of Normal Deposits',
        'rod_std_area': 'Area Std Dev of ROD Deposits',
        'area_px': 'Area',
        
        # IOD metrics
        'total_iod': 'Total IOD',
        'mean_iod': 'Mean IOD',
        'normal_total_iod': 'Total IOD of Normal Deposits',
        'normal_mean_iod': 'Mean IOD of Normal Deposits',
        'rod_total_iod': 'Total IOD of ROD Deposits',
        'rod_mean_iod': 'Mean IOD of ROD Deposits',
        'iod': 'IOD',
        
        # Color metrics - Hue
        'mean_hue': 'Mean Hue',
        'normal_mean_hue': 'Mean Hue of Normal Deposits',
        'rod_mean_hue': 'Mean Hue of ROD Deposits',
        'mean_saturation': 'Mean Saturation',
        'hue_cv': 'Hue CV',
        
        # Lightness
        'mean_lightness': 'Mean Lightness',
        'normal_mean_lightness': 'Mean Lightness of Normal Deposits',
        'rod_mean_lightness': 'Mean Lightness of ROD Deposits',
        'mean_brightness': 'Mean Brightness',
        'mean_red': 'Mean Red Channel',
        'mean_green': 'Mean Green Channel',
        'mean_blue': 'Mean Blue Channel',
        
        # Shape metrics
        'mean_circularity': 'Mean Circularity',
        'normal_mean_circularity': 'Mean Circularity of Normal Deposits',
        'rod_mean_circularity': 'Mean Circularity of ROD Deposits',
        'mean_eccentricity': 'Mean Eccentricity',
        'mean_solidity': 'Mean Solidity',
        'mean_compactness': 'Mean Compactness',
        'mean_aspect_ratio': 'Mean Aspect Ratio',
        'circularity': 'Circularity',
        'eccentricity': 'Eccentricity',
        'solidity': 'Solidity',
        'aspect_ratio': 'Aspect Ratio',
        
        # Density metrics
        'deposit_density': 'Deposit Density',
        'coverage_ratio': 'Coverage Ratio',
        'deposits_per_fly': 'Deposits per Fly',
        
        # pH metrics
        'estimated_ph': 'Estimated pH',
        'acidity_index': 'Acidity Index',
        
        # Spatial metrics
        'mean_nnd': 'Mean NND',
        'clark_evans_r': 'Clark-Evans R',
        'edge_fraction': 'Edge Fraction',
    }
    
    # Labels for correlation keys
    CORRELATION_KEY_LABELS = {
        'size_vs_iod': 'Size vs IOD',
        'size_vs_hue': 'Size vs Hue',
        'size_vs_circularity': 'Size vs Circularity',
        'iod_vs_hue': 'IOD vs Hue',
        'size_vs_lightness': 'Size vs Lightness',
        'circularity_vs_aspect': 'Circularity vs Aspect Ratio',
        'pigment_density_vs_hue': 'Pigment Density vs Hue',
    }
    
    @classmethod
    def get_metric_label(cls, metric_name: str) -> str:
        """Convert metric variable name to human-readable label."""
        if metric_name in cls.METRIC_LABELS:
            return cls.METRIC_LABELS[metric_name]
        # Fallback: convert snake_case to Title Case
        return metric_name.replace('_', ' ').title()
    
    @classmethod
    def get_metric_display_name(cls, metric_name: str) -> str:
        """Get proper display name for metric (without biological interpretation)."""
        if metric_name in cls.METRIC_DISPLAY_NAMES:
            return cls.METRIC_DISPLAY_NAMES[metric_name]
        # Fallback: convert snake_case to Title Case, preserving acronyms
        parts = metric_name.split('_')
        formatted = []
        for part in parts:
            # Preserve uppercase acronyms
            if part.upper() in ('IOD', 'ROD', 'NND', 'CV', 'PH'):
                formatted.append(part.upper())
            else:
                formatted.append(part.title())
        return ' '.join(formatted)
    
    @classmethod
    def get_correlation_label(cls, key: str) -> str:
        """Get proper label for correlation key."""
        if key in cls.CORRELATION_KEY_LABELS:
            return cls.CORRELATION_KEY_LABELS[key]
        # Fallback with proper formatting
        parts = key.split('_')
        formatted = []
        for part in parts:
            if part.upper() in ('IOD', 'ROD', 'NND', 'CV', 'PH'):
                formatted.append(part.upper())
            elif part == 'vs':
                formatted.append('vs')
            else:
                formatted.append(part.title())
        return ' '.join(formatted)
    
    @staticmethod
    def format_correlation_interpretation(interpretation: str) -> str:
        """Format correlation interpretation for display."""
        if not interpretation:
            return ''
        # Replace underscores with spaces
        return interpretation.replace('_', ' ')
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_html_report(
        self,
        film_summary: pd.DataFrame,
        deposit_data: pd.DataFrame = None,
        spatial_stats: Dict = None,
        statistical_results: Dict = None,
        visualization_paths: Dict[str, str] = None,
        metadata: pd.DataFrame = None,
        group_by: str = None,
        title: str = "SCAT Analysis Report",
        analysis: dict = None
    ) -> str:
        """
        Generate comprehensive HTML report.

        Returns:
            Path to generated HTML file
        """
        # Calculate summary statistics
        summary = self._calculate_summary(film_summary)
        
        # Generate inline plots
        inline_plots = {}
        if HAS_MATPLOTLIB:
            # Summary distribution plots (order: Count, Area, IOD, pH, ROD, Circularity)
            # Count and ROD fraction: image-level, others: individual deposit-level
            inline_plots['count_distribution'] = self._generate_count_distribution(film_summary)
            inline_plots['area_distribution'] = self._generate_area_distribution(deposit_data)
            inline_plots['iod_distribution'] = self._generate_iod_distribution(deposit_data)
            inline_plots['ph_distribution'] = self._generate_ph_distribution(deposit_data)
            inline_plots['rod_distribution'] = self._generate_rod_histogram(film_summary)
            inline_plots['circularity_distribution'] = self._generate_circularity_distribution(deposit_data)
            
            # Legacy: overview bar (deposit count by classification)
            inline_plots['overview_bar'] = self._generate_overview_bar(film_summary)
            
            if group_by and group_by in film_summary.columns:
                # Generate all group comparison boxplots
                group_plots = self._generate_all_group_comparisons(film_summary, group_by)
                inline_plots.update(group_plots)
                # Legacy: keep single group_comparison for backward compatibility
                inline_plots['group_comparison'] = self._generate_group_comparison(
                    film_summary, group_by
                )
        
        # Build HTML
        html_content = self._build_html(
            title=title,
            summary=summary,
            film_summary=film_summary,
            deposit_data=deposit_data,
            spatial_stats=spatial_stats,
            statistical_results=statistical_results,
            visualization_paths=visualization_paths,
            inline_plots=inline_plots,
            group_by=group_by,
            analysis=analysis
        )

        # Save
        output_path = self.output_dir / 'report.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _calculate_summary(self, film_summary: pd.DataFrame) -> Dict:
        """Calculate overall summary statistics."""
        def safe_sum(col):
            return int(film_summary[col].sum()) if col in film_summary.columns else 0
        
        def safe_mean(col):
            return float(film_summary[col].mean()) if col in film_summary.columns else 0.0
        
        def safe_std(col):
            return float(film_summary[col].std()) if col in film_summary.columns else 0.0
        
        # Calculate mean_circularity and mean_hue from normal and rod values
        # Weighted average based on deposit counts
        def weighted_mean(normal_col, rod_col):
            n_normal = safe_sum('n_normal')
            n_rod = safe_sum('n_rod')
            total = n_normal + n_rod
            if total == 0:
                return 0.0
            
            normal_mean = safe_mean(normal_col)
            rod_mean = safe_mean(rod_col)
            
            # Handle NaN values
            if np.isnan(normal_mean):
                normal_mean = 0
                n_normal = 0
            if np.isnan(rod_mean):
                rod_mean = 0
                n_rod = 0
            
            total = n_normal + n_rod
            if total == 0:
                return 0.0
            
            return (normal_mean * n_normal + rod_mean * n_rod) / total
        
        return {
            'total_films': len(film_summary),
            # Total deposits excludes artifacts (Normal + ROD only)
            'total_deposits': safe_sum('n_normal') + safe_sum('n_rod'),
            'total_normal': safe_sum('n_normal'),
            'total_rod': safe_sum('n_rod'),
            'mean_rod_fraction': safe_mean('rod_fraction'),
            'std_rod_fraction': safe_std('rod_fraction'),
            # Summary statistics
            'mean_area': safe_mean('mean_area'),
            'mean_circularity': weighted_mean('normal_mean_circularity', 'rod_mean_circularity'),
            'mean_hue': weighted_mean('normal_mean_hue', 'rod_mean_hue'),
            'mean_density': weighted_mean('normal_mean_lightness', 'rod_mean_lightness'),  # Will be converted to 1-lightness
            'mean_total_iod': safe_mean('total_iod'),
            'mean_normal_area': safe_mean('normal_mean_area'),
            'mean_rod_area': safe_mean('rod_mean_area'),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _generate_overview_bar(self, film_summary: pd.DataFrame) -> str:
        """Generate overview bar chart as base64."""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        def safe_sum(col):
            return film_summary[col].sum() if col in film_summary.columns else 0
        
        # Only show Normal and ROD (exclude Artifact from deposit statistics)
        totals = {
            'Normal': safe_sum('n_normal'),
            'ROD': safe_sum('n_rod')
        }
        
        # Use DEPOSIT_COLORS for consistency with visualization module
        colors = [DEPOSIT_COLORS['normal'], DEPOSIT_COLORS['rod']]
        bars = ax.bar(totals.keys(), totals.values(), color=colors, width=0.4)
        
        ax.set_ylabel('Count')
        ax.set_title('Total Deposits by Classification')
        
        # Add value labels
        for bar, val in zip(bars, totals.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{int(val)}', ha='center', va='bottom', fontweight='bold')
        
        # Apply publication style
        apply_publication_style(ax)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    @staticmethod
    def _deposit_values(deposit_data: pd.DataFrame, column: str):
        """Non-artifact, non-null values of `column`, or None if unavailable/empty.

        Shared data step for the deposit-level distributions (area/IOD/pH/circularity).
        """
        if deposit_data is None or column not in deposit_data.columns:
            return None
        valid = deposit_data[deposit_data['label'].isin(['normal', 'rod'])]
        values = valid[column].dropna()
        return values if len(values) > 0 else None

    def _histogram_figure(self, draw) -> str:
        """Shared 8x4 figure lifecycle: draw onto the axes, apply style, encode to base64.

        `draw(ax)` runs each metric's own plotting steps (it may be a no-op when a
        deposit-level column is missing, yielding an empty styled axes as before).
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        draw(ax)
        apply_publication_style(ax)
        plt.tight_layout()
        return self._fig_to_base64(fig)

    @staticmethod
    def _hist_with_mean(ax, data, *, bins, mean, mean_label) -> None:
        """Grey histogram with a red dashed mean line — the common core of most metrics."""
        ax.hist(data, bins=bins, color=DEFAULT_GRAY, edgecolor='white', alpha=0.8)
        ax.axvline(mean, color=_MEAN_LINE, linestyle='--', linewidth=2, label=mean_label)

    def _generate_rod_histogram(self, film_summary: pd.DataFrame) -> str:
        """Generate ROD fraction histogram as base64."""
        def draw(ax):
            self._hist_with_mean(ax, film_summary['rod_fraction'] * 100, bins=15,
                                 mean=film_summary['rod_fraction'].mean() * 100, mean_label='Mean')
            ax.set_xlabel('ROD Fraction (%)')
            ax.set_ylabel('Number of Images')
            ax.set_title('Distribution of ROD Fraction')
            ax.legend()
        return self._histogram_figure(draw)

    def _generate_count_distribution(self, film_summary: pd.DataFrame) -> str:
        """Generate deposit count distribution histogram."""
        def draw(ax):
            counts = film_summary['n_total'] if 'n_total' in film_summary.columns else \
                     film_summary['n_normal'] + film_summary['n_rod']
            self._hist_with_mean(ax, counts, bins=15, mean=counts.mean(), mean_label='Mean')
            ax.set_xlabel('Deposit Count per Image')
            ax.set_ylabel('Number of Images')
            ax.set_title('Distribution of Deposit Counts')
            ax.legend()
        return self._histogram_figure(draw)

    def _generate_area_distribution(self, deposit_data: pd.DataFrame) -> str:
        """Generate area distribution histogram from individual deposits."""
        def draw(ax):
            data = self._deposit_values(deposit_data, 'area_px')
            if data is None:
                return
            # Use 99th percentile as upper limit for better visualization
            upper_limit = min(data.quantile(0.99) * 1.2, data.max())
            upper_limit = int(np.ceil(upper_limit / 50) * 50)  # Round to nearest 50
            # Bin size: 10px
            bins = np.arange(0, upper_limit + 10, 10)
            self._hist_with_mean(ax, data[data <= upper_limit], bins=bins,
                                 mean=data.mean(), mean_label=f'Mean: {data.mean():.1f}')
            # Set x-axis ticks at 50 unit intervals
            ax.set_xticks(np.arange(0, upper_limit + 50, 50))
            ax.set_xlim(0, upper_limit)
            ax.set_xlabel('Deposit Area (px²)')
            ax.set_ylabel('Number of Deposits')
            ax.set_title('Distribution of Deposit Size')
            ax.legend()
            # Add note if data was truncated
            n_truncated = len(data[data > upper_limit])
            if n_truncated > 0:
                ax.text(0.98, 0.95, f'({n_truncated} deposits > {upper_limit} not shown)',
                       transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#666')
        return self._histogram_figure(draw)

    def _generate_iod_distribution(self, deposit_data: pd.DataFrame) -> str:
        """Generate IOD distribution histogram from individual deposits."""
        def draw(ax):
            data = self._deposit_values(deposit_data, 'iod')
            if data is None:
                return
            # Use 99th percentile as upper limit for better visualization
            upper_limit = min(data.quantile(0.99) * 1.2, data.max())
            upper_limit = int(np.ceil(upper_limit / 10) * 10)  # Round to nearest 10
            # Bin size: 5
            bins = np.arange(0, upper_limit + 5, 5)
            self._hist_with_mean(ax, data[data <= upper_limit], bins=bins,
                                 mean=data.mean(), mean_label=f'Mean: {data.mean():.1f}')
            # Set appropriate x-axis ticks
            tick_interval = 20 if upper_limit > 100 else 10
            ax.set_xticks(np.arange(0, upper_limit + tick_interval, tick_interval))
            ax.set_xlim(0, upper_limit)
            ax.set_xlabel('IOD (Integrated Optical Density)')
            ax.set_ylabel('Number of Deposits')
            ax.set_title('Distribution of Pigment Amount (IOD)')
            ax.legend()
            # Add note if data was truncated
            n_truncated = len(data[data > upper_limit])
            if n_truncated > 0:
                ax.text(0.98, 0.95, f'({n_truncated} deposits > {upper_limit} not shown)',
                       transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#666')
        return self._histogram_figure(draw)

    def _generate_ph_distribution(self, deposit_data: pd.DataFrame) -> str:
        """Generate pH (Hue) distribution histogram from individual deposits with actual colors."""
        def draw(ax):
            all_hues = self._deposit_values(deposit_data, 'mean_hue')
            if all_hues is None:
                return
            # Bin size: 10 degrees
            bins = np.arange(0, 370, 10)
            # Create histogram with colored bars based on hue values
            n, bin_edges, patches = ax.hist(all_hues, bins=bins, edgecolor='white', alpha=0.8)
            # Color each bar based on its hue value
            for patch, bin_left, bin_right in zip(patches, bin_edges[:-1], bin_edges[1:]):
                bin_center = (bin_left + bin_right) / 2
                patch.set_facecolor(hue_to_rgb(bin_center))
            ax.axvline(np.mean(all_hues), color=_MEAN_LINE, linestyle='--',
                      linewidth=2, label=f'Mean: {np.mean(all_hues):.1f}°')
            ax.set_xlabel('pH Indicator Hue (°)')
            ax.set_ylabel('Number of Deposits')
            ax.set_title('Distribution of pH Indicator (Hue)')
            ax.legend()
        return self._histogram_figure(draw)

    def _generate_circularity_distribution(self, deposit_data: pd.DataFrame) -> str:
        """Generate circularity distribution histogram from individual deposits."""
        def draw(ax):
            all_circ = self._deposit_values(deposit_data, 'circularity')
            if all_circ is None:
                return
            # Bin size: 0.05
            bins = np.arange(0, 1.05, 0.05)
            self._hist_with_mean(ax, all_circ, bins=bins, mean=np.mean(all_circ),
                                 mean_label=f'Mean: {np.mean(all_circ):.3f}')
            ax.set_xlabel('Circularity (0-1)')
            ax.set_ylabel('Number of Deposits')
            ax.set_title('Distribution of Circularity')
            ax.legend()
            ax.set_xlim(0, 1)
        return self._histogram_figure(draw)

    def _generate_group_comparison(
        self, 
        film_summary: pd.DataFrame, 
        group_by: str
    ) -> str:
        """Generate group comparison box plot for ROD fraction (legacy)."""
        return self._generate_metric_boxplot(film_summary, 'rod_fraction', group_by, 
                                              'ROD Fraction (%)', scale=100)
    
    @staticmethod
    def _natural_sort_key(value):
        """Natural sort key that handles numbers correctly.
        
        Examples:
            - "1", "2", "10" → sorted as 1, 2, 10 (not 1, 10, 2)
            - "Group1", "Group10", "Group2" → Group1, Group2, Group10
            - Mixed: numbers first, then strings
        """
        import re
        value_str = str(value)
        
        # Check if it's a pure number
        try:
            return (0, float(value_str), value_str)
        except ValueError:
            pass
        
        # Split into numeric and non-numeric parts for natural sorting
        parts = re.split(r'(\d+)', value_str)
        converted = []
        for part in parts:
            if part.isdigit():
                converted.append((0, int(part)))
            elif part:
                converted.append((1, part.lower()))
        
        return (1, converted, value_str)
    
    def _generate_metric_boxplot(
        self,
        film_summary: pd.DataFrame,
        metric: str,
        group_by: str,
        ylabel: str = None,
        scale: float = 1.0,
        use_hue_colors: bool = False
    ) -> str:
        """Generate boxplot comparing groups for a specific metric."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Logical group order (control first, then low<mid<high or numeric dose), not alphabetical
        groups = order_groups(film_summary[group_by].dropna().unique())
        
        # Check if metric exists
        if metric not in film_summary.columns:
            plt.close(fig)
            return ""
        
        data = [film_summary[film_summary[group_by] == g][metric].dropna() * scale 
                for g in groups]
        
        # Set tick labels separately (the boxplot 'labels' kwarg was renamed to
        # 'tick_labels' in matplotlib 3.9; this form works across versions).
        bp = ax.boxplot(data, patch_artist=True)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([str(g) for g in groups])
        
        # Color boxes
        if use_hue_colors or 'hue' in metric.lower():
            # pH-related metric: colour each box by its group's MEAN HUE (the Bromophenol-Blue
            # indicator colour). Fill + a darker same-hue edge/median so the hue reads clearly even
            # when the box is thin (low within-group hue variance).
            for patch, median, group in zip(bp['boxes'], bp['medians'], groups):
                mean_hue = film_summary[film_summary[group_by] == group][metric].mean()
                if not np.isnan(mean_hue):
                    fill = hue_to_rgb(mean_hue)
                    edge = hue_to_rgb(mean_hue, lightness=0.32)   # darker shade of the same hue
                else:
                    fill, edge = DEFAULT_GRAY, '#333333'
                patch.set_facecolor(fill)
                patch.set_edgecolor(edge)
                patch.set_linewidth(1.3)
                patch.set_alpha(0.9)
                median.set_color(edge)
                median.set_linewidth(1.6)
        else:
            # Standard categorical palette per group
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(PASTEL_PALETTE[i % len(PASTEL_PALETTE)])
                patch.set_edgecolor('#333333')
                patch.set_alpha(0.85)
        
        metric_label = self.get_metric_label(metric)
        ax.set_ylabel(ylabel or metric_label)
        ax.set_xlabel(group_by.replace('_', ' ').title())
        ax.set_title(f'{metric_label} by {group_by.replace("_", " ").title()}')
        
        apply_publication_style(ax)
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _generate_all_group_comparisons(
        self,
        film_summary: pd.DataFrame,
        group_by: str
    ) -> Dict[str, str]:
        """Generate boxplots for all comparison metrics."""
        plots = {}
        
        # Metrics to compare in order matching Summary section:
        # Count, Area, IOD, pH(Hue), ROD, Circularity
        comparison_metrics = [
            ('n_total', 'Deposit Count', 1.0, False),
            ('mean_area', 'Mean Deposit Area (px²)', 1.0, False),
            ('total_iod', 'Total IOD', 1.0, False),
            ('mean_hue', 'pH Indicator (Hue °)', 1.0, True),  # Use hue colors
            ('rod_fraction', 'ROD Fraction (%)', 100.0, False),
            ('mean_circularity', 'Mean Circularity', 1.0, False),
        ]
        
        for metric, ylabel, scale, use_hue in comparison_metrics:
            # Check if metric exists, try alternatives
            actual_metric = metric
            if metric == 'mean_hue':
                # Try to use combined hue
                if metric not in film_summary.columns:
                    if 'normal_mean_hue' in film_summary.columns:
                        actual_metric = 'normal_mean_hue'  # Fallback
                    else:
                        continue
            if metric == 'mean_circularity':
                # Try to use combined circularity
                if metric not in film_summary.columns:
                    if 'normal_mean_circularity' in film_summary.columns:
                        actual_metric = 'normal_mean_circularity'  # Fallback
                    else:
                        continue
            
            if actual_metric in film_summary.columns:
                plot = self._generate_metric_boxplot(
                    film_summary, actual_metric, group_by, ylabel, scale, use_hue
                )
                if plot:
                    plots[f'group_{metric}'] = plot
        
        return plots
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def _build_html(
        self,
        title: str,
        summary: Dict,
        film_summary: pd.DataFrame,
        deposit_data: pd.DataFrame,
        spatial_stats: Dict,
        statistical_results: Dict,
        visualization_paths: Dict,
        inline_plots: Dict,
        group_by: str,
        analysis: dict = None
    ) -> str:
        """Build complete HTML document from per-section builders."""
        return "".join([
            self._html_document_head(title, summary),                                  # masthead only
            self._html_finding_lede(film_summary, deposit_data, statistical_results, group_by, analysis),
            self._html_group_comparison(inline_plots, statistical_results, group_by, analysis),  # analysis added (Task 2)
            self._html_population_overview(summary, inline_plots),                      # demoted pooled context
            self._html_stats_appendix(statistical_results),
            self._html_spatial_section(spatial_stats),
            self._html_film_table(film_summary),
            self._html_methods(),
            _REPORT_FOOTER,
        ])

    def _html_finding_lede(self, film_summary, deposit_data, statistical_results, group_by, analysis):
        from scat import metrics as _metrics, confidence as _confidence, findings as _findings
        import html as _h
        analysis = analysis or {}
        pm = _metrics.resolve_metric(analysis.get("primary_metric"))
        norm = analysis.get("normalization") or _metrics.DEFAULT_NORMALIZATION
        thr = float(analysis.get("confidence_threshold", _metrics.DEFAULT_THRESHOLD))
        headline = _metrics.format_headline(film_summary, pm, norm, meta={})
        n_images = len(film_summary)
        grouped = bool(group_by) and group_by in film_summary.columns
        n_groups = int(film_summary[group_by].dropna().nunique()) if grouped else 0
        group_label = self.get_metric_label(group_by) if grouped else None
        if group_label and group_label.strip().lower() in ("group", "groups", "condition"):
            group_label = group_label.lower()
        f = _findings.compose_finding(stats=statistical_results, primary_metric=pm, headline=headline,
                                      n_images=n_images, n_groups=n_groups, group_label=group_label)
        trust = _confidence.run_trust(deposit_data, thr)
        return f'''
    <div class="section">
      <div class="lede">
        <div class="finding">{_h.escape(f["sentence"])}</div>
        <div class="lede-trio">
          <span><b>Primary metric</b>{_h.escape(f["metric"])}</span>
          <span><b>Test</b>{_h.escape(f["test"])}</span>
          <span><b>Scope</b>{_h.escape(f["scope"])}</span>
        </div>
        <div class="lede-trust">{_h.escape(trust["line"])}</div>
      </div>
    </div>
'''

    def _html_document_head(self, title: str, summary: Dict) -> str:
        """Document boilerplate, CSS, and the page masthead only.

        The pooled Summary stat cards were demoted into ``_html_population_overview``;
        this method now opens ``<html><body>`` and the header and closes neither the
        body nor any section, so every downstream section is self-contained.
        """
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{_REPORT_CSS}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="subtitle">Generated: {summary['generated_at']}</div>
    </div>
'''
        return html

    def _html_population_overview(self, summary: Dict, inline_plots: Dict) -> str:
        """Demoted pooled context: the Summary stat-card grid + distribution histograms.

        A single self-contained ``<div class="section">``. The stat cards moved here
        verbatim from ``_html_document_head``; the histogram body comes from
        ``_html_distributions`` (which no longer emits the closing ``</div>``), and this
        method owns the section's opening and closing tags.
        """
        html = f'''
    <!-- Population overview -->
    <div class="section">
        <h2>Population overview</h2>
        <p class="section-intro">Pooled characteristics across all images (context, not the headline).</p>
        <div class="stats-grid">
            <div class="stat-card rod">
                <div class="value">{summary['mean_rod_fraction']*100:.1f}%</div>
                <div class="label">ROD Fraction (±{summary['std_rod_fraction']*100:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="value">{summary['total_films']}</div>
                <div class="label">Total Images</div>
            </div>
            <div class="stat-card">
                <div class="value">{summary['total_deposits']}</div>
                <div class="label">Total Deposits</div>
            </div>
            <div class="stat-card">
                <div class="value">{summary['mean_circularity']:.3f}</div>
                <div class="label">Mean Circularity</div>
            </div>
            <div class="stat-card">
                <div class="value">{summary['mean_area']:.1f}</div>
                <div class="label">Mean Area (px²)</div>
            </div>
            <div class="stat-card">
                <div class="value">{summary['mean_total_iod']:.0f}</div>
                <div class="label">Mean IOD</div>
            </div>
            <div class="stat-card">
                <div class="value">{1.0 - summary['mean_density']:.3f}</div>
                <div class="label">Mean Pigment Density</div>
            </div>
            <div class="stat-card">
                <div class="value">{summary['mean_hue']:.1f}°</div>
                <div class="label">Mean <span class="lc">pH</span> Indicator (Hue)</div>
            </div>
        </div>
'''
        html += self._html_distributions(inline_plots)
        html += '    </div>\n'
        return html

    def _html_distributions(self, inline_plots: Dict) -> str:
        """Deposit distribution histograms (body only; the enclosing section is owned
        by ``_html_population_overview``, which appends the closing ``</div>``)."""
        html = ""
        # Add distribution plots (2 columns x 3 rows)
        if 'count_distribution' in inline_plots:
            html += '''
        <h3>Distributions</h3>
        
        <!-- Row 1: Count and Area -->
        <div class="two-column">
            <div class="plot-container">
'''
            html += f'                <img src="data:image/png;base64,{inline_plots["count_distribution"]}" alt="Count Distribution">\n'
            html += '''                <p class="plot-description">
                    <strong>Deposit Count:</strong> Number of deposits detected per image (image-level).
                </p>
            </div>
            <div class="plot-container">
'''
            html += f'                <img src="data:image/png;base64,{inline_plots["area_distribution"]}" alt="Area Distribution">\n'
            html += '''                <p class="plot-description">
                    <strong>Deposit Size:</strong> Area of individual deposits in pixels² (deposit-level).
                </p>
            </div>
        </div>
        
        <!-- Row 2: IOD and pH -->
        <div class="two-column">
            <div class="plot-container">
'''
            html += f'                <img src="data:image/png;base64,{inline_plots["iod_distribution"]}" alt="IOD Distribution">\n'
            html += '''                <p class="plot-description">
                    <strong>IOD (Integrated Optical Density):</strong> Pigment amount of individual deposits. IOD = Area × Density (deposit-level).
                </p>
            </div>
            <div class="plot-container">
'''
            html += f'                <img src="data:image/png;base64,{inline_plots["ph_distribution"]}" alt="pH Distribution">\n'
            html += '''                <p class="plot-description">
                    <strong>pH Indicator (Hue):</strong> Hue of individual deposits. ~60° = acidic, ~240° = alkaline (deposit-level).
                </p>
            </div>
        </div>
        
        <!-- Row 3: ROD and Circularity -->
        <div class="two-column">
            <div class="plot-container">
'''
            html += f'                <img src="data:image/png;base64,{inline_plots["rod_distribution"]}" alt="ROD Distribution">\n'
            html += '''                <p class="plot-description">
                    <strong>ROD Fraction:</strong> Percentage of ROD among all deposits per image (image-level).
                </p>
            </div>
            <div class="plot-container">
'''
            html += f'                <img src="data:image/png;base64,{inline_plots["circularity_distribution"]}" alt="Circularity Distribution">\n'
            html += '''                <p class="plot-description">
                    <strong>Circularity:</strong> Shape regularity of individual deposits (0-1). 1.0 = perfect circle (deposit-level).
                </p>
            </div>
        </div>
'''
        return html

    def _html_group_comparison(self, inline_plots: Dict, statistical_results: Dict, group_by: str, analysis: dict = None) -> str:
        """Per-metric group-comparison boxplots with omnibus test results."""
        html = ""
        # Group comparison - show metrics vertically with omnibus results
        if group_by and 'group_comparison' in inline_plots:
            group_label = self.get_metric_label(group_by)
            # Don't echo a generic column name back ("Group Comparison (Group)" / "across different group groups").
            _generic = group_label.strip().lower() in ("group", "groups", "condition")
            _heading = "Group Comparison" if _generic else f"Group Comparison ({group_label})"
            _across = "groups" if _generic else f"{group_label.lower()} groups"
            html += f'''
    <div class="section">
        <h2>{_heading}</h2>
        <p class="section-intro">Comparing deposit characteristics across {_across}.</p>
'''
            
            # Define metrics in order matching Summary section
            # Order: Count, Area, IOD, pH(Hue), ROD, Circularity
            group_metrics = [
                ('group_n_total', 'n_total', 'Deposit Count', 'Number of deposits detected per image.'),
                ('group_mean_area', 'mean_area', 'Deposit Size', 'Mean area of deposits in pixels².'),
                ('group_total_iod', 'total_iod', 'Pigment Amount (IOD)', 'Total Integrated Optical Density per image.'),
                ('group_mean_hue', 'mean_hue', 'pH Indicator (Hue)', 'pH indicator hue. Bar colors reflect actual pH-indicator colors.'),
                ('group_rod_fraction', 'rod_fraction', 'ROD Fraction', 'Percentage of ROD among all deposits.'),
                ('group_mean_circularity', 'mean_circularity', 'Circularity', 'Shape regularity (0-1). 1.0 = perfect circle.'),
            ]
            
            # Build appendix index mapping
            appendix_index = {}
            idx = 1
            for _, stat_key, _, _ in group_metrics:
                if statistical_results and stat_key in statistical_results:
                    appendix_index[stat_key] = idx
                    idx += 1
            
            # Each metric = a boxplot cell (chart → description → compact one-line verdict).
            # Two cells per row so adjacent metrics can be compared without a long scroll.
            def _verdict(is_sig: bool, label: str) -> str:
                cls = 'verdict--sig' if is_sig else 'verdict--ns'
                return f'<span class="verdict {cls}">{label}</span>'

            cells = []
            for plot_key, stat_key, title, desc in group_metrics:
                if plot_key not in inline_plots:
                    continue
                cell = (
                    '            <div class="plot-container">\n'
                    f'                <img src="data:image/png;base64,{inline_plots[plot_key]}" alt="{title}">\n'
                    f'                <p class="plot-description"><strong>{title}:</strong> {desc}</p>\n'
                )
                # Compact omnibus caption if available
                if statistical_results and stat_key in statistical_results:
                    result = statistical_results[stat_key]
                    if isinstance(result, dict) and 'error' not in result:
                        appendix_num = appendix_index.get(stat_key, '?')
                        if 'overall_test' in result:
                            test_name = result['overall_test']
                            p_val = result.get('overall_p_value', 1.0)
                            is_sig = result.get('overall_significant', False)
                            verdict = _verdict(is_sig, 'At least one group differs' if is_sig else 'No significant difference')
                        else:
                            test_name = result.get('test_name', 'N/A')
                            p_val = result.get('p_value', 1.0)
                            is_sig = result.get('significant', False)
                            verdict = _verdict(is_sig, 'Significant' if is_sig else 'Not significant')
                        cell += (
                            f'                <p class="omnibus-line">{test_name}, p = {p_val:.4f} &nbsp;·&nbsp; {verdict}'
                            f'<span class="appendix-ref">→ Appendix {appendix_num}</span></p>\n'
                        )
                cell += '            </div>\n'
                cells.append(cell)

            for i in range(0, len(cells), 2):
                html += '        <div class="two-column">\n'
                html += ''.join(cells[i:i + 2])
                html += '        </div>\n'
            
            # Add summary of significant differences (without bullet points)
            if statistical_results and isinstance(statistical_results, dict):
                significant_findings = []
                for _, stat_key, title, _ in group_metrics:
                    if stat_key in statistical_results:
                        result = statistical_results[stat_key]
                        if isinstance(result, dict) and 'error' not in result:
                            if result.get('overall_significant') or result.get('significant'):
                                p_val = result.get('overall_p_value', result.get('p_value', 1.0))
                                significant_findings.append((title, p_val))
                
                if significant_findings:
                    html += '''
        <div class="result-box">
            <h4 style="margin:0 0 6px;">Comparisons reaching significance</h4>
            <p style="margin-bottom:0;">
'''
                    findings_text = ' · '.join([f'<strong>{m}</strong> (p={p:.4f})' for m, p in significant_findings])
                    html += f'                {findings_text}\n'
                    html += '''            </p>
        </div>
'''
            
            html += '    </div>\n'
        return html

    def _html_stats_appendix(self, statistical_results: Dict) -> str:
        """Appendix: pairwise comparisons and group statistics tables."""
        html = ""
        # Statistical results - shown as Appendix
        has_valid_stats = (
            statistical_results 
            and isinstance(statistical_results, dict) 
            and len(statistical_results) > 0
            and any(isinstance(v, dict) and 'error' not in v for v in statistical_results.values())
        )
        
        if has_valid_stats:
            html += '''
    <div class="section">
        <h2>Appendix: Statistical Details</h2>
        <p class="section-intro">Detailed pairwise comparison results for group comparisons.</p>
'''
            # Define metrics in order matching Summary/Group Comparison sections
            ordered_metrics = [
                ('n_total', 'Total Deposit Count'),
                ('mean_area', 'Mean Deposit Area'),
                ('total_iod', 'Total IOD'),
                ('mean_hue', 'pH Indicator (Hue)'),
                ('rod_fraction', 'ROD Fraction'),
                ('mean_circularity', 'Mean Circularity'),
            ]
            
            appendix_num = 0
            for metric_key, metric_title in ordered_metrics:
                if metric_key not in statistical_results:
                    continue
                    
                result = statistical_results[metric_key]
                
                # Skip if result is not a dict or has error
                if not isinstance(result, dict):
                    continue
                if 'error' in result:
                    continue
                
                appendix_num += 1
                html += f'        <h3>{appendix_num}. {metric_title}</h3>\n'
                
                if 'overall_test' in result:
                    # 3+ groups: Show pairwise comparisons (omnibus already shown in Group Comparison)
                    pairwise = result.get('pairwise_comparisons', [])
                    if pairwise:
                        correction = result.get('correction_method', 'none')
                        correction_label = f" ({correction.capitalize()} corrected)" if correction != 'none' else ""
                        
                        # Check if correction was applied
                        has_correction = any('p_value_corrected' in pw for pw in pairwise if 'error' not in pw)
                        
                        html += f'''        <div style="margin-top:10px;">
            <p><strong>🔄 Pairwise Comparisons{correction_label}:</strong></p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Comparison</th>
                        <th class="num">Test</th>
                        <th class="num">p (raw)</th>
                        <th class="num">{"p (corrected)" if has_correction else "p-value"}</th>
                        <th class="num">Effect Size</th>
                        <th class="num">Result</th>
                    </tr>
                </thead>
                <tbody>
'''
                        for pw in pairwise:
                            if 'error' in pw:
                                continue
                            g1 = pw.get('group1_name', '?')
                            g2 = pw.get('group2_name', '?')
                            test_name = pw.get('test_name', 'N/A')
                            
                            # Get both raw and corrected p-values
                            p_raw = pw.get('p_value', 1.0)
                            
                            # Use corrected p-value if available for significance
                            if 'p_value_corrected' in pw:
                                p_corrected = pw['p_value_corrected']
                                is_sig = pw.get('significant_corrected', False)
                                p_display = f"{p_corrected:.4f}"
                            else:
                                p_corrected = p_raw
                                is_sig = pw.get('significant', False)
                                p_display = f"{p_raw:.4f}"
                            
                            effect_d = pw.get('cohens_d', 0)
                            effect_label = pw.get('effect_size', 'N/A')
                            
                            sig_icon = ('<span class="verdict verdict--sig">Yes</span>' if is_sig
                                        else '<span class="verdict verdict--ns">No</span>')
                            # Highlight row if raw p-value was significant but corrected wasn't
                            raw_sig = p_raw < 0.05
                            if raw_sig and not is_sig:
                                row_bg = 'var(--warn-bg)'  # amber tint - significant before correction
                            elif is_sig:
                                row_bg = 'var(--ok-bg)'    # teal tint - significant after correction
                            else:
                                row_bg = 'var(--surface)'  # plain - not significant
                            
                            html += f'''                    <tr style="background:{row_bg};">
                        <td>{g1} vs {g2}</td>
                        <td class="num">{test_name}</td>
                        <td class="num">{p_raw:.4f}</td>
                        <td class="num">{p_display}</td>
                        <td class="num">{effect_d:.2f} ({effect_label})</td>
                        <td class="num">{sig_icon}</td>
                    </tr>
'''
                        html += '''                </tbody>
            </table>
            <p style="margin-top:8px; font-size:0.85em; color:var(--muted);">
                <span style="display:inline-block; width:12px; height:12px; background:var(--ok-bg); border:1px solid var(--hair); margin-right:4px;"></span> Significant after correction &nbsp;
                <span style="display:inline-block; width:12px; height:12px; background:var(--warn-bg); border:1px solid var(--hair); margin-right:4px;"></span> Significant before correction only (lost after adjustment)
            </p>
        </div>
'''
                    # Show group statistics summary (sorted: Normal first, then natural sort, ROD last)
                    group_stats = result.get('group_statistics', {})
                    if group_stats:
                        # Sort groups: 'normal' first, then natural sort, 'rod' last
                        def sort_key(name):
                            import re
                            name_str = str(name).lower()
                            if name_str == 'normal':
                                return (0, [], str(name))
                            elif name_str == 'rod':
                                return (2, [], str(name))
                            else:
                                # Natural sort for other groups
                                parts = re.split(r'(\d+)', str(name))
                                converted = []
                                for part in parts:
                                    if part.isdigit():
                                        converted.append((0, int(part)))
                                    elif part:
                                        converted.append((1, part.lower()))
                                return (1, converted, str(name))
                        
                        sorted_groups = sorted(group_stats.keys(), key=sort_key)
                        
                        html += '''        <div style="margin-top:15px;">
            <p><strong>Group Statistics:</strong></p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Group</th>
                        <th class="num">n</th>
                        <th class="num">Mean ± SD</th>
                        <th class="num">Median</th>
                        <th class="num">CV (%)</th>
                    </tr>
                </thead>
                <tbody>
'''
                        for gname in sorted_groups:
                            gstat = group_stats[gname]
                            mean_val = gstat.get('mean', 0)
                            std_val = gstat.get('std', 0)
                            median_val = gstat.get('median', 0)
                            cv_val = gstat.get('cv', 0)
                            n_val = gstat.get('n', 0)
                            
                            html += f'''                    <tr>
                        <td>{gname}</td>
                        <td class="num">{n_val}</td>
                        <td class="num">{mean_val:.3f} ± {std_val:.3f}</td>
                        <td class="num">{median_val:.3f}</td>
                        <td class="num">{cv_val:.1f}</td>
                    </tr>
'''
                        html += '''                </tbody>
            </table>
        </div>
'''
                else:
                    # 2 groups: Direct comparison - sort Normal first
                    g1_name = result.get('group1_name', 'Group1')
                    g2_name = result.get('group2_name', 'Group2')
                    g1_mean = result.get('mean1', 0)
                    g1_std = result.get('std1', 0)
                    g2_mean = result.get('mean2', 0)
                    g2_std = result.get('std2', 0)
                    
                    # Sort: Normal first
                    if str(g2_name).lower() == 'normal' and str(g1_name).lower() != 'normal':
                        g1_name, g2_name = g2_name, g1_name
                        g1_mean, g2_mean = g2_mean, g1_mean
                        g1_std, g2_std = g2_std, g1_std
                    
                    html += f'''        <div class="result-box">
            <p><strong>Two-Group Comparison:</strong> {result.get('test_name', 'N/A')}</p>
            <table class="data-table">
                <tr>
                    <td><strong>{g1_name}</strong></td>
                    <td class="num">{g1_mean:.3f} ± {g1_std:.3f}</td>
                </tr>
                <tr>
                    <td><strong>{g2_name}</strong></td>
                    <td class="num">{g2_mean:.3f} ± {g2_std:.3f}</td>
                </tr>
            </table>
            <p><strong>p-value:</strong> {result.get('p_value', 0):.4f} &nbsp;·&nbsp;
            {'<span class="verdict verdict--sig">Significant</span>' if result.get('significant', False) else '<span class="verdict verdict--ns">Not significant</span>'}</p>
            <p><strong>Effect size (Cohen's d):</strong> {result.get('cohens_d', 0):.2f} ({result.get('effect_size', 'N/A')})</p>
        </div>
'''
            html += '    </div>\n'
        return html

    def _html_spatial_section(self, spatial_stats: Dict) -> str:
        """Spatial analysis summary (nearest-neighbour, Clark-Evans, clustering)."""
        html = ""
        # Spatial statistics
        if spatial_stats:
            html += f'''
    <div class="section">
        <h2>Spatial Analysis</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{spatial_stats.get('mean_nnd', 0):.1f}</div>
                <div class="label">Mean NND (pixels)</div>
            </div>
            <div class="stat-card">
                <div class="value">{spatial_stats.get('mean_clark_evans', 0):.2f}</div>
                <div class="label">Clark-Evans R</div>
            </div>
            <div class="stat-card">
                <div class="value">{spatial_stats.get('mean_edge_fraction', 0)*100:.1f}%</div>
                <div class="label">Edge Fraction</div>
            </div>
        </div>
        <div class="callout callout--warn">
            <strong>Clustering:</strong>
            {spatial_stats.get('n_clustered', 0)} clustered, 
            {spatial_stats.get('n_random', 0)} random, 
            {spatial_stats.get('n_dispersed', 0)} dispersed 
            (out of {spatial_stats.get('n_images', 0)} images)
        </div>
    </div>
'''
        return html

    def _html_film_table(self, film_summary: pd.DataFrame) -> str:
        """Per-image summary table."""
        html = ""
        # Film summary table
        html += '''
    <div class="section">
        <h2>Image Summary</h2>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Filename</th>
                        <th>Normal</th>
                        <th>ROD</th>
                        <th>ROD %</th>
                        <th>Total IOD</th>
                    </tr>
                </thead>
                <tbody>
'''
        for _, row in film_summary.iterrows():
            n_normal = int(row.get('n_normal', 0)) if 'n_normal' in row.index else 0
            n_rod = int(row.get('n_rod', 0)) if 'n_rod' in row.index else 0
            rod_fraction = row.get('rod_fraction', 0) if 'rod_fraction' in row.index else 0
            total_iod = row.get('total_iod', 0) if 'total_iod' in row.index else 0
            
            html += f'''                    <tr>
                        <td>{row.get('filename', 'N/A')}</td>
                        <td>{n_normal}</td>
                        <td>{n_rod}</td>
                        <td>{rod_fraction*100:.1f}%</td>
                        <td>{total_iod:.0f}</td>
                    </tr>
'''
        
        html += '''                </tbody>
            </table>
        </div>
    </div>
'''
        return html

    def _html_methods(self) -> str:
        """Methods appendix — conditional, honest wording (no false blanket claims)."""
        return '''
    <div class="section">
      <h2>Appendix — Methods</h2>
      <p class="section-intro">How the numbers were produced.</p>
      <p><b>Detection &amp; classification.</b> Deposits are detected, then a Random-Forest (or
      rule-based) classifier labels each Normal, ROD, or Artifact. ROD Fraction = ROD / (Normal + ROD);
      Artifacts are the reject class, excluded from deposit counts and metrics.</p>
      <p><b>Confidence.</b> The per-deposit confidence is the classifier score (the RF class
      probability, or a circularity-derived score in rule-based mode). It is <b>uncalibrated</b> — not
      a calibrated probability of correctness — and covers classification only, not detection. The
      "below the confidence-score threshold" counts are a review/workload signal, not a reliability
      measure.</p>
      <p><b>Statistics.</b> Group comparisons test <b>image-level</b> aggregates (the experimental unit
      is the image, not the deposit — avoiding pseudoreplication). The omnibus test is chosen by
      normality and group count (one-way ANOVA or Kruskal-Wallis for three or more groups; an
      independent t-test or Mann-Whitney U for two), with Holm-corrected pairwise comparisons when
      there are more than two groups; effect sizes are reported alongside.</p>
    </div>
'''

    def generate_pdf_report(
        self,
        film_summary: pd.DataFrame,
        **kwargs
    ) -> Optional[str]:
        """
        Generate PDF report (requires weasyprint or pdfkit).
        Falls back to HTML if PDF libraries not available.
        """
        # First generate HTML
        html_path = self.generate_html_report(film_summary, **kwargs)
        
        # Try to convert to PDF
        pdf_path = self.output_dir / 'report.pdf'
        
        try:
            import weasyprint
            weasyprint.HTML(html_path).write_pdf(str(pdf_path))
            return str(pdf_path)
        except ImportError:
            pass
        
        try:
            import pdfkit
            pdfkit.from_file(html_path, str(pdf_path))
            return str(pdf_path)
        except ImportError:
            pass
        
        print("PDF generation requires 'weasyprint' or 'pdfkit'. HTML report generated instead.")
        return html_path


def generate_report(
    film_summary: pd.DataFrame,
    output_dir: Union[str, Path],
    deposit_data: pd.DataFrame = None,
    spatial_stats: Dict = None,
    statistical_results: Dict = None,
    visualization_paths: Dict = None,
    group_by: str = None,
    format: str = 'html',
    analysis: dict = None
) -> str:
    """
    Convenience function to generate report.
    
    Args:
        film_summary: Film-level summary DataFrame
        output_dir: Output directory
        format: 'html' or 'pdf'
        
    Returns:
        Path to generated report
    """
    generator = ReportGenerator(output_dir)
    
    if format == 'pdf':
        return generator.generate_pdf_report(
            film_summary=film_summary,
            deposit_data=deposit_data,
            spatial_stats=spatial_stats,
            statistical_results=statistical_results,
            visualization_paths=visualization_paths,
            group_by=group_by,
            analysis=analysis
        )
    else:
        return generator.generate_html_report(
            film_summary=film_summary,
            deposit_data=deposit_data,
            spatial_stats=spatial_stats,
            statistical_results=statistical_results,
            visualization_paths=visualization_paths,
            group_by=group_by,
            analysis=analysis
        )
