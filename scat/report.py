"""
Report generation module for SCAT.
Generates HTML and PDF reports from analysis results.
"""

import base64
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from io import BytesIO

# Check for optional dependencies
try:
    from jinja2 import Template
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import color constants from visualization for consistency
from .visualization import (
    DEFAULT_GRAY, PASTEL_PALETTE, CONTROL_COLOR, DEPOSIT_COLORS,
    get_palette, apply_publication_style
)


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
        comprehensive_stats: Dict = None
    ) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            comprehensive_stats: Optional dict with comprehensive analysis results
                                 from run_comprehensive_analysis()
        
        Returns:
            Path to generated HTML file
        """
        # Calculate summary statistics
        summary = self._calculate_summary(film_summary)
        
        # Generate inline plots
        inline_plots = {}
        if HAS_MATPLOTLIB:
            inline_plots['overview_bar'] = self._generate_overview_bar(film_summary)
            inline_plots['rod_distribution'] = self._generate_rod_histogram(film_summary)
            if group_by and group_by in film_summary.columns:
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
            comprehensive_stats=comprehensive_stats
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
        
        return {
            'total_films': len(film_summary),
            'total_deposits': safe_sum('n_total'),
            'total_normal': safe_sum('n_normal'),
            'total_rod': safe_sum('n_rod'),
            'total_artifact': safe_sum('n_artifact'),
            'mean_rod_fraction': safe_mean('rod_fraction'),
            'std_rod_fraction': safe_std('rod_fraction'),
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
        
        totals = {
            'Normal': safe_sum('n_normal'),
            'ROD': safe_sum('n_rod'),
            'Artifact': safe_sum('n_artifact')
        }
        
        # Use DEPOSIT_COLORS for consistency with visualization module
        colors = [DEPOSIT_COLORS['normal'], DEPOSIT_COLORS['rod'], DEPOSIT_COLORS['artifact']]
        bars = ax.bar(totals.keys(), totals.values(), color=colors)
        
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
    
    def _generate_rod_histogram(self, film_summary: pd.DataFrame) -> str:
        """Generate ROD fraction histogram as base64."""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.hist(film_summary['rod_fraction'] * 100, bins=15, 
                color=DEFAULT_GRAY, edgecolor='white', alpha=0.8)
        ax.axvline(film_summary['rod_fraction'].mean() * 100, 
                   color='#E53935', linestyle='--', linewidth=2, label='Mean')
        
        ax.set_xlabel('ROD Fraction (%)')
        ax.set_ylabel('Number of Films')
        ax.set_title('Distribution of ROD Fraction')
        ax.legend()
        
        # Apply publication style
        apply_publication_style(ax)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _generate_group_comparison(
        self, 
        film_summary: pd.DataFrame, 
        group_by: str
    ) -> str:
        """Generate group comparison box plot as base64."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        groups = list(film_summary[group_by].unique())
        data = [film_summary[film_summary[group_by] == g]['rod_fraction'] * 100 
                for g in groups]
        
        bp = ax.boxplot(data, labels=groups, patch_artist=True)
        
        # Use PASTEL_PALETTE for consistency with visualization module
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(PASTEL_PALETTE[i % len(PASTEL_PALETTE)])
            patch.set_alpha(0.8)
        
        ax.set_ylabel('ROD Fraction (%)')
        ax.set_xlabel(group_by)
        ax.set_title(f'ROD Fraction by {group_by}')
        
        # Apply publication style
        apply_publication_style(ax)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
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
        comprehensive_stats: Dict = None
    ) -> str:
        """Build complete HTML document."""
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            color: #1a237e;
            border-bottom: 3px solid #3949ab;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #3949ab;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #1a237e;
        }}
        .stat-card .label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .stat-card.normal {{ border-left-color: #4CAF50; }}
        .stat-card.normal .value {{ color: #4CAF50; }}
        .stat-card.rod {{ border-left-color: #F44336; }}
        .stat-card.rod .value {{ color: #F44336; }}
        .stat-card.artifact {{ border-left-color: #9E9E9E; }}
        .stat-card.artifact .value {{ color: #9E9E9E; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.9em;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #1a237e;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 768px) {{
            .two-column {{
                grid-template-columns: 1fr;
            }}
        }}
        .highlight {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="subtitle">Generated: {summary['generated_at']}</div>
    </div>
    
    <!-- Summary Section -->
    <div class="section">
        <h2>📊 Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{summary['total_films']}</div>
                <div class="label">Total Films</div>
            </div>
            <div class="stat-card">
                <div class="value">{summary['total_deposits']}</div>
                <div class="label">Total Deposits</div>
            </div>
            <div class="stat-card normal">
                <div class="value">{summary['total_normal']}</div>
                <div class="label">Normal</div>
            </div>
            <div class="stat-card rod">
                <div class="value">{summary['total_rod']}</div>
                <div class="label">ROD</div>
            </div>
            <div class="stat-card artifact">
                <div class="value">{summary['total_artifact']}</div>
                <div class="label">Artifact</div>
            </div>
            <div class="stat-card">
                <div class="value">{summary['mean_rod_fraction']*100:.1f}%</div>
                <div class="label">Mean ROD Fraction (±{summary['std_rod_fraction']*100:.1f}%)</div>
            </div>
        </div>
'''
        
        # Add inline plots
        if 'overview_bar' in inline_plots:
            html += f'''
        <div class="two-column">
            <div class="plot-container">
                <img src="data:image/png;base64,{inline_plots['overview_bar']}" alt="Overview">
            </div>
            <div class="plot-container">
                <img src="data:image/png;base64,{inline_plots['rod_distribution']}" alt="ROD Distribution">
            </div>
        </div>
'''
        
        html += '    </div>\n'
        
        # Group comparison
        if group_by and 'group_comparison' in inline_plots:
            group_label = self.get_metric_label(group_by)
            html += f'''
    <div class="section">
        <h2>📈 Condition Comparison ({group_label})</h2>
        <div class="plot-container">
            <img src="data:image/png;base64,{inline_plots['group_comparison']}" alt="Group Comparison">
        </div>
    </div>
'''
        
        # Statistical results
        if statistical_results and isinstance(statistical_results, dict):
            html += '''
    <div class="section">
        <h2>📉 Statistical Analysis</h2>
'''
            for metric, result in statistical_results.items():
                # Skip if result is not a dict or has error
                if not isinstance(result, dict):
                    continue
                if 'error' in result:
                    continue
                
                # Convert metric name to human-readable label
                metric_label = self.get_metric_label(metric)
                html += f'        <h3 style="color:#3949ab; margin-top:20px;">{metric_label}</h3>\n'
                
                if 'overall_test' in result:
                    # 3+ groups: Omnibus test + pairwise comparisons
                    n_groups = result.get('n_groups', 0)
                    group_names = result.get('group_names', [])
                    
                    html += f'''        <div style="background:#f5f5f5; padding:10px; border-radius:5px; margin-bottom:10px;">
            <p><strong>🔬 Omnibus Test ({n_groups} groups):</strong> {result['overall_test']}</p>
            <p style="margin-left:20px; color:#666;">
                <em>Tests whether at least one group differs from the others</em>
            </p>
            <p><strong>p-value:</strong> {result['overall_p_value']:.4f} 
            {'✅ At least one group is different' if result['overall_significant'] else '❌ No overall difference detected'}</p>
        </div>
'''
                    # Show pairwise comparisons if available
                    pairwise = result.get('pairwise_comparisons', [])
                    if pairwise:
                        correction = result.get('correction_method', 'none')
                        correction_label = f" ({correction.capitalize()} corrected)" if correction != 'none' else ""
                        
                        # Check if correction was applied
                        has_correction = any('p_value_corrected' in pw for pw in pairwise if 'error' not in pw)
                        
                        html += f'''        <div style="margin-top:15px;">
            <p><strong>🔄 Pairwise Comparisons{correction_label}:</strong></p>
            <p style="margin-left:20px; color:#666; margin-bottom:10px;">
                <em>Tests which specific group pairs are different</em>
            </p>
            <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
                <thead>
                    <tr style="background:#e8eaf6;">
                        <th style="padding:8px; text-align:left; border:1px solid #ddd;">Comparison</th>
                        <th style="padding:8px; text-align:center; border:1px solid #ddd;">Test</th>
                        <th style="padding:8px; text-align:center; border:1px solid #ddd;">p (raw)</th>
                        <th style="padding:8px; text-align:center; border:1px solid #ddd;">{"p (corrected)" if has_correction else "p-value"}</th>
                        <th style="padding:8px; text-align:center; border:1px solid #ddd;">Effect Size</th>
                        <th style="padding:8px; text-align:center; border:1px solid #ddd;">Result</th>
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
                            
                            sig_icon = '✅' if is_sig else '❌'
                            # Highlight row if raw p-value was significant but corrected wasn't
                            raw_sig = p_raw < 0.05
                            if raw_sig and not is_sig:
                                row_bg = '#fff3e0'  # Orange tint - significant before correction
                            elif is_sig:
                                row_bg = '#e8f5e9'  # Green - significant after correction
                            else:
                                row_bg = '#fff'  # White - not significant
                            
                            html += f'''                    <tr style="background:{row_bg};">
                        <td style="padding:8px; border:1px solid #ddd;">{g1} vs {g2}</td>
                        <td style="padding:8px; text-align:center; border:1px solid #ddd;">{test_name}</td>
                        <td style="padding:8px; text-align:center; border:1px solid #ddd;">{p_raw:.4f}</td>
                        <td style="padding:8px; text-align:center; border:1px solid #ddd;">{p_display}</td>
                        <td style="padding:8px; text-align:center; border:1px solid #ddd;">{effect_d:.2f} ({effect_label})</td>
                        <td style="padding:8px; text-align:center; border:1px solid #ddd;">{sig_icon}</td>
                    </tr>
'''
                        html += '''                </tbody>
            </table>
            <p style="margin-top:8px; font-size:0.85em; color:#666;">
                <span style="display:inline-block; width:12px; height:12px; background:#e8f5e9; border:1px solid #ddd; margin-right:4px;"></span> Significant after correction &nbsp;
                <span style="display:inline-block; width:12px; height:12px; background:#fff3e0; border:1px solid #ddd; margin-right:4px;"></span> Significant before correction only (lost after adjustment)
            </p>
        </div>
'''
                    # Show group statistics summary
                    group_stats = result.get('group_statistics', {})
                    if group_stats:
                        html += '''        <div style="margin-top:15px;">
            <p><strong>📊 Group Statistics:</strong></p>
            <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
                <thead>
                    <tr style="background:#e8eaf6;">
                        <th style="padding:8px; text-align:left; border:1px solid #ddd;">Group</th>
                        <th style="padding:8px; text-align:center; border:1px solid #ddd;">n</th>
                        <th style="padding:8px; text-align:center; border:1px solid #ddd;">Mean ± SD</th>
                        <th style="padding:8px; text-align:center; border:1px solid #ddd;">Median</th>
                        <th style="padding:8px; text-align:center; border:1px solid #ddd;">CV (%)</th>
                    </tr>
                </thead>
                <tbody>
'''
                        for gname, gstat in group_stats.items():
                            mean_val = gstat.get('mean', 0)
                            std_val = gstat.get('std', 0)
                            median_val = gstat.get('median', 0)
                            cv_val = gstat.get('cv', 0)
                            n_val = gstat.get('n', 0)
                            
                            html += f'''                    <tr>
                        <td style="padding:8px; border:1px solid #ddd;">{gname}</td>
                        <td style="padding:8px; text-align:center; border:1px solid #ddd;">{n_val}</td>
                        <td style="padding:8px; text-align:center; border:1px solid #ddd;">{mean_val:.3f} ± {std_val:.3f}</td>
                        <td style="padding:8px; text-align:center; border:1px solid #ddd;">{median_val:.3f}</td>
                        <td style="padding:8px; text-align:center; border:1px solid #ddd;">{cv_val:.1f}</td>
                    </tr>
'''
                        html += '''                </tbody>
            </table>
        </div>
'''
                else:
                    # 2 groups: Direct comparison
                    html += f'''        <div style="background:#f5f5f5; padding:10px; border-radius:5px;">
            <p><strong>🔬 Two-Group Comparison:</strong> {result.get('test_name', 'N/A')}</p>
            <table style="width:100%; border-collapse:collapse; font-size:0.9em; margin:10px 0;">
                <tr>
                    <td style="padding:8px; border:1px solid #ddd;"><strong>{result.get('group1_name', 'Group1')}</strong></td>
                    <td style="padding:8px; text-align:center; border:1px solid #ddd;">{result.get('mean1', 0):.3f} ± {result.get('std1', 0):.3f}</td>
                </tr>
                <tr>
                    <td style="padding:8px; border:1px solid #ddd;"><strong>{result.get('group2_name', 'Group2')}</strong></td>
                    <td style="padding:8px; text-align:center; border:1px solid #ddd;">{result.get('mean2', 0):.3f} ± {result.get('std2', 0):.3f}</td>
                </tr>
            </table>
            <p><strong>p-value:</strong> {result.get('p_value', 0):.4f} 
            {'✅ Significant difference' if result.get('significant', False) else '❌ No significant difference'}</p>
            <p><strong>Effect size (Cohen's d):</strong> {result.get('cohens_d', 0):.2f} ({result.get('effect_size', 'N/A')})</p>
        </div>
'''
            html += '    </div>\n'
        
        # Spatial statistics
        if spatial_stats:
            html += f'''
    <div class="section">
        <h2>🗺️ Spatial Analysis</h2>
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
        <div class="highlight">
            <strong>Clustering:</strong> 
            {spatial_stats.get('n_clustered', 0)} clustered, 
            {spatial_stats.get('n_random', 0)} random, 
            {spatial_stats.get('n_dispersed', 0)} dispersed 
            (out of {spatial_stats.get('n_images', 0)} images)
        </div>
    </div>
'''
        
        # Film summary table
        html += '''
    <div class="section">
        <h2>📋 Film Summary</h2>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Filename</th>
                        <th>Normal</th>
                        <th>ROD</th>
                        <th>Artifact</th>
                        <th>ROD %</th>
                        <th>Total IOD</th>
                    </tr>
                </thead>
                <tbody>
'''
        for _, row in film_summary.iterrows():
            n_normal = int(row.get('n_normal', 0)) if 'n_normal' in row.index else 0
            n_rod = int(row.get('n_rod', 0)) if 'n_rod' in row.index else 0
            n_artifact = int(row.get('n_artifact', 0)) if 'n_artifact' in row.index else 0
            rod_fraction = row.get('rod_fraction', 0) if 'rod_fraction' in row.index else 0
            total_iod = row.get('total_iod', 0) if 'total_iod' in row.index else 0
            
            html += f'''                    <tr>
                        <td>{row.get('filename', 'N/A')}</td>
                        <td>{n_normal}</td>
                        <td>{n_rod}</td>
                        <td>{n_artifact}</td>
                        <td>{rod_fraction*100:.1f}%</td>
                        <td>{total_iod:.0f}</td>
                    </tr>
'''
        
        html += '''                </tbody>
            </table>
        </div>
    </div>
'''
        
        # Comprehensive Statistics Section
        if comprehensive_stats and isinstance(comprehensive_stats, dict):
            html += self._build_comprehensive_stats_html(comprehensive_stats)
        
        html += '''    
    <div class="footer">
        <p>Generated by SCAT - Spot Classification and Analysis Tool</p>
        <p>Anthropic Claude AI Assistant</p>
    </div>
</body>
</html>'''
        
        return html
    
    def _build_comprehensive_stats_html(self, stats: Dict) -> str:
        """Build HTML section for comprehensive statistics."""
        html = '''
    <div class="section">
        <h2>🔬 Comprehensive Analysis</h2>
'''
        
        # Summary section
        if 'summary' in stats:
            summary = stats['summary']
            
            # Key metrics
            if summary.get('key_metrics'):
                html += '''
        <h3 style="color:#3949ab; margin-top:20px;">Key Metrics</h3>
        <div class="stats-grid">
'''
                metrics_display = {
                    'mean_estimated_ph': ('Est. pH', ''),
                    'fraction_acidic': ('Acidic Fraction', '%', 100),
                    'total_iod': ('Total IOD', ''),
                    'mean_deposit_area': ('Mean Area', ' px²'),
                    'mean_deposits_per_film': ('Deposits/Film', ''),
                    'mean_rod_fraction': ('ROD Fraction', '%', 100),
                    'mean_circularity': ('Mean Circularity', ''),
                }
                
                for key, display_info in metrics_display.items():
                    if key in summary['key_metrics']:
                        val = summary['key_metrics'][key]
                        if len(display_info) == 3:
                            val = val * display_info[2]
                        label, suffix = display_info[0], display_info[1]
                        html += f'''            <div class="stat-card">
                <div class="value">{val:.2f}{suffix}</div>
                <div class="label">{label}</div>
            </div>
'''
                html += '        </div>\n'
            
            # Significant findings
            if summary.get('significant_findings'):
                html += '''
        <h3 style="color:#3949ab; margin-top:20px;">⭐ Significant Findings</h3>
        <ul style="margin-left: 20px;">
'''
                for finding in summary['significant_findings'][:10]:  # Limit to 10
                    p_val = finding.get('p_value', 0)
                    # Format the finding text - extract metric name and format it
                    finding_text = finding.get("finding", "")
                    category = finding.get("category", "").title()
                    
                    # Try to extract and format metric name from finding text
                    # Pattern: "metric_name differs significantly..."
                    if ' differs significantly' in finding_text:
                        parts = finding_text.split(' differs significantly')
                        if parts[0]:
                            metric_name = parts[0].strip()
                            formatted_metric = self.get_metric_display_name(metric_name)
                            finding_text = f"{formatted_metric} differs significantly{parts[1] if len(parts) > 1 else ''}"
                    
                    html += f'            <li><strong>{category}</strong>: {finding_text} (p={p_val:.4f})</li>\n'
                html += '        </ul>\n'
            
            # Recommendations
            if summary.get('recommendations'):
                html += '''
        <h3 style="color:#3949ab; margin-top:20px;">💡 Recommendations</h3>
        <ul style="margin-left: 20px;">
'''
                for rec in summary['recommendations']:
                    html += f'            <li>{rec}</li>\n'
                html += '        </ul>\n'
        
        # pH Analysis
        if 'ph' in stats and 'error' not in stats['ph']:
            ph_data = stats['ph']
            if 'deposit_level' in ph_data:
                dl = ph_data['deposit_level']
                html += f'''
        <h3 style="color:#3949ab; margin-top:20px;">🧪 pH Analysis (BPB-based)</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{dl.get('mean_ph', 0):.2f}</div>
                <div class="label">Mean Est. pH</div>
            </div>
            <div class="stat-card">
                <div class="value">{dl.get('fraction_acidic', 0)*100:.1f}%</div>
                <div class="label">Acidic</div>
            </div>
            <div class="stat-card">
                <div class="value">{dl.get('fraction_transitional', 0)*100:.1f}%</div>
                <div class="label">Transitional</div>
            </div>
            <div class="stat-card">
                <div class="value">{dl.get('fraction_basic', 0)*100:.1f}%</div>
                <div class="label">Basic</div>
            </div>
        </div>
'''
        
        # Size Distribution
        if 'size_distribution' in stats and 'error' not in stats['size_distribution']:
            size_data = stats['size_distribution']
            if 'distribution' in size_data:
                dist = size_data['distribution']
                html += f'''
        <h3 style="color:#3949ab; margin-top:20px;">📏 Size Distribution</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{dist.get('mean_area', 0):.0f}</div>
                <div class="label">Mean Area (px²)</div>
            </div>
            <div class="stat-card">
                <div class="value">{dist.get('fraction_small', 0)*100:.0f}%</div>
                <div class="label">Small</div>
            </div>
            <div class="stat-card">
                <div class="value">{dist.get('fraction_medium', 0)*100:.0f}%</div>
                <div class="label">Medium</div>
            </div>
            <div class="stat-card">
                <div class="value">{dist.get('fraction_large', 0)*100:.0f}%</div>
                <div class="label">Large</div>
            </div>
        </div>
'''
                if dist.get('is_bimodal'):
                    html += '        <div class="highlight">⚠️ Size distribution appears bimodal</div>\n'
        
        # Morphology
        if 'morphology' in stats and 'error' not in stats['morphology']:
            morph_data = stats['morphology']
            if 'distribution' in morph_data:
                dist = morph_data['distribution']
                html += f'''
        <h3 style="color:#3949ab; margin-top:20px;">🔷 Morphology</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{dist.get('mean_circularity', 0):.2f}</div>
                <div class="label">Mean Circularity</div>
            </div>
            <div class="stat-card">
                <div class="value">{dist.get('mean_aspect_ratio', 0):.2f}</div>
                <div class="label">Mean Aspect Ratio</div>
            </div>
            <div class="stat-card">
                <div class="value">{dist.get('fraction_regular', 0)*100:.0f}%</div>
                <div class="label">Regular Shape</div>
            </div>
            <div class="stat-card">
                <div class="value">{dist.get('fraction_irregular', 0)*100:.0f}%</div>
                <div class="label">Irregular Shape</div>
            </div>
        </div>
'''
        
        # Correlation highlights
        if 'correlation' in stats and 'error' not in stats['correlation']:
            corr_data = stats['correlation']
            if 'key_correlations' in corr_data:
                strong_corrs = [(k, v) for k, v in corr_data['key_correlations'].items() 
                               if isinstance(v, dict) and v.get('significant') and abs(v.get('r', 0)) >= 0.3]
                
                if strong_corrs:
                    html += '''
        <h3 style="color:#3949ab; margin-top:20px;">🔗 Key Correlations</h3>
        <table style="max-width: 600px;">
            <thead>
                <tr><th>Relationship</th><th>r</th><th>Interpretation</th></tr>
            </thead>
            <tbody>
'''
                    for key, corr in strong_corrs[:5]:  # Top 5
                        label = self.get_correlation_label(key)
                        interpretation = self.format_correlation_interpretation(corr.get('interpretation', ''))
                        html += f'''                <tr>
                    <td>{label}</td>
                    <td>{corr['r']:.2f}</td>
                    <td>{interpretation}</td>
                </tr>
'''
                    html += '''            </tbody>
        </table>
'''
        
        html += '    </div>\n'
        return html
    
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
    format: str = 'html'
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
            group_by=group_by
        )
    else:
        return generator.generate_html_report(
            film_summary=film_summary,
            deposit_data=deposit_data,
            spatial_stats=spatial_stats,
            statistical_results=statistical_results,
            visualization_paths=visualization_paths,
            group_by=group_by
        )
