"""
Command-line interface for SCAT.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from .analyzer import Analyzer, ReportGenerator
from .classifier import ClassifierConfig
from .detector import DepositDetector


def analyze_command(args):
    config = ClassifierConfig(
        model_type=args.model_type,
        circularity_threshold=args.threshold,
        model_path=args.model_path
    )
    detector = DepositDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        edge_margin=args.edge_margin
    )
    analyzer = Analyzer(detector=detector, classifier_config=config)
    
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff'))
        image_paths += list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    
    if not image_paths:
        print(f"No images found in {input_path}")
        sys.exit(1)
    
    print(f"Found {len(image_paths)} images")
    
    metadata = pd.read_csv(args.metadata) if args.metadata else None
    
    def progress(current, total):
        print(f"\rProcessing: {current}/{total}", end='', flush=True)
    
    results = analyzer.analyze_batch(image_paths, metadata, progress_callback=progress)
    print()
    
    output_dir = Path(args.output)
    reporter = ReportGenerator(output_dir)
    group_by = args.group_by.split(',') if args.group_by else None
    reports = reporter.save_all(results, metadata, group_by)
    
    # Annotated images
    if args.annotate:
        from PIL import Image
        import numpy as np
        annotated_dir = output_dir / 'annotated'
        annotated_dir.mkdir(exist_ok=True)
        for path, result in zip(image_paths, results):
            img = np.array(Image.open(path))
            annotated = analyzer.generate_annotated_image(
                img, result.deposits, show_labels=True, skip_artifacts=True
            )
            Image.fromarray(annotated).save(annotated_dir / f"{path.stem}_annotated.png")
    
    # Visualizations
    if args.visualize:
        try:
            from .visualization import generate_all_visualizations
            print("\nGenerating visualizations...")
            viz_dir = output_dir / 'visualizations'
            viz_results = generate_all_visualizations(
                reports['film_summary'],
                reports['deposit_data'],
                viz_dir,
                group_by=group_by[0] if group_by else None
            )
            print(f"  Saved {len(viz_results)} visualizations to {viz_dir}")
        except ImportError as e:
            print(f"  Visualization skipped (missing dependencies): {e}")
    
    # Statistical analysis
    if args.stats and group_by:
        try:
            from .statistics import generate_statistics_report
            print("\nStatistical analysis...")
            stats_results = generate_statistics_report(
                reports['film_summary'],
                group_column=group_by[0],
                metrics=['rod_fraction', 'n_total', 'total_iod', 'normal_mean_area', 'rod_mean_area']
            )
            
            # Save statistics report
            stats_file = output_dir / 'statistics_report.txt'
            with open(stats_file, 'w') as f:
                f.write("SCAT Statistical Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                
                for metric, result in stats_results.items():
                    f.write(f"\n{metric.upper()}\n")
                    f.write("-" * 30 + "\n")
                    
                    if 'error' in result:
                        f.write(f"  Error: {result['error']}\n")
                        continue
                    
                    if 'overall_test' in result:
                        # Multiple groups
                        f.write(f"  Overall test: {result['overall_test']}\n")
                        f.write(f"  p-value: {result['overall_p_value']:.4f}\n")
                        f.write(f"  Significant: {result['overall_significant']}\n\n")
                        
                        f.write("  Pairwise comparisons:\n")
                        for comp in result['pairwise_comparisons']:
                            f.write(f"    {comp['group1_name']} vs {comp['group2_name']}: ")
                            f.write(f"p={comp['p_value']:.4f}, d={comp['cohens_d']:.2f} ({comp['effect_size']})\n")
                    else:
                        # Two groups
                        f.write(f"  Test: {result['test_name']}\n")
                        f.write(f"  {result['group1_name']}: {result['mean1']:.3f} ± {result['std1']:.3f} (n={result['n1']})\n")
                        f.write(f"  {result['group2_name']}: {result['mean2']:.3f} ± {result['std2']:.3f} (n={result['n2']})\n")
                        f.write(f"  p-value: {result['p_value']:.4f}\n")
                        f.write(f"  Effect size (Cohen's d): {result['cohens_d']:.2f} ({result['effect_size']})\n")
                        f.write(f"  Significant: {result['significant']}\n")
            
            print(f"  Saved to {stats_file}")
        except Exception as e:
            print(f"  Statistics skipped: {e}")
    
    # Print summary
    summary = reports['film_summary']
    print(f"\n{'='*50}")
    print("ANALYSIS COMPLETE")
    print('='*50)
    print(f"  Total films: {len(results)}")
    print(f"  Total deposits: {summary['n_total'].sum():.0f}")
    print(f"  Normal: {summary['n_normal'].sum():.0f}, ROD: {summary['n_rod'].sum():.0f}, Artifact: {summary['n_artifact'].sum():.0f}")
    print(f"  Mean ROD fraction: {summary['rod_fraction'].mean():.3f} ± {summary['rod_fraction'].std():.3f}")
    print(f"\nOutput saved to: {output_dir}")
    print(f"  - film_summary.csv")
    print(f"  - deposits/ (individual files per image)")
    print(f"  - all_deposits.csv")
    if args.annotate:
        print(f"  - annotated/")
    if args.visualize:
        print(f"  - visualizations/")


def train_command(args):
    from .trainer import train_from_labels
    
    kwargs = {}
    if args.model_type == 'rf':
        kwargs['n_estimators'] = args.n_estimators
    elif args.model_type == 'cnn':
        kwargs['epochs'] = args.epochs
        kwargs['batch_size'] = args.batch_size
        kwargs['learning_rate'] = args.learning_rate
    
    train_from_labels(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_path=args.output,
        model_type=args.model_type,
        **kwargs
    )


def label_command(args):
    from .labeling_gui import run_labeling_gui
    run_labeling_gui()


def gui_command(args):
    from .main_gui import run_gui
    run_gui()


def main():
    parser = argparse.ArgumentParser(description="SCAT - Spot Classification and Analysis Tool")
    subparsers = parser.add_subparsers(dest='command')
    
    # gui (main GUI app)
    gp = subparsers.add_parser('gui', help='Launch GUI application')
    gp.set_defaults(func=gui_command)
    
    # analyze
    ap = subparsers.add_parser('analyze', help='Analyze images')
    ap.add_argument('input', help='Input image or directory')
    ap.add_argument('-o', '--output', default='./results')
    ap.add_argument('-m', '--metadata', help='Metadata CSV')
    ap.add_argument('--model-type', default='threshold', choices=['threshold', 'rf', 'cnn'])
    ap.add_argument('--model-path')
    ap.add_argument('--threshold', type=float, default=0.6)
    ap.add_argument('--min-area', type=int, default=20)
    ap.add_argument('--max-area', type=int, default=10000)
    ap.add_argument('--edge-margin', type=int, default=20)
    ap.add_argument('--group-by', help='Column(s) for grouping, comma-separated')
    ap.add_argument('--annotate', action='store_true', help='Generate annotated images')
    ap.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    ap.add_argument('--stats', action='store_true', help='Perform statistical analysis')
    ap.set_defaults(func=analyze_command)
    
    # train
    tp = subparsers.add_parser('train', help='Train classifier from labeled data')
    tp.add_argument('--image-dir', required=True, help='Directory containing images')
    tp.add_argument('--label-dir', help='Directory containing label JSONs (default: same as image-dir)')
    tp.add_argument('--output', '-o', required=True, help='Output model path')
    tp.add_argument('--model-type', default='rf', choices=['rf', 'cnn'], help='Model type')
    tp.add_argument('--n-estimators', type=int, default=100, help='Number of trees (RF)')
    tp.add_argument('--epochs', type=int, default=20, help='Training epochs (CNN)')
    tp.add_argument('--batch-size', type=int, default=32, help='Batch size (CNN)')
    tp.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (CNN)')
    tp.set_defaults(func=train_command)
    
    # label
    lp = subparsers.add_parser('label', help='Launch labeling GUI')
    lp.set_defaults(func=label_command)
    
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
