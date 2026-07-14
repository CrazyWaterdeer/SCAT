"""
Command-line interface for SCAT.
"""

import argparse
import sys


def analyze_command(args):
    """Analyze a folder — thin adapter over the shared pipeline services."""
    import pandas as pd
    from .pipeline import analyze_folder_service, run_statistics_service, generate_report_service
    from .config import config

    groups = None
    if args.metadata:
        meta = pd.read_csv(args.metadata)
        col = args.group_by
        if col:
            col = col.split(',')[0]  # first grouping column
        elif len(meta.columns) > 1:
            col = meta.columns[1]
        if col and col in meta.columns and 'filename' in meta.columns:
            groups = dict(zip(meta['filename'], meta[col]))

    res = analyze_folder_service(
        args.input, groups=groups, model_type=args.model_type, model_path=args.model_path,
        min_area=args.min_area, max_area=args.max_area, edge_margin=args.edge_margin,
        circularity=args.threshold, annotate=args.annotate, visualize=args.visualize,
        output_dir=args.output)

    print(f"Analyzed {res.n_images} images -> {res.output_dir}")
    print(f"  Normal={res.n_normal} ROD={res.n_rod} Artifact={res.n_artifact} failed={res.n_failed}")
    for w in res.warnings:
        print(f"  ! {w}")

    stats = None
    if args.stats:
        if len(res.groups) >= 2:
            stats = run_statistics_service(res.output_dir, group_col="group")
        else:
            print(f"  stats skipped: {len(res.groups)} group(s)")

    report_default = config.get("analysis.report", True)
    if args.report if args.report is not None else report_default:
        path = generate_report_service(res.output_dir, statistical_results=stats, group_by="group")
        print(f"  report: {path}")


def chat_command(args):
    """Conversational agent: describe a folder to analyze; the agent runs the pipeline."""
    from .agent.backend import build_runner
    from .agent.provenance import start_session, set_driver
    from .agent.runner import TextDelta, ToolUse, ToolResult, TurnDone

    start_session(driver="cli-chat")
    set_driver("cli-chat")
    runner, desc = build_runner(backend=args.backend, model=args.model)
    print(f"[backend] {desc}\nType a request (Ctrl-D to exit).")
    try:
        while True:
            try:
                text = input("\n> ").strip()
            except EOFError:
                break
            if not text:
                continue
            from .progress import run_progress
            with run_progress(lambda c, t, note="": print(f"\r  [{note} {c}/{t}]   ", end="", flush=True)):
                for ev in runner.turn(text):
                    if isinstance(ev, TextDelta):
                        print(ev.text, end="", flush=True)
                    elif isinstance(ev, ToolUse):
                        print(f"\n  [tool] {ev.name}({ev.input})", flush=True)
                    elif isinstance(ev, ToolResult):
                        tag = "ERR" if ev.is_error else "ok"
                        print(f"\n  [result:{tag}] {ev.name}", flush=True)
                    elif isinstance(ev, TurnDone):
                        print(f"\n[turn done: {ev.stop_reason}]", flush=True)
    finally:
        close = getattr(runner, "close", None)
        if close:
            close()


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

    # gui
    gp = subparsers.add_parser('gui', help='Launch GUI application')
    gp.set_defaults(func=gui_command)

    # chat (conversational agent)
    cp = subparsers.add_parser('chat', help='Conversational agent (analyze a folder by asking)')
    cp.add_argument('--backend', default='auto', choices=['auto', 'subscription', 'api'])
    cp.add_argument('--model', default='claude-opus-4-8')
    cp.set_defaults(func=chat_command)

    # analyze
    ap = subparsers.add_parser('analyze', help='Analyze images')
    ap.add_argument('input', help='Input image or directory')
    ap.add_argument('-o', '--output', default=None, help='Output dir (default: timestamped)')
    ap.add_argument('-m', '--metadata', help='Metadata CSV')
    ap.add_argument('--model-type', default=None, choices=['threshold', 'rf', 'cnn'],
                    help='Classifier (default: rf if a model exists, else threshold)')
    ap.add_argument('--model-path')
    ap.add_argument('--threshold', type=float, default=0.6)
    ap.add_argument('--min-area', type=int, default=20)
    ap.add_argument('--max-area', type=int, default=10000)
    ap.add_argument('--edge-margin', type=int, default=20)
    ap.add_argument('--group-by', help='Column(s) for grouping, comma-separated')
    ap.add_argument('--annotate', action='store_true', help='Generate annotated images')
    ap.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    ap.add_argument('--stats', action='store_true', help='Perform statistical analysis')
    ap.add_argument('--report', dest='report', action='store_true', default=None, help='Generate HTML report')
    ap.add_argument('--no-report', dest='report', action='store_false', help='Skip the HTML report')
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
