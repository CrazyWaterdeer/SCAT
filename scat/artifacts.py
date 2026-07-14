"""Canonical output-artifact filenames — the single source of truth.

The analysis writer (``analyzer.ReportGenerator``) and the readers that must agree with it
(``results_index`` discovery/resume, ``combine``, ``pipeline`` stats/report, ``manifest``) all
reference these names. Keeping them here means a rename can't silently break discovery/combine/resume
by drifting one writer or reader out of sync. Values are frozen — changing a string here changes the
on-disk contract.
"""
IMAGE_SUMMARY = "image_summary.csv"
ALL_DEPOSITS = "all_deposits.csv"
CONDITION_SUMMARY = "condition_summary.csv"
RUN_MANIFEST = "run_manifest.json"
