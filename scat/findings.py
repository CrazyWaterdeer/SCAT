# scat/findings.py
"""Compose the report's answer-first finding (spec §2.1/§5). PURE. Conservative + verdict-driven:
never "did not differ"/causal; omnibus (3+ groups) says "across … groups"; uses ONLY the predeclared
primary metric; descriptive when ungrouped/single-group/stats missing."""
from __future__ import annotations

from scat import metrics as _metrics

# primary-metric -> stats key mapping. Single source of truth in metrics.py (the report imports it
# too); aliased here for this module's internal use. The FALLBACK covers area/hue/circularity, which
# the stats module compares only per-class (normal_*/rod_*) — see metrics.STATS_KEY_FALLBACK.
_STATS_KEY = _metrics.STATS_KEY
_STATS_KEY_FALLBACK = _metrics.STATS_KEY_FALLBACK


def _fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _lookup_comparison(stats, pm):
    for key in (_STATS_KEY.get(pm), _STATS_KEY_FALLBACK.get(pm)):
        if key:
            e = stats.get(key)
            if isinstance(e, dict) and "error" not in e:
                return e
    return None


def _primary_comparison(stats, primary_metric):
    """{test, p, significant, is_omnibus} for the primary metric, or None if unavailable/skipped."""
    if not isinstance(stats, dict) or stats.get("skipped"):
        return None
    entry = _lookup_comparison(stats, _metrics.resolve_metric(primary_metric))
    if not isinstance(entry, dict) or "error" in entry:
        return None
    if "overall_test" in entry:                       # 3+ groups (omnibus)
        test, p = entry.get("overall_test"), entry.get("overall_p_value")
        sig, omni = bool(entry.get("overall_significant")), True
    else:                                             # two groups
        test, p = entry.get("test_name"), entry.get("p_value")
        sig, omni = bool(entry.get("significant")), False
    if test is None or p is None:
        return None
    return {"test": test, "p": float(p), "significant": sig, "is_omnibus": omni}


def compose_finding(*, stats, primary_metric, headline, n_images, n_groups, group_label) -> dict:
    label = _metrics.METRICS[_metrics.resolve_metric(primary_metric)].label
    comp = _primary_comparison(stats, primary_metric) if (n_groups or 0) >= 2 else None
    if comp is None:
        return {"sentence": f"Across {n_images} images, {label} averaged {headline}.",
                "metric": label, "test": "no group comparison", "scope": f"{n_images} images"}
    gl = group_label or "group"
    grp = "groups" if gl in ("group", "groups") else f"{gl} groups"   # avoid "across group groups"
    pstr = _fmt_p(comp["p"])
    if comp["significant"]:
        verb = (f"showed a difference across {grp}" if comp["is_omnibus"]
                else f"differed between the {grp}")
    else:
        conn = "across" if comp["is_omnibus"] else "between"
        verb = f"showed no statistically detected difference {conn} {grp}"
    return {"sentence": f"{label} {verb} ({comp['test']}, {pstr}).",
            "metric": f"{label} · {headline}",
            "test": f"{comp['test']}, {pstr}" + (" (significant)" if comp["significant"] else " (n.s.)"),
            "scope": f"{n_images} images · {n_groups} groups"}
