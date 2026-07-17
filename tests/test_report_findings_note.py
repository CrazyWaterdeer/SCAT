from pathlib import Path


def _report(tmp_path, synth_dir, primary="total_deposits"):
    from scat.pipeline import analyze_folder_service, generate_report_service, run_statistics_service
    groups = {f"ctrl_{i}.tif": "Control" for i in range(3)}
    groups.update({f"treat_{i}.tif": "Treatment" for i in range(3)})
    res = analyze_folder_service(str(synth_dir), output_dir=str(tmp_path / "out"),
                                 primary_metric=primary, groups=groups, annotate=False)
    rd = Path(res.output_dir)
    stats = run_statistics_service(str(rd))
    generate_report_service(str(rd), statistical_results=stats, group_by="group")
    return (rd / "report.html").read_text()


def test_html_structure_is_balanced(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir)
    assert html.count("<body") == 1 and html.count("</body>") == 1
    assert html.count("<html") == 1 and html.count("</html>") == 1
    # no premature imbalance: every <div ...> is closed
    assert html.count("<div") == html.count("</div>")


def test_finding_leads_summary_demotes_to_population_overview(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir)
    i_finding = html.find('class="finding"')
    i_group = html.find("Group Comparison")
    i_pop = html.lower().find("population overview")
    i_grid = html.find('class="stats-grid"')
    assert -1 < i_finding < i_group < i_pop < i_grid       # finding → evidence → demoted pooled cards
    # grouped run: the scope-count grid + the per-group means table live INSIDE the
    # Population overview slice (pooled distributions are demoted to that table).
    pop_slice = html[i_pop:i_pop + 60000]
    assert 'class="stats-grid"' in pop_slice and "Deposits / img" in pop_slice


def test_primary_metric_is_figure_1(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir, primary="rod_fraction")
    i_f1, i_f2 = html.find("Figure 1"), html.find("Figure 2")
    assert -1 < i_f1 < i_f2
    assert "exploratory" in html[i_f2:i_f2 + 400].lower()
    # the primary metric's plot (alt="ROD Fraction") sits in the Figure-1 slice, not Figure-2
    assert 'alt="ROD Fraction"' in html[i_f1:i_f2]
    assert 'alt="ROD Fraction"' not in html[i_f2:]


def test_spatial_exploratory_and_film_table_collapsed(synth_dir, tmp_path):
    html = _report(tmp_path, synth_dir)
    assert "<details" in html.lower()                        # per-image ledger collapsible
    assert "<details open" not in html.lower()               # collapsed by default (not open)
    if "Spatial Analysis" in html:
        i = html.find("Spatial Analysis")
        assert "exploratory" in html[max(0, i - 40):i + 260].lower()
