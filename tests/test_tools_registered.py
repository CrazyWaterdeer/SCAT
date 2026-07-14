import scat.tools as tools


def test_expected_tools_registered():
    names = {e.name for e in tools.iter_tools()}
    assert {"scan_folder", "analyze_folder", "run_statistics", "generate_report"} <= names
    # grouping is the LLM's job now — there is no deterministic infer_groups tool.
    assert "infer_groups" not in names


def test_tool_specs_have_schemas():
    specs = {s["name"]: s for s in tools.tools_for_anthropic()}
    assert specs["analyze_folder"]["input_schema"]["properties"]["path"]["type"] == "string"
