import scat.tools as tools


def test_all_tools_registered():
    names = {e.name for e in tools.iter_tools()}
    assert {"scan_folder", "infer_groups", "analyze_folder", "run_statistics", "generate_report"} <= names


def test_tool_specs_have_schemas():
    specs = {s["name"]: s for s in tools.tools_for_anthropic()}
    assert specs["analyze_folder"]["input_schema"]["properties"]["path"]["type"] == "string"
