import subprocess
import sys


def test_core_imports_without_agent_extras():
    """Core SCAT must import with the [agent] extras (pydantic/anthropic/claude_agent_sdk)
    blocked — they belong only to scat.agent / scat.tools."""
    code = (
        "import sys\n"
        "class Block:\n"
        "  def find_spec(self, name, path=None, target=None):\n"
        "    if name.split('.')[0] in {'pydantic','anthropic','claude_agent_sdk'}:\n"
        "      raise ImportError('blocked: ' + name)\n"
        "    return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import scat, scat.analyzer, scat.pipeline, scat.grouping_util, scat.cli\n"
        "from scat.pipeline import scan_folder_service, analyze_folder_service\n"
        "print('ok')\n"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True, stderr=subprocess.STDOUT)
    assert out.strip().endswith("ok")


def test_pipeline_module_has_no_agent_imports():
    src = open("scat/pipeline.py").read()
    assert "import pydantic" not in src and "import anthropic" not in src
    assert "from scat.agent" not in src and "import scat.tools" not in src
