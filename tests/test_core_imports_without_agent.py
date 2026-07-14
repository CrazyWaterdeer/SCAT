import os
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
        "import scat.main_gui\n"  # the GUI (incl. chat dock) imports the agent stack lazily
        "from scat.pipeline import scan_folder_service, analyze_folder_service\n"
        # Constructing the chat dock itself must NOT need the extra; only sending a message does,
        # and that must degrade gracefully (build_runner -> scat.tools -> pydantic is blocked).
        "from PySide6.QtWidgets import QApplication\n"
        "_app = QApplication([])\n"
        "from scat.agent.chat_widget import ChatDockWidget\n"
        "_w = ChatDockWidget()\n"
        "assert _w._ensure_runner() is False, 'build_runner should fail gracefully without the extra'\n"
        "print('ok')\n"
    )
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    out = subprocess.check_output([sys.executable, "-c", code], text=True, stderr=subprocess.STDOUT, env=env)
    assert out.strip().endswith("ok")


def test_pipeline_module_has_no_agent_imports():
    src = open("scat/pipeline.py").read()
    assert "import pydantic" not in src and "import anthropic" not in src
    assert "from scat.agent" not in src and "import scat.tools" not in src


def test_gui_has_no_top_level_agent_imports():
    """The GUI may import the agent stack LAZILY (indented, inside a method — e.g. the chat dock
    builds its runner on first send), but never at module top level, so `import scat.main_gui`
    stays cheap and works without the [agent] extra. The runtime guard above proves the effect;
    this pins the mechanism."""
    banned = ("import scat.agent", "from scat.agent", "from .agent",
              "import scat.tools", "from scat.tools", "from .tools",
              "import pydantic", "import anthropic", "import claude_agent_sdk")
    for i, line in enumerate(open("scat/main_gui.py").read().splitlines(), 1):
        assert not line.startswith(banned), f"top-level agent import at main_gui.py:{i}: {line!r}"
