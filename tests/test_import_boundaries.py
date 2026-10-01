from __future__ import annotations

import subprocess
import sys


def test_user_intake_imports_in_fresh_interpreter():
    result = subprocess.run(
        [sys.executable, "-c", "import mint_scout.user_intake"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_agent_public_exports_are_available_lazily():
    from mint_scout import agent

    assert agent.AgentGraphContext.__name__ == "AgentGraphContext"
    assert agent.LLMExplanationContext.__name__ == "LLMExplanationContext"
    assert agent.LLMIntakeContext.__name__ == "LLMIntakeContext"
