from __future__ import annotations

import sys
from types import ModuleType

from mint_scout.cli import main


def test_cli_routes_lifecycle_subcommand(monkeypatch):
    module = ModuleType("mint_scout.run_agent_lifecycle")
    received: list[str] = []

    def fake_main(args: list[str]) -> int:
        received.extend(args)
        return 17

    module.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mint_scout.run_agent_lifecycle", module)

    assert main(["lifecycle", "status", "--state", "/tmp/run.json"]) == 17
    assert received == ["status", "--state", "/tmp/run.json"]
