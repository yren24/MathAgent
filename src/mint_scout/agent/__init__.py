"""LangGraph orchestration for verified mint-agent pipeline tools.

Public objects are loaded lazily so importing one agent submodule does not initialize
the complete orchestration stack.  In particular, the deterministic pipeline and LLM
intake modules intentionally reference each other only at execution time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mint_scout.agent.graph import AgentGraphContext, build_agent_graph
    from mint_scout.agent.llm_explanation import (
        LLMExplanationContext,
        build_llm_explanation_graph,
    )
    from mint_scout.agent.llm_intake import LLMIntakeContext, build_llm_intake_graph
    from mint_scout.agent.llm_scientific import LLMScientificContext

__all__ = [
    "AgentGraphContext",
    "LLMExplanationContext",
    "LLMIntakeContext",
    "LLMScientificContext",
    "build_agent_graph",
    "build_llm_explanation_graph",
    "build_llm_intake_graph",
]


def __getattr__(name: str) -> Any:
    if name in {"AgentGraphContext", "build_agent_graph"}:
        from mint_scout.agent.graph import AgentGraphContext, build_agent_graph

        exports = {
            "AgentGraphContext": AgentGraphContext,
            "build_agent_graph": build_agent_graph,
        }
    elif name in {"LLMExplanationContext", "build_llm_explanation_graph"}:
        from mint_scout.agent.llm_explanation import (
            LLMExplanationContext,
            build_llm_explanation_graph,
        )

        exports = {
            "LLMExplanationContext": LLMExplanationContext,
            "build_llm_explanation_graph": build_llm_explanation_graph,
        }
    elif name in {"LLMIntakeContext", "build_llm_intake_graph"}:
        from mint_scout.agent.llm_intake import (
            LLMIntakeContext,
            build_llm_intake_graph,
        )

        exports = {
            "LLMIntakeContext": LLMIntakeContext,
            "build_llm_intake_graph": build_llm_intake_graph,
        }
    elif name in {"LLMScientificContext"}:
        from mint_scout.agent.llm_scientific import LLMScientificContext

        exports = {"LLMScientificContext": LLMScientificContext}
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals().update(exports)
    return exports[name]
