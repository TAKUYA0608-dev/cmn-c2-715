"""CMN-C2-715 — inner domain workflow graph (Cat 2).

Instantiated by GovernanceAdvisoryWorkflowGraphNode.get_subgraph() in graph.py. Linear topology with
per-node skip guards (the portable Cat 2 form; conditional edges don't propagate across the subgraph
boundary):

    START → hybrid_retrieve → clause_mapping → governance_eval → END

On rejected / 0-hit input, hybrid_retrieve sets retrieval_hit_count=0 (+error_code); clause_mapping no-ops
and governance_eval emits the out-of-scope safe answer — no fabricated governance advice.
"""

from __future__ import annotations
from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState

from src.nodes.clause_mapping_node import ClauseMappingNode
from src.nodes.governance_eval_node import GovernanceEvalNode
from src.nodes.hybrid_retrieve_node import HybridRetrieveNode
from src.schemas.state import State


class GovernanceAdvisoryWorkflow(BaseGraph):
    """Inner graph: hybrid_retrieve → clause_mapping → governance_eval."""

    @property
    def name(self) -> str:
        return "GovernanceAdvisoryWorkflow"

    @property
    def state_schema(self) -> type:
        return State

    def _validate_config(self) -> None:
        pass

    def register_nodes(self) -> None:
        # No super() — BaseGraph.register_nodes() is abstract.
        self._nodes["hybrid_retrieve"] = HybridRetrieveNode()
        self._nodes["clause_mapping"] = ClauseMappingNode()
        self._nodes["governance_eval"] = GovernanceEvalNode()

    def add_edges(self) -> None:
        # Static linear backbone; the 0-hit / rejected skip is handled by per-node guards.
        self._sg.add_edge(START, "hybrid_retrieve")
        self._sg.add_edge("hybrid_retrieve", "clause_mapping")
        self._sg.add_edge("clause_mapping", "governance_eval")
        self._sg.add_edge("governance_eval", END)

    def route(self, state: AgentState) -> str:
        """Required by the BaseGraph ABC. Linear topology → not wired to a conditional edge."""
        if state.get("error_code") or state.get("retrieval_hit_count", 0) == 0:
            return "governance_eval"
        return "clause_mapping"

    def get_output(self, state: AgentState) -> dict[str, Any]:
        return {
            "output": state.get("result"),
            "status": state.get("status"),
            "retrieval_hit_count": state.get("retrieval_hit_count", 0),
            "error_code": state.get("error_code"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
