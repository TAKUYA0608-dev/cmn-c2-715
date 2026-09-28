"""CMN-C2-715 — inner workflow step 2: clause_mapping (ClauseMapping).

Deterministic mapping of each retrieved governance clause to the caller context (jurisdictions / training
technique / data types) → a per-clause applicability annotation. Skips (no-op) on rejected / 0-hit input.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.services.service import FineTuningGovernanceService
from src.utils.audit import emit_trace_event


class ClauseMappingNode(FunctionNode):
    """Annotate each retrieved clause with its applicability to the caller context."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("error_code") or state.get("retrieval_hit_count", 0) == 0:
            emit_trace_event("clause_mapping.skip", {"reason": state.get("error_code") or "no_clause"}, state)
            return {}
        clauses = json.loads(state.get("retrieved_clauses") or "[]")
        slots = json.loads(state.get("validated_input") or "{}")
        technique = slots.get("training_technique")
        domains = slots.get("query_domain") or []
        mapped = [FineTuningGovernanceService.map_clause(c, technique, domains) for c in clauses]
        emit_trace_event(
            "clause_mapping.complete",
            {"count": len(mapped), "applicable": sum(1 for m in mapped if m["applies"])},
            state,
        )
        return {"mapped_obligations": json.dumps(mapped, ensure_ascii=False), "status": AgentStatus.SUCCESS.value}
