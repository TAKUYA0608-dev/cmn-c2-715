"""CMN-C2-715 — inner workflow step 1: hybrid_retrieve (HybridRetrieve).

Deterministic BM25-lite retrieval over the public-law governance KB (APPI 2026 / EU AI Act Art.10 / Japan AI
Act Ch.III-IV / NIST AI RMF). Sets `retrieval_hit_count`; **0 hits (rejected input or out-of-domain query)
routes to the out-of-scope safe answer** — the agent never fabricates governance advice that is not grounded
in a cited public-law clause.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.services.service import FineTuningGovernanceService
from src.utils.audit import emit_trace_event


class HybridRetrieveNode(FunctionNode):
    """Retrieve grounded governance clauses for the query."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        slots = json.loads(state.get("validated_input") or state.get("user_input") or "{}")
        canonical = json.dumps(slots, ensure_ascii=False)
        query = slots.get("query") or ""
        domains = slots.get("query_domain") or []

        if state.get("error_code") or not query.strip():
            emit_trace_event("hybrid_retrieve.skip", {"reason": state.get("error_code") or "empty_query"}, state)
            return {
                "validated_input": canonical,
                "retrieved_clauses": "[]",
                "retrieval_hit_count": 0,
                "error_code": state.get("error_code") or "NO_CLAUSE",
                "status": AgentStatus.SUCCESS.value,
            }

        clauses = FineTuningGovernanceService.retrieve(query, domains)
        emit_trace_event("hybrid_retrieve.complete", {"hit_count": len(clauses), "query_domains": domains}, state)
        out = {
            "validated_input": canonical,
            "retrieved_clauses": json.dumps(clauses, ensure_ascii=False),
            "retrieval_hit_count": len(clauses),
            "status": AgentStatus.SUCCESS.value,
        }
        if not clauses:
            out["error_code"] = "NO_CLAUSE"
        return out
