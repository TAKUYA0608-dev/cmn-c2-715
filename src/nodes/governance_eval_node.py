"""CMN-C2-715 — inner workflow step 3: governance_eval (GovernanceEval).

Composes the advisory deliverable — a compliance checklist (one obligation per retrieved clause: requirement,
recommended action, audit step, cited to law+article) + a per-dataset audit trail (opaque `ds:<sha8>` +
provenance status). On the 0-hit / rejected branch it emits the out-of-scope safe answer — no fabricated
governance advice.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.services.service import FineTuningGovernanceService
from src.utils.audit import emit_trace_event

_OUT_OF_SCOPE = (
    "ご質問に該当するデータガバナンス条項が公開法令 KB（APPI 2026 / EU AI Act Art.10 / Japan AI Act / "
    "NIST AI RMF）に見つかりませんでした。ファインチューニングの学習データに関する適用法令（個人情報の同意 / "
    "仮名・匿名加工 / データ品質 / provenance 文書化 など）を具体化のうえ再度お問い合わせください。"
)


class GovernanceEvalNode(FunctionNode):
    """Compose the compliance checklist + dataset audit trail with citations (or safe answer on 0-hit)."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        clauses = json.loads(state.get("retrieved_clauses") or "[]")
        if state.get("error_code") or not clauses:
            emit_trace_event("governance_eval.safe", {"reason": state.get("error_code") or "no_clause"}, state)
            report = {
                "status_kind": "out_of_scope",
                "message": _OUT_OF_SCOPE,
                "compliance_checklist": [],
                "dataset_audit_trail": [],
                "citations": [],
            }
            return {"result": json.dumps(report, ensure_ascii=False), "status": AgentStatus.SUCCESS.value}

        slots = json.loads(state.get("validated_input") or "{}")
        mapped = json.loads(state.get("mapped_obligations") or "[]")
        checklist = FineTuningGovernanceService.build_checklist(clauses, mapped)
        audit_trail = FineTuningGovernanceService.build_dataset_audit_trail(slots.get("datasets") or [])
        report = FineTuningGovernanceService.synthesize(
            slots.get("query_domain") or [], slots.get("training_technique"), checklist, audit_trail
        )
        emit_trace_event(
            "governance_eval.complete",
            {
                "obligation_count": len(checklist),
                "citation_count": len(report["citations"]),
                "dataset_count": len(audit_trail),
                "unverified_datasets": sum(1 for a in audit_trail if a["citation"] is None),
            },
            state,
        )
        return {"result": json.dumps(report, ensure_ascii=False), "status": AgentStatus.SUCCESS.value}
