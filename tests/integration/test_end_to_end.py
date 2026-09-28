# CMN-C2-715 — Integration: pre → inner workflow (linear) → post, and the real outer invoke path

import json

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import Graph
from src.nodes.clause_mapping_node import ClauseMappingNode
from src.nodes.governance_eval_node import GovernanceEvalNode
from src.nodes.hybrid_retrieve_node import HybridRetrieveNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode

_SUCCESS = AgentStatus.SUCCESS.value

_REQUEST = {
    "query": ("We are fine-tuning our customer-service LLM with RLHF on historical support tickets. "
              "Under APPI and the EU AI Act Article 10, what consent, pseudonymization and data quality "
              "requirements and provenance documentation apply?"),
    "training_technique": "RLHF",
    "jurisdictions": ["JP", "EU", "US"],
    "datasets": [
        {"dataset_id": "support-tickets-2026", "source": "data_catalog:support",
         "data_types": ["support_tickets", "personal_data"]},
    ],
}


def _run(user_input: str) -> dict:
    state: dict = {"user_input": user_input, "input_context": {"channel": "governance_console"},
                   "node_history": [], "error_log": []}
    state.update(PreProcessNode().execute(state) or {})
    for node in (HybridRetrieveNode(), ClauseMappingNode(), GovernanceEvalNode()):
        state.update(node.execute(state) or {})
    state.update(PostProcessNode().execute(state) or {})
    return state


class TestEndToEnd:
    def test_grounded_checklist_with_citations(self):
        state = _run(json.dumps(_REQUEST, ensure_ascii=False))
        assert state["status"] == _SUCCESS
        assert state["audit_logged"] is True
        env = json.loads(state["formatted_output"])
        assert env["status_kind"] == "governance_advisory"
        assert env["compliance_checklist"] and env["citations"]
        assert env["laws_covered"]
        assert env["dataset_audit_trail"][0]["provenance_status"] == "documented"
        assert "DRAFT" in env["disclaimer"] or "参考" in env["disclaimer"]

    def test_multiple_laws_covered(self):
        env = json.loads(_run(json.dumps(_REQUEST, ensure_ascii=False))["formatted_output"])
        laws = set(env["laws_covered"])
        assert "APPI" in laws
        assert any(law.startswith("EU") for law in laws)

    def test_out_of_scope_safe(self):
        env = json.loads(_run("recommend a good ramen shop in Osaka")["formatted_output"])
        assert env["status_kind"] == "out_of_scope"
        assert env["citations"] == []
        assert "DRAFT" in env["disclaimer"] or "参考" in env["disclaimer"]

    def test_empty_degrades_but_audits(self):
        state = _run("   ")
        assert state["status"] == _SUCCESS
        assert state["audit_logged"] is True
        assert json.loads(state["formatted_output"])["status_kind"] == "out_of_scope"

    def test_dataset_name_never_in_output(self):
        req = dict(_REQUEST)
        req["datasets"] = [{"dataset_id": "山田太郎の顧客リスト", "source": "mlflow:run-9"}]
        state = _run(json.dumps(req, ensure_ascii=False))
        assert "山田太郎" not in state["formatted_output"]
        assert "山田太郎" not in state["validated_input"]

    def test_real_invoke_end_to_end(self):
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
        out = Graph().invoke(json.dumps(_REQUEST, ensure_ascii=False), ctx=ctx)
        assert out["status"] == _SUCCESS
        assert "PostProcessNode" in out["node_history"]
        env = json.loads(out["output"])
        assert env["status_kind"] == "governance_advisory"
        assert env["compliance_checklist"] and env["citations"]

    def test_forged_dataset_id_surrogate_rehashed(self):
        # ★ F-02: a caller value SHAPED like an internal surrogate (ds:deadbeef) is re-hashed at S-1 (no
        # syntactic passthrough), so it can never forge an internal join key / impersonate another dataset.
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
        req = dict(_REQUEST)
        req["datasets"] = [{"dataset_id": "ds:deadbeef", "source": "data_catalog:support",
                            "data_types": ["support_tickets"]}]
        out = Graph().invoke(json.dumps(req, ensure_ascii=False), ctx=ctx)
        env = json.loads(out["output"])
        assert env["status_kind"] == "governance_advisory"
        tok = env["dataset_audit_trail"][0]["dataset_id"]
        assert tok.startswith("ds:") and tok != "ds:deadbeef"   # re-hashed, not passthrough
        assert "ds:deadbeef" not in out["output"]
