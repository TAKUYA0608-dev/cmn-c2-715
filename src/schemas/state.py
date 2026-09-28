"""CMN-C2-715 — Agent state (LLM Fine-Tuning Data Governance Q&A, Cat 2).

ADR-005: State is a flat TypedDict — never a validation/BaseModel instance. Complex fields are stored
as JSON strings (``NotRequired[str]`` + ``# JSON:``); nodes ``json.dumps`` on write / ``json.loads`` on read.

Data-governance / APPI posture: the agent answers governance *questions* against a public-law KB — it never
ingests or processes the fine-tuning training data itself. A caller-supplied dataset descriptor is reduced to
an opaque ``ds:<sha8>`` surrogate (``safe_identifier``) and its provenance is resolved to a grounded
``src:<sha8>`` citation only when it names an authorized data-governance system of record
(``resolve_provenance``) — otherwise ``None``, and S-3 blocks a per-dataset audit trail as fail-closed. Any
free-text field a caller supplies (query / dataset descriptor) is hygiened at S-1 so a credential / My-Number /
email / phone never persists in State.

All agent-specific fields are NotRequired (populated progressively; absent at empty-start invoke).
"""

from __future__ import annotations


from framework.schemas.agent_state import AgentState


class State(AgentState):
    """Agent state for the fine-tuning data-governance compliance-advisory workflow."""

    # ── pre_process (QueryNormalize, S-1 validated request + field-level hygiene) ──────────────
    validated_input: str  # JSON: {query, datasets[], training_technique, jurisdictions, query_domain}
    input_format: str  # "json" | "text" | "empty" | "rejected"
    enriched_context: str  # JSON: {source, channel} (read-only caller context)
    masked_request_fields: str  # JSON: [field name] masked by the platform's S-2 pass (names only)

    # ── inner workflow (hybrid_retrieve → clause_mapping → governance_eval) ──────────────────────
    retrieved_clauses: str  # JSON: [{clause_id, law, article, citation, requirement, ...}]
    retrieval_hit_count: int  # governance clauses retrieved (0 → out-of-scope safe answer)
    mapped_obligations: str  # JSON: [{clause_id, applicability, applies}]
    result: str  # JSON: assembled governance-advisory deliverable

    # ── post_process (S-3 gate + S-4 audit) ─────────────────────────────────────────────────────
    formatted_output: str  # JSON: final response envelope (deliverable + disclaimer)
    disclaimer: str  # mandatory "confirm with legal / DPO" DRAFT disclaimer
    audit_logged: bool  # True once the terminal audit event is emitted

    # ── degraded-path signalling (SUCCESS + error_code, never status=ERROR) ──────────────────────
    error_code: str  # INPUT_REJECTED | INJECTION_REJECTED | INPUT_TOO_LONG
    #                                       | NO_CLAUSE | CITATION_INCOMPLETE
    error_message: str  # operator-facing detail
