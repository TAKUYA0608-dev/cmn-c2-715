"""CMN-C2-715 — pre_process node: QueryNormalize (S-1/S-2 validation + field-level input hygiene).

Accepts a structured JSON governance request (`{query, datasets[], training_technique, jurisdictions}`) or NL
text, normalizes it (NFKC), enforces S-1/S-2, classifies the legal domain, and extracts the advisory slots.
The agent answers governance *questions* only — it never ingests the fine-tuning data itself.

Degraded contract (SDK 1.0.0): injection markers / oversize / empty never set `status=ERROR`. They return
`status=SUCCESS + error_code` (`INJECTION_REJECTED` / `INPUT_TOO_LONG` / `INPUT_REJECTED`) and **discard the
offending body** so `main`/`post_process` still run (disclaimer + S-3 + S-4). The `@final` framework hook is
not invoked by the local stub framework, so `execute()` re-checks the same S-2 conditions itself.

Field-level input hygiene: every free-text value written into `validated_input` (the query text, dataset
descriptors, jurisdictions) is passed through `_hygiene()` (redacts credential / My-Number / email / phone
patterns) so a secret a caller inadvertently pastes never persists in State. Each dataset's `dataset_id` is
UNCONDITIONALLY tokenized to an opaque `ds:<sha8>` surrogate (a name can never leak), and its `source` is
constrained to a **safe opaque provenance reference** (authorized-registry allowlist); anything else is
dropped to `None` so untrusted / forged text can never reach an output citation.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.services.service import (
    FineTuningGovernanceService,
    masked_request_fields,
    resolve_provenance,
    safe_identifier,
)
from src.utils.audit import emit_trace_event

_MAX_INPUT = 20_000  # a governance question is a short prompt (+ a small dataset descriptor list)
_INJECTION_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "disregard the above",
    "system prompt",
    "you are now",
    "###system",
    "<|im_start|>",
)
_REJECT_CODES = frozenset({"INJECTION_REJECTED", "INPUT_TOO_LONG"})
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# ── input hygiene: redact secrets a caller may inadvertently include before persisting to State ──
_CREDENTIAL = re.compile(r"\b(sk-[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{12,}|eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,})\b")
_MY_NUMBER = re.compile(r"\b\d{12}\b")  # Japanese My-Number / 個人番号
_EMAIL = re.compile(r"\b[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,4}\b")
_PHONE = re.compile(r"(?<![\d.])(?:(?:\+81[-\s]?\d{1,4}|0\d{1,4})[-\s]?\d{1,4}[-\s]?\d{3,4})(?![\d.])")
_REDACTED = "[REDACTED]"


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def _hygiene(text: str) -> str:
    """Redact credential / My-Number / email / phone patterns from a free-text value."""
    out = _CREDENTIAL.sub(_REDACTED, text)
    out = _MY_NUMBER.sub(_REDACTED, out)
    out = _EMAIL.sub(_REDACTED, out)
    out = _PHONE.sub(_REDACTED, out)
    return out


class PreProcessNode(FunctionNode):
    """Validate the governance request, hygiene it, classify the legal domain, and extract its slots."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _reject_code(self, raw: str) -> str | None:
        if len(raw) > _MAX_INPUT:
            return "INPUT_TOO_LONG"
        if any(marker in _nfkc(raw).lower() for marker in _INJECTION_MARKERS):
            return "INJECTION_REJECTED"
        return None

    def _extra_security_gate_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """S-2 domain checks: size cap + prompt-injection markers.

        SDK 1.0.0 contract: MUST NOT raise, and MUST NOT set status=ERROR (that would short-circuit the
        pipeline past post_process). A rejection is surfaced as a degraded `SUCCESS + error_code`; the
        offending body is discarded by execute().
        """
        raw = state.get("user_input", "") or ""
        code = self._reject_code(raw)
        if code:
            out = dict(state)
            out["error_code"] = code
            return out
        return dict(state)

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        raw = state.get("user_input", "") or ""
        input_context = state.get("input_context", {})  # read-only [C1]
        enriched = json.dumps(
            {"source": "LLMFineTuningGovernanceQaAgent", "channel": input_context.get("channel", "unknown")},
            ensure_ascii=False,
        )

        # Degrade on rejection: the S-2 hook may already have set error_code (real SDK); re-detect here
        # because the local stub framework does not invoke the hook. Discard the offending body entirely.
        prior = state.get("error_code")
        code = prior if prior in _REJECT_CODES else self._reject_code(raw)
        if code:
            emit_trace_event("query_normalize.rejected", {"reason": code}, state)
            return {
                "validated_input": "{}",
                "input_format": "rejected",
                "enriched_context": enriched,
                "user_input": "",
                "error_code": code,
                "status": AgentStatus.SUCCESS.value,
            }

        if not raw.strip():
            emit_trace_event("query_normalize.rejected", {"reason": "empty_input"}, state)
            return {
                "validated_input": "{}",
                "input_format": "empty",
                "enriched_context": enriched,
                "error_code": "INPUT_REJECTED",
                "status": AgentStatus.SUCCESS.value,
            }

        slots, fmt, masked_fields = self._parse(_CONTROL.sub("", _nfkc(raw)))
        emit_trace_event(
            "query_normalize.validated",
            {
                "input_format": fmt,
                "query_domains": slots["query_domain"],
                "dataset_count": len(slots["datasets"]),
                "masked_fields": masked_fields,
            },
            state,
        )
        return {
            "validated_input": json.dumps(slots, ensure_ascii=False),
            "input_format": fmt,
            "enriched_context": enriched,
            # Request fields the platform's S-2 pass masked before this node ran (field names only);
            # post_process reports them instead of presenting the answer as complete.
            "masked_request_fields": json.dumps(masked_fields),
            "status": AgentStatus.SUCCESS.value,
        }

    def _hygiene_dataset(self, raw: Any) -> dict[str, Any] | None:
        """Reduce a caller dataset descriptor to an opaque, hygiened record (id tokenized, source resolved)."""
        if not isinstance(raw, dict):
            return None
        raw_id = str(raw.get("dataset_id") or raw.get("id") or "").strip()
        if not raw_id:
            return None
        data_types = [_hygiene(str(t)) for t in raw.get("data_types", []) if isinstance(t, (str, int))]
        return {
            # Privacy: the dataset_id primary key is always tokenized (idempotent join key).
            "dataset_id": safe_identifier(raw_id),
            # Provenance: a grounded citation only if `source` names an authorized system of record
            # (privacy-tokenized); unverifiable / forged text → None → S-3 blocks (needs_review).
            "source": resolve_provenance(raw.get("source")),
            "data_types": data_types,
        }

    def _parse(self, text: str) -> tuple[dict[str, Any], str, list[str]]:
        query = ""
        datasets: list[dict[str, Any]] = []
        technique = None
        raw_technique: Any = None
        jurisdictions: list[str] = []
        fmt = "text"
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            obj = None
        if isinstance(obj, dict):
            fmt = "json"
            query = _hygiene(str(obj.get("query") or obj.get("question") or ""))
            for d in obj.get("datasets") or []:
                hd = self._hygiene_dataset(d)
                if hd is not None:
                    datasets.append(hd)
            raw_technique = obj.get("training_technique")
            technique = FineTuningGovernanceService.normalize_technique(raw_technique)
            jurisdictions = [_hygiene(str(j)) for j in (obj.get("jurisdictions") or []) if isinstance(j, str)]
        else:
            query = _hygiene(text)

        domains = FineTuningGovernanceService.classify_domains(query, jurisdictions)
        return (
            {
                "query": query,
                "datasets": datasets,
                "training_technique": technique,
                "jurisdictions": jurisdictions,
                "query_domain": domains,
            },
            fmt,
            masked_request_fields(query, jurisdictions, raw_technique, technique),
        )
