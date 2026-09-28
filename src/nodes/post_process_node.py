"""CMN-C2-715 — post_process node: ResponseValidate (S-3 output gate + S-4 audit).

S-3 (fail-closed): **enforce** citation completeness — a grounded governance advisory that has no statutory
citation, or that references a supplied dataset whose provenance did not resolve to an authorized system of
record, is never presented; it degrades to a safe `needs_review` answer with the deliverable body withheld
(`error_code=CITATION_INCOMPLETE`, still SUCCESS so post/S-4/disclaimer run). Re-redact any credential /
My-Number / email / phone leakage (defense-in-depth), and append the mandatory DRAFT advisory disclaimer —
the checklist is a decision aid, not a legal determination; the final compliance judgement and any filing are
an authorized legal / DPO reviewer's. S-4: emit an audit event (query domain / counts / verdict / error_code
only — never the raw query, a dataset name, or provenance). Runs on the full deliverable, the citation-blocked
branch, and the out-of-scope safe branch.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar, cast

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.services.service import QUERY_TERMS_MASKED, statutory_citation
from src.utils.audit import emit_trace_event

_DISCLAIMER = (
    "本チェックリストは、提供された問い合わせに該当する公開法令（APPI 2026 / EU AI Act Art.10 / Japan AI Act / "
    "NIST AI RMF）に基づく参考用の DRAFT データガバナンス助言であり、特定のファインチューニングの適法性・"
    "実施可否を確定するものではありません。最終的な適法性判断・データ利用の可否・文書化義務の充足は、必ず"
    "認可された法務 / DPO の確認を経てください。本エージェントは助言のみを行い、学習データ自体の処理・加工・"
    "申請の実行は行いません（APPI-safe: 個人情報を処理しません）。"
)

_CITATION_INCOMPLETE_MSG = (
    "助言の一部に検証可能な出典（公開法令の条項 / データセットの provenance）が確認できなかったため、"
    "根拠不十分な助言草案の提示を差し控えました。各データセットに認可されたデータガバナンス登録先の"
    "provenance を付与のうえ再実行してください。"
)
# Platform masking (S-2): the explanation for the `limitations` code QUERY_TERMS_MASKED. `{fields}` lists the
# request field names that reached the agent masked (never their values).
_MASKED_NOTE = (
    "注: 問い合わせの一部（{fields}）はプラットフォームの個人情報保護処理により [MASKED] に置き換えられてから"
    "検索されました（連続する大文字始まりの語、例: 'EU AI Act Article 10' の 'Act Article' を人名として扱うため）。"
    "伏せられた語でしか一致しない条項は取得できないため、このチェックリストは不完全な可能性があります。"
    "大文字始まりの語を連続させない表記（例: 'EU AI Act, article 10'）で再度お問い合わせください。"
)
_NOT_EVALUATED_MSG = (
    "問い合わせの一部（{fields}）がプラットフォームの個人情報保護処理により [MASKED] に置き換えられたため、"
    "該当するデータガバナンス条項を照合できませんでした。公開法令 KB の対象外と判定したものではありません。"
    "大文字始まりの語を連続させない表記（例: 'EU AI Act, article 10'）で再度お問い合わせください。"
    "ファインチューニングの学習データガバナンス以外のご質問は、本エージェントの対象外です。"
)

_NEEDS_REVIEW_NOTE = (
    "Grounding could not be verified for every obligation / dataset; the draft advisory is withheld pending "
    "valid provenance and authorized legal / DPO review."
)

# S-3 defense-in-depth: re-redact secrets / contact info that could leak into any free-text field of the
# deliverable (applied to the whole serialized report before it becomes the output envelope).
_SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{12,}|eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}|\d{12})\b")
_EMAIL = re.compile(r"[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,4}")
_PHONE = re.compile(r"(?<![\d.])(?:(?:\+81[-\s]?\d{1,4}|0\d{1,4})[-\s]?\d{1,4}[-\s]?\d{3,4})(?![\d.])")
_REDACTORS = (_SECRET, _EMAIL, _PHONE)


def _redact_report(report: dict[str, Any]) -> dict[str, Any]:
    """Serialize → redact secret / contact patterns → deserialize (whole-report defense-in-depth)."""
    text = json.dumps(report, ensure_ascii=False)
    for pattern in _REDACTORS:
        text = pattern.sub("[REDACTED]", text)
    return cast(dict[str, Any], json.loads(text))


class PostProcessNode(FunctionNode):
    """Verify citation completeness, redact leakage, append the DRAFT advisory disclaimer, emit audit."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _extra_security_gate_output(self, result: dict[str, Any]) -> dict[str, Any]:
        """S-3 preservation check: the DRAFT advisory disclaimer must be present in the output envelope.

        SDK 1.0.0 contract: receives the **result dict from `execute()`**; returns the (possibly filtered)
        result. MAY raise to block an output missing the mandatory disclaimer.
        """
        out = result.get("formatted_output", "")
        if out and "参考" not in out and "DRAFT" not in out:
            raise ValueError("S-3: DRAFT advisory disclaimer missing from output")
        return dict(result)

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        report: dict[str, Any] = _redact_report(json.loads(state.get("result", "{}") or "{}"))
        # Platform masking (S-2): request fields that reached retrieval as [MASKED]. A masked request is never
        # presented as complete, and "nothing matched" on a masked request is "not evaluated", not out of scope.
        masked_fields: list[str] = json.loads(state.get("masked_request_fields") or "[]")
        limitations = [QUERY_TERMS_MASKED] if masked_fields else []
        if masked_fields:
            fields = ", ".join(masked_fields)
            if report.get("status_kind") == "out_of_scope":
                report["status_kind"] = "not_evaluated"
                report["message"] = _NOT_EVALUATED_MSG.format(fields=fields)
            else:
                report["message"] = f"{report.get('message') or ''} {_MASKED_NOTE.format(fields=fields)}".strip()

        grounded = report.get("status_kind") == "governance_advisory"
        citations = report.get("citations", [])
        checklist = report.get("compliance_checklist", [])
        audit_trail = report.get("dataset_audit_trail", [])
        # S-3 per-obligation authoritative correspondence (fail-closed): a grounded advisory is complete only
        # when EVERY delivered checklist obligation (a) carries its OWN authentic KB statutory citation
        # (stat:sha8(obligation_id|citation_label) — a substituted citation copied from another clause fails)
        # AND (b) that citation has a corresponding member in the top-level `citations` list (a missing
        # top-level citation, or one belonging to a different obligation, fails). Every supplied dataset must
        # likewise carry a resolved provenance citation (a forged / unverifiable source resolved to None
        # upstream → item citation None → incomplete). Not merely a non-empty citation list. Vacuously
        # complete when not grounded.
        cited = {c.get("citation") for c in citations if c.get("citation")}
        citation_complete = (not grounded) or (
            bool(checklist)
            and all(
                item.get("citation")
                == statutory_citation(str(item.get("obligation_id", "")), str(item.get("citation_label", "")))
                and item.get("citation") in cited
                for item in checklist
            )
            and all(a.get("citation") for a in audit_trail)
        )

        if grounded and not citation_complete:
            error_code = state.get("error_code") or "CITATION_INCOMPLETE"
            blocked = {
                "status_kind": "needs_review",
                "query_domains": report.get("query_domains", []),
                "compliance_checklist": [],  # incomplete deliverable body withheld
                "dataset_audit_trail": [],
                "human_review": {"required": True, "status": "pending_legal_review", "note": _NEEDS_REVIEW_NOTE},
                "citations": [],
                "citation_complete": False,
                "limitations": limitations,
                "message": (
                    f"{_CITATION_INCOMPLETE_MSG} {_MASKED_NOTE.format(fields=', '.join(masked_fields))}"
                    if masked_fields
                    else _CITATION_INCOMPLETE_MSG
                ),
                "disclaimer": _DISCLAIMER,
            }
            emit_trace_event(
                "response_validate.citation_blocked",
                {
                    "obligation_count": len(checklist),
                    "dataset_count": len(audit_trail),
                    "limitations": limitations,
                    "error_code": error_code,
                },
                state,
            )
            return {
                "formatted_output": json.dumps(blocked, ensure_ascii=False),
                "disclaimer": _DISCLAIMER,
                "audit_logged": True,
                "error_code": error_code,
                "status": AgentStatus.SUCCESS.value,
            }

        formatted = {
            "status_kind": report.get("status_kind"),
            "query_domains": report.get("query_domains", []),
            "laws_covered": report.get("laws_covered", []),
            "training_technique": report.get("training_technique"),
            "compliance_checklist": checklist,
            "dataset_audit_trail": audit_trail,
            "citations": citations,
            "citation_complete": citation_complete,
            "limitations": limitations,
            "confidence": "low" if masked_fields else report.get("confidence"),
            "message": report.get("message"),
            "disclaimer": _DISCLAIMER,
        }
        emit_trace_event(
            "response_validate.complete",
            {
                "status_kind": report.get("status_kind"),
                "obligation_count": len(checklist),
                "dataset_count": len(audit_trail),
                "citation_count": len(citations),
                "citation_complete": citation_complete,
                "limitations": limitations,
                "error_code": state.get("error_code"),
            },
            state,
        )
        return {
            "formatted_output": json.dumps(formatted, ensure_ascii=False),
            "disclaimer": _DISCLAIMER,
            "audit_logged": True,
            "status": AgentStatus.SUCCESS.value,
        }
