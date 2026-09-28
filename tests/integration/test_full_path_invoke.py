"""Production path with the platform's S-2 personal-data masking active.

The framework input gate (final, runs before every node of this template) replaces any run of two or more
Title-Case words in `user_input` with "[MASKED]" — it reads "Act Article" in "EU AI Act Article 10" or
"Personal Information Protection Commission" as a person's name. Measured 2026-09-24 on AgentCore 1.0.3:
"EU AI Act Article 10" reached retrieval as "EU AI [MASKED] 10" and the checklist silently dropped two of the
three Article 10 obligations while still reporting confidence "medium"; a fully masked in-domain question was
answered "out_of_scope". These tests call the real `Graph().invoke()` as a VERIFIED_EXTERNAL caller so the gate
runs exactly as in production.
"""

from __future__ import annotations

import json
from typing import Any

from framework.nodes.function_node import detect_pii, mask_pii
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import Graph

_MASKED_Q = (
    "Under APPI and the EU AI Act Article 10, what consent and data quality requirements apply "
    "when we fine-tune on customer support tickets containing personal data?"
)
_PLAIN_Q = (
    "Under APPI and the EU AI Act, article 10, what consent and data quality requirements apply "
    "when we fine-tune on customer support tickets containing personal data?"
)
_ART10 = {"euaiact-art10-governance", "euaiact-art10-quality", "euaiact-art10-special"}


def _env(text: str) -> dict[str, Any]:
    ctx = InvocationContext(caller_id="integration-test", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
    out = Graph().invoke(text, ctx=ctx)
    assert str(out.get("status")).lower().endswith("success"), out.get("status")
    body = out["output"]
    env: dict[str, Any] = json.loads(body) if isinstance(body, str) else body
    return env


def _platform_view(text: str) -> str:
    """What the platform's input gate turns `text` into before this template sees it."""
    return str(mask_pii(text, detect_pii(text)))


def _ids(env: dict[str, Any]) -> set[str]:
    return {str(c["obligation_id"]) for c in env.get("compliance_checklist") or []}


class TestPlatformMaskedQuery:
    def test_a_partly_masked_question_is_flagged_and_low_confidence(self) -> None:
        assert "EU AI [MASKED] 10" in _platform_view(_MASKED_Q), "precondition: the platform masks the span"
        env = _env(_MASKED_Q)
        assert not _ART10 <= _ids(env), "precondition: the masked words were missing from retrieval"
        assert env["status_kind"] == "governance_advisory", env
        assert env["confidence"] == "low", env
        assert env["limitations"] == ["QUERY_TERMS_MASKED"], env
        assert "[MASKED]" in env["message"] and "query" in env["message"], env["message"]

    def test_a_fully_masked_in_domain_question_is_not_evaluated_rather_than_out_of_scope(self) -> None:
        question = "Supervised Fine Tuning Consent Requirements"
        assert _platform_view(question) == "[MASKED]", "precondition: the whole question is masked"
        assert _env(question.lower())["status_kind"] == "governance_advisory", "precondition: in-domain words"
        env = _env(question)
        assert env["status_kind"] == "not_evaluated", env
        assert env["compliance_checklist"] == [] and env["citations"] == [], env
        assert env["limitations"] == ["QUERY_TERMS_MASKED"], env
        assert env["confidence"] == "low", env
        assert "対象外と判定したものではありません" in env["message"], env["message"]

    def test_an_unmasked_non_matching_term_plus_a_masked_term_is_not_a_confident_out_of_scope(self) -> None:
        # The unmasked term ("quantum annealing") reaches retrieval and matches nothing; the second term is
        # the one that would have matched (the commission that supervises APPI), but it was masked. That is
        # not evidence the question is outside the KB.
        question = "Does the Personal Information Protection Commission regulate quantum annealing?"
        assert _platform_view(question) == "Does the [MASKED] regulate quantum annealing?", "precondition"
        assert (
            _env("Does it regulate quantum annealing?")["status_kind"] == "out_of_scope"
        ), "precondition: the unmasked term alone matches nothing"
        env = _env(question)
        assert env["status_kind"] == "not_evaluated", env
        assert env["limitations"] == ["QUERY_TERMS_MASKED"], env
        assert env["confidence"] == "low", env

    def test_masked_jurisdiction_and_technique_fields_are_reported(self) -> None:
        plain = {
            "query": "what consent is needed to fine-tune on support tickets?",
            "jurisdictions": ["us"],
            "training_technique": "supervised fine-tuning",
        }
        masked = dict(plain, jurisdictions=["United States"], training_technique="Supervised Fine-Tuning")
        control = _env(json.dumps(plain))
        env = _env(json.dumps(masked))
        assert control["training_technique"] == "sft" and control["query_domains"] == ["US"], control
        assert env["training_technique"] is None and env["query_domains"] == [], "precondition: both masked"
        assert env["confidence"] == "low", env
        assert env["limitations"] == ["QUERY_TERMS_MASKED"], env
        assert "jurisdictions, training_technique" in env["message"], env["message"]

    def test_a_masked_technique_that_still_normalises_is_not_flagged(self) -> None:
        request = {"query": "what consent is needed to fine-tune on support tickets?", "jurisdictions": ["us"]}
        request["training_technique"] = "Supervised Fine-Tuning (SFT)"
        assert "[MASKED]-Tuning (SFT)" in _platform_view(json.dumps(request)), "precondition: part of it is masked"
        env = _env(json.dumps(request))
        assert env["training_technique"] == "sft", env
        assert env["limitations"] == [], env
        assert env["confidence"] == "medium", env

    def test_an_unmasked_question_is_unchanged(self) -> None:
        env = _env(_PLAIN_Q)
        assert env["limitations"] == []
        assert env["confidence"] == "medium"
        assert _ART10 <= _ids(env)
        assert "[MASKED]" not in env["message"]

    def test_an_unmasked_off_topic_question_stays_out_of_scope(self) -> None:
        env = _env("recommend a good ramen shop in Osaka")
        assert env["status_kind"] == "out_of_scope"
        assert env["limitations"] == []
        assert env["confidence"] is None

    def test_an_off_topic_question_with_a_masked_name_is_not_evaluated(self) -> None:
        # Documented trade-off: the template cannot tell what the masked words were, so it does not claim
        # the question is outside the KB even when the rest of it is clearly off-topic.
        env = _env("What does Taro Yamada recommend for ramen in Osaka?")
        assert env["status_kind"] == "not_evaluated", env
        assert env["limitations"] == ["QUERY_TERMS_MASKED"], env
