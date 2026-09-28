"""CMN-C2-715 — deterministic domain services (no framework imports, no LLM).

FineTuningGovernanceService: classifies a fine-tuning data-governance question into legal domains, retrieves
the relevant public-law clauses from the seeded governance KB (APPI 2026 / EU AI Act Art.10 / Japan AI Act
Ch.III-IV / NIST AI RMF) with a deterministic BM25-lite scorer, maps each clause to the caller context
(jurisdictions / training technique / data types), and synthesizes an advisory deliverable — a compliance
checklist (each obligation cited to law+article) + a per-dataset audit trail.

Everything here is deterministic and auditable (tokenization + keyed KB composition) — there is **no LLM**.
The agent answers governance *questions* against public-law text only; it never ingests or processes the
fine-tuning training data itself. A caller dataset descriptor is keyed by an opaque ``ds:<sha8>`` surrogate
(the raw name is never carried into the deliverable), its provenance is resolved to a grounded ``src:<sha8>``
citation only when it names an authorized data-governance system of record, and the S-3 output gate re-redacts
any secret/PII that leaks. Seeded KB clauses are overridable by CoE via a change-controlled engineer MR (see
docs/07) without touching node logic.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# Two SEPARATE concerns — do not conflate them:
#   (1) PRIVACY (safe_identifier): every caller dataset identifier (dataset_id) is UNCONDITIONALLY tokenized to
#       a deterministic opaque surrogate so PII (even a bare name like ``Alice`` / ``John.Smith`` /
#       ``TaroYamada``, no spaces/symbols) can never reach a citation or the output. Tokenizing is a privacy
#       measure — it does NOT assert the value is authorized/verifiable.
#   (2) PROVENANCE (resolve_provenance): a caller ``source`` becomes a grounded CITATION only when it is
#       resolvable against the authorized data-governance provenance registry (names a trusted system of
#       record). Any other free text (a dataset name, ``unknown``, a fabricated value, or a value merely
#       SHAPED like a surrogate ``src:1a2b3c4d``) is NOT verifiable provenance → it yields NO citation → S-3
#       blocks the per-dataset audit trail as CITATION_INCOMPLETE (fail-closed). "Tokenized" is never
#       sufficient for a citation; the value must first pass provenance validation.
# Tokenization is UNCONDITIONAL (no syntactic passthrough): a caller value merely *shaped* like a surrogate
# (``ds:deadbeef``) is re-hashed, never trusted, so it can never forge an internal join key. Dataset
# identifiers are tokenized exactly once at S-1 (pre_process); any downstream re-tokenization is a
# deterministic label-only no-op (the surrogate is a per-invocation display key, never rejoined against the
# S-1 value).

# Authorized provenance registry: the data-governance systems of record an org trusts as verifiable dataset
# provenance sources. A caller ``source`` is accepted as a grounded citation ONLY when its leading namespace
# names one of these (the "trusted context"). This is the deploying org's / CoE's registry — overridable
# without touching node logic; it is a SEMANTIC allowlist of authorized systems, not a syntactic character
# class.
AUTHORIZED_PROVENANCE_SYSTEMS = frozenset(
    {
        "data_catalog",
        "datacatalog",
        "collibra",
        "alation",
        "amundsen",
        "datahub",
        "dataset_registry",
        "feature_store",
        "feast",
        "consent_ledger",
        "consent_registry",
        "consent_platform",
        "cmp",
        "dvc",
        "lakefs",
        "mlflow",
        "wandb",
        "model_registry",
        "data_warehouse",
        "dwh",
        "lakehouse",
        "snowflake",
        "bigquery",
        "lineage",
        "provenance_store",
        "openlineage",
        "data_lineage",
        "system_of_record",
        "sor",
        "authorized_feed",
        "data_governance_platform",
        "dgp",
    }
)


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def statutory_citation(clause_id: str, citation_label: str) -> str:
    """The KB-owned grounded statutory-citation surrogate for a clause (``stat:<sha8>``), never
    caller-supplied. This is the single authoritative formula, shared by retrieval (to mint each clause's
    citation) and the S-3 output gate (to re-derive and authenticate every delivered obligation's citation),
    so a substituted / fabricated obligation citation can never pass grounding.
    """
    return "stat:" + _sha8(f"{clause_id}|{citation_label}")


def safe_identifier(value: Any) -> str:
    """PRIVACY tokenize a caller dataset identifier to a deterministic opaque surrogate ``ds:<sha8>``.

    Caller identifiers are **always** tokenized — no syntactic passthrough — so a name (with or without
    spaces) can never survive into a citation or the output, and a caller value merely *shaped* like a
    surrogate (``ds:deadbeef``) is re-hashed rather than trusted (it can never forge an internal join key).
    Same input → same surrogate (checklist / audit trail stay joinable within one invocation). This is a
    privacy measure only; it makes no claim that the identifier is authorized.
    """
    return "ds:" + _sha8(str(value or "").strip())


def resolve_provenance(value: Any) -> str | None:
    """Resolve a **raw** caller ``source`` to a grounded, privacy-tokenized CITATION — or ``None``.

    Provenance validation (separate from privacy) and the **single** resolution point (S-1 / pre_process).
    A citation is emitted **only** when the source names an authorized data-governance system of record
    (``<authorized-namespace>[:<ref>]``). Any other value — a dataset name, ``unknown``, a fabricated value,
    **or a value that merely looks like a surrogate (``src:1a2b3c4d``)** — is not verifiable provenance and
    returns ``None`` so the S-3 gate blocks the per-dataset audit trail as CITATION_INCOMPLETE (fail-closed).
    When authorized, the raw label is never used verbatim: the citation is a privacy hash (``src:<sha8>``) of
    the authorized reference. No synthetic provenance is fabricated.

    ★ Forged-surrogate defence: there is **no format-based passthrough**. A caller-supplied ``src:<hex>``
    has namespace ``src``, which is not an authorized system of record, so it resolves to ``None`` — it is
    dropped here at S-1 and can never reach a citation. Because provenance is resolved exactly once (here),
    the produced ``src:<sha8>`` is the trusted citation downstream and is **never** fed back through this
    function (which would, correctly, reject it), so no forged value can imitate an internal surrogate.
    """
    text = str(value or "").strip()
    if not text:
        return None
    namespace = text.split(":", 1)[0].strip().lower()
    if namespace not in AUTHORIZED_PROVENANCE_SYSTEMS:
        return None  # unverifiable / forged-surrogate provenance → fail-closed (no citation → needs_review)
    return "src:" + _sha8(text)


# ── seeded governance KB (public-law clauses; APPI 2026 / EU AI Act Art.10 / Japan AI Act / NIST AI RMF) ──
# Each clause: {clause_id, law, article, citation_label, title, requirement, recommended_action, audit_step,
#               jurisdiction, keywords, applies_to}. `citation_label` is the human-readable law+article; the
# grounded surrogate is a KB-owned `stat:<sha8>` (never caller-supplied). `applies_to` narrows applicability
# by training technique / data type for clause_mapping.
GOVERNANCE_KB: list[dict[str, Any]] = [
    {
        "clause_id": "appi-consent",
        "law": "APPI",
        "article": "Art.17-18 / Art.27",
        "citation_label": "APPI (個人情報保護法) Art.17-18 purpose specification & Art.27 third-party provision",
        "title": "Consent & purpose-of-use for personal data used in fine-tuning",
        "requirement": "Personal data used to fine-tune a model must have a specified purpose of use that "
        "covers model training, and third-party provision / joint use requires a lawful basis.",
        "recommended_action": "Confirm the original purpose of use covers fine-tuning; obtain or map consent, "
        "and document the lawful basis for any customer / employee personal data.",
        "audit_step": "Record, per personal-data field, the purpose-of-use basis (consent / statutory) that "
        "authorizes its use in fine-tuning.",
        "jurisdiction": "JP",
        "keywords": [
            "consent",
            "personal data",
            "personal information",
            "purpose",
            "appi",
            "customer",
            "ticket",
            "tickets",
            "support",
            "employee",
            "third party",
            "同意",
            "個人情報",
            "利用目的",
        ],
        "applies_to": ["rlhf", "sft", "dpo", "instruction_tuning", "personal_data", "customer_data"],
    },
    {
        "clause_id": "appi-pseudonymization",
        "law": "APPI",
        "article": "Art.41-42",
        "citation_label": "APPI Art.41-42 pseudonymized (仮名加工情報) / anonymized (匿名加工情報) processing",
        "title": "Pseudonymization / anonymization of training data",
        "requirement": "Where consent is not obtained, personal data may be used for training only when "
        "processed into pseudonymized or anonymized information under the APPI standards, with "
        "the prescribed safeguards and no re-identification.",
        "recommended_action": "Apply pseudonymization / anonymization to the training corpus before "
        "fine-tuning; prohibit re-identification and retain the deletion-information "
        "safeguards.",
        "audit_step": "Document the pseudonymization / anonymization method and the re-identification-"
        "prohibition controls applied to the corpus.",
        "jurisdiction": "JP",
        "keywords": [
            "pseudonym",
            "pseudonymize",
            "pseudonymization",
            "anonymize",
            "anonymization",
            "anonymous",
            "de-identify",
            "deidentify",
            "masking",
            "仮名加工",
            "匿名加工",
            "加工",
        ],
        "applies_to": ["rlhf", "sft", "dpo", "instruction_tuning", "personal_data", "customer_data"],
    },
    {
        "clause_id": "euaiact-art10-governance",
        "law": "EU AI Act",
        "article": "Art.10(2)",
        "citation_label": "EU AI Act Art.10(2) data governance & management practices",
        "title": "Data governance & management practices for training/validation/testing datasets",
        "requirement": "Training, validation and testing datasets must be subject to data-governance and "
        "management practices covering design choices, collection, preparation, assumptions, "
        "and examination for biases.",
        "recommended_action": "Establish documented data-governance practices for the fine-tuning datasets: "
        "provenance, collection process, labelling, and bias examination.",
        "audit_step": "Maintain a data-governance record covering design choices, data origin, preparation "
        "steps, and the bias examination performed on each dataset.",
        "jurisdiction": "EU",
        "keywords": [
            "data governance",
            "governance",
            "management",
            "training data",
            "training dataset",
            "validation",
            "testing",
            "collection",
            "preparation",
            "eu ai act",
            "article 10",
            "art.10",
            "dataset",
            "provenance",
            "documentation",
            "high-risk",
            "high risk",
        ],
        "applies_to": ["rlhf", "sft", "dpo", "instruction_tuning", "pretraining", "any"],
    },
    {
        "clause_id": "euaiact-art10-quality",
        "law": "EU AI Act",
        "article": "Art.10(3)",
        "citation_label": "EU AI Act Art.10(3) data quality — relevant, representative, error-free, complete",
        "title": "Data quality criteria for datasets",
        "requirement": "Datasets must be relevant, sufficiently representative, and to the best extent "
        "possible free of errors and complete in view of the intended purpose.",
        "recommended_action": "Assess and document dataset relevance, representativeness, error rate, and "
        "completeness for the fine-tuning objective.",
        "audit_step": "Record the data-quality assessment (relevance / representativeness / error rate / "
        "completeness) with the metrics and thresholds used.",
        "jurisdiction": "EU",
        "keywords": [
            "data quality",
            "quality",
            "representative",
            "representativeness",
            "error",
            "errors",
            "complete",
            "completeness",
            "relevant",
            "bias",
            "eu ai act",
            "article 10",
            "art.10",
        ],
        "applies_to": ["rlhf", "sft", "dpo", "instruction_tuning", "pretraining", "any"],
    },
    {
        "clause_id": "euaiact-art10-special",
        "law": "EU AI Act",
        "article": "Art.10(5)",
        "citation_label": "EU AI Act Art.10(5) special categories of personal data for bias monitoring",
        "title": "Processing special-category personal data for bias detection/correction",
        "requirement": "Special categories of personal data may be processed for bias monitoring / correction "
        "only under strict conditions (necessity, safeguards, access controls, deletion).",
        "recommended_action": "Restrict any special-category data used for bias correction to the necessary "
        "minimum with documented safeguards, access controls, and deletion policy.",
        "audit_step": "Document the necessity justification, technical safeguards, access controls, and "
        "deletion policy for any special-category data used for bias monitoring.",
        "jurisdiction": "EU",
        "keywords": [
            "special category",
            "special categories",
            "sensitive",
            "bias monitoring",
            "bias detection",
            "bias correction",
            "safeguards",
            "eu ai act",
            "article 10",
            "art.10",
            "race",
            "health",
        ],
        "applies_to": ["rlhf", "sft", "dpo", "bias", "special_category"],
    },
    {
        "clause_id": "japan-ai-act-governance",
        "law": "Japan AI Act",
        "article": "Ch.III-IV (AI事業者ガイドライン)",
        "citation_label": "Japan AI Act Ch.III-IV / METI-MIC AI事業者ガイドライン governance & transparency",
        "title": "AI business-operator governance, risk management & transparency",
        "requirement": "AI business operators should implement risk-based governance, maintain transparency "
        "and accountability, and manage training-data appropriateness across the AI lifecycle.",
        "recommended_action": "Adopt the AI事業者ガイドライン governance controls: risk assessment, "
        "training-data appropriateness review, and transparency / accountability records "
        "for the fine-tuned model.",
        "audit_step": "Record the lifecycle governance controls: risk assessment, training-data appropriateness "
        "review, and transparency / accountability documentation.",
        "jurisdiction": "JP",
        "keywords": [
            "japan ai act",
            "ai business operator",
            "ai事業者",
            "ガイドライン",
            "governance",
            "transparency",
            "accountability",
            "risk",
            "lifecycle",
            "guideline",
            "guidelines",
        ],
        "applies_to": ["rlhf", "sft", "dpo", "instruction_tuning", "pretraining", "any"],
    },
    {
        "clause_id": "nist-rmf-provenance",
        "law": "NIST AI RMF",
        "article": "MAP-2 / MEASURE-2",
        "citation_label": "NIST AI RMF 1.0 MAP-2 / MEASURE-2 dataset documentation & provenance",
        "title": "Dataset documentation, provenance & lineage",
        "requirement": "Document dataset provenance, lineage, and characteristics (data cards / datasheets) so "
        "training-data origin and processing are traceable and measurable.",
        "recommended_action": "Produce a data card / datasheet per fine-tuning dataset capturing provenance, "
        "lineage, collection, and known limitations.",
        "audit_step": "Maintain a data card / datasheet with dataset provenance, lineage, and known limitations "
        "for each fine-tuning dataset.",
        "jurisdiction": "US",
        "keywords": [
            "nist",
            "ai rmf",
            "rmf",
            "provenance",
            "lineage",
            "data card",
            "datasheet",
            "documentation",
            "traceable",
            "traceability",
            "measure",
            "map",
            "origin",
        ],
        "applies_to": ["rlhf", "sft", "dpo", "instruction_tuning", "pretraining", "any"],
    },
    {
        "clause_id": "nist-rmf-govern",
        "law": "NIST AI RMF",
        "article": "GOVERN-1",
        "citation_label": "NIST AI RMF 1.0 GOVERN-1 data-governance policies & accountability",
        "title": "Data-governance policies, roles & accountability",
        "requirement": "Establish data-governance policies, roles, and accountability so data used across the "
        "AI lifecycle (incl. fine-tuning) is managed under a documented governance structure.",
        "recommended_action": "Define data-governance policies, ownership roles, and accountability for the "
        "fine-tuning data pipeline.",
        "audit_step": "Record the data-governance policy, the accountable owner, and the review cadence for the "
        "fine-tuning data pipeline.",
        "jurisdiction": "US",
        "keywords": [
            "nist",
            "ai rmf",
            "rmf",
            "govern",
            "policy",
            "policies",
            "accountability",
            "roles",
            "ownership",
            "data governance",
            "governance structure",
        ],
        "applies_to": ["rlhf", "sft", "dpo", "instruction_tuning", "pretraining", "any"],
    },
]

# jurisdiction hint tokens → canonical jurisdiction code (used to boost relevant clauses)
_JURISDICTION_ALIASES = {
    "jp": "JP",
    "japan": "JP",
    "japanese": "JP",
    "appi": "JP",
    "日本": "JP",
    "eu": "EU",
    "europe": "EU",
    "european": "EU",
    "gdpr": "EU",
    "us": "US",
    "usa": "US",
    "nist": "US",
    "united states": "US",
}
_TECHNIQUE_ALIASES = {
    "rlhf": "rlhf",
    "reinforcement learning from human feedback": "rlhf",
    "sft": "sft",
    "supervised fine-tuning": "sft",
    "supervised fine tuning": "sft",
    "dpo": "dpo",
    "direct preference optimization": "dpo",
    "instruction tuning": "instruction_tuning",
    "instruction-tuning": "instruction_tuning",
    "pretraining": "pretraining",
    "pre-training": "pretraining",
    "continued pretraining": "pretraining",
    "fine-tuning": "sft",
    "fine tuning": "sft",
    "finetuning": "sft",
}

_TOKEN = re.compile(r"[a-z0-9]+|[ぁ-んァ-ヶ一-龠]+")
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "for",
        "and",
        "or",
        "in",
        "on",
        "we",
        "our",
        "is",
        "are",
        "what",
        "which",
        "how",
        "do",
        "does",
        "under",
        "with",
        "this",
        "that",
        "when",
        "using",
        "use",
        "used",
        "apply",
        "applies",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase word/kana/kanji tokens, stopwords removed (deterministic, no LLM)."""
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 1]


# The platform's S-2 personal-data pass runs before any template code and replaces anything its person-name
# heuristic matches (a run of two or more Title-Case words, e.g. "Act Article" in "EU AI Act Article 10",
# "Personal Information Protection Commission", "United States") with this token. The template cannot switch
# that pass off (the input gate is final). Clauses that only the masked words would have matched cannot be
# retrieved, so a masked request is reported with a limitation code instead of as a complete answer.
PLATFORM_MASK_TOKEN = "[MASKED]"
# Stable machine-readable limitation code (the human-readable explanation goes in the envelope `message`).
QUERY_TERMS_MASKED = "QUERY_TERMS_MASKED"


def masked_request_fields(
    query: str, jurisdictions: list[str], technique: Any, technique_code: str | None
) -> list[str]:
    """Names of the request fields that drive retrieval / applicability and reached the agent masked.

    `query` drives retrieval and domain classification, `jurisdictions` the jurisdiction boost and
    applicability, `training_technique` the per-clause applicability. A masked `training_technique` that still
    normalised (e.g. "[MASKED]-Tuning (SFT)" → sft) lost nothing and is not listed. Dataset descriptors are not
    read by retrieval (they are echoed in the audit trail, where a masked value stays visible), so they are not
    listed.
    """
    fields: list[str] = []
    if PLATFORM_MASK_TOKEN in (query or ""):
        fields.append("query")
    if any(PLATFORM_MASK_TOKEN in str(j) for j in jurisdictions or []):
        fields.append("jurisdictions")
    if PLATFORM_MASK_TOKEN in str(technique or "") and technique_code is None:
        fields.append("training_technique")
    return fields


class FineTuningGovernanceService:
    """Deterministic legal-domain classification, BM25-lite retrieval, mapping, and checklist synthesis."""

    # ── legal-domain classification ───────────────────────────────────────────
    @staticmethod
    def classify_domains(query: str, jurisdictions: list[str] | None) -> list[str]:
        """Classify a query + explicit jurisdictions into legal-domain codes (deterministic keyword match).

        Single-word jurisdiction aliases are matched against whole query tokens (so ``us`` does not
        false-positive on ``customer``); multi-word aliases match as a phrase substring.
        """
        low = (query or "").lower()
        tokens = set(_tokenize(query))
        domains: set[str] = set()
        code: str | None
        for token, code in _JURISDICTION_ALIASES.items():
            if " " in token:
                if token in low:
                    domains.add(code)
            elif token in tokens:
                domains.add(code)
        for j in jurisdictions or []:
            code = _JURISDICTION_ALIASES.get(str(j).strip().lower())
            if code:
                domains.add(code)
        # EU AI Act / NIST are commonly implied by data-governance vocabulary even without an explicit region.
        if "eu ai act" in low or "article 10" in low or "art.10" in low:
            domains.add("EU")
        return sorted(domains)

    @staticmethod
    def normalize_technique(value: Any) -> str | None:
        low = str(value or "").strip().lower()
        if not low:
            return None
        for token, code in _TECHNIQUE_ALIASES.items():
            if token in low:
                return code
        return None

    # ── BM25-lite retrieval ────────────────────────────────────────────────────
    @staticmethod
    def retrieve(query: str, domains: list[str], top_k: int = 8) -> list[dict[str, Any]]:
        """Keyword-scored retrieval over the governance KB. [] when nothing matches (out-of-scope).

        Score = (matched clause keywords, each weighted) + jurisdiction-domain boost. Deterministic and
        auditable — no vector model, no LLM. Ties break on clause_id for a stable order.
        """
        q_tokens = set(_tokenize(query))
        q_low = (query or "").lower()
        scored: list[tuple[int, dict[str, Any]]] = []
        for rec in GOVERNANCE_KB:
            score = 0
            for kw in rec["keywords"]:
                kw_low = kw.lower()
                if " " in kw_low:
                    if kw_low in q_low:
                        score += 3  # multi-word phrase match is a strong signal
                elif kw_low in q_tokens:
                    score += 2
            if score and domains and rec["jurisdiction"] in domains:
                score += 2  # jurisdiction relevance boost (only when the clause already matched)
            if score:
                scored.append((score, rec))
        scored.sort(key=lambda x: (-x[0], x[1]["clause_id"]))
        out: list[dict[str, Any]] = []
        for score, rec in scored[:top_k]:
            out.append(
                {
                    "clause_id": rec["clause_id"],
                    "law": rec["law"],
                    "article": rec["article"],
                    "citation_label": rec["citation_label"],
                    "title": rec["title"],
                    "requirement": rec["requirement"],
                    "recommended_action": rec["recommended_action"],
                    "audit_step": rec["audit_step"],
                    "jurisdiction": rec["jurisdiction"],
                    "applies_to": rec["applies_to"],
                    "score": score,
                    # KB-owned grounded citation surrogate (never caller-supplied); minted via the single
                    # authoritative formula the S-3 gate re-derives to authenticate each delivered obligation.
                    "citation": statutory_citation(rec["clause_id"], rec["citation_label"]),
                }
            )
        return out

    # ── clause → context mapping ────────────────────────────────────────────────
    @staticmethod
    def map_clause(clause: dict[str, Any], technique: str | None, domains: list[str]) -> dict[str, Any]:
        """Annotate a retrieved clause with its applicability to the caller context (deterministic)."""
        applies_to = set(clause.get("applies_to", []))
        technique_applies = (technique is None) or (technique in applies_to) or ("any" in applies_to)
        jurisdiction_applies = (not domains) or (clause["jurisdiction"] in domains)
        reasons: list[str] = []
        if technique and technique in applies_to:
            reasons.append(f"applies to training technique '{technique}'")
        if domains and clause["jurisdiction"] in domains:
            reasons.append(f"in scope for jurisdiction {clause['jurisdiction']}")
        if not reasons:
            reasons.append("general data-governance obligation")
        return {
            "clause_id": clause["clause_id"],
            "applies": bool(technique_applies and jurisdiction_applies),
            "applicability": "; ".join(reasons),
        }

    # ── deliverable synthesis ────────────────────────────────────────────────────
    @staticmethod
    def build_checklist(clauses: list[dict[str, Any]], mapped: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compose one compliance-checklist item per retrieved clause, cited to its law+article."""
        applic_by_id = {m["clause_id"]: m for m in mapped}
        checklist: list[dict[str, Any]] = []
        for c in clauses:
            m = applic_by_id.get(c["clause_id"], {})
            checklist.append(
                {
                    "obligation_id": c["clause_id"],
                    "law": c["law"],
                    "article": c["article"],
                    "requirement": c["requirement"],
                    "recommended_action": c["recommended_action"],
                    "audit_step": c["audit_step"],
                    "applicability": m.get("applicability", "general data-governance obligation"),
                    "applies": m.get("applies", True),
                    "citation": c["citation"],  # KB-owned grounded statutory citation
                    "citation_label": c["citation_label"],
                }
            )
        return checklist

    @staticmethod
    def build_dataset_audit_trail(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build the per-dataset audit trail. Each dataset's provenance was resolved once at S-1 (pre_process)
        to a grounded ``src:<sha8>`` citation or ``None`` (unverifiable / forged surrogate dropped). Trust that
        value verbatim; never re-resolve or fabricate. A ``None`` provenance → the item carries no citation →
        S-3 blocks the trail as CITATION_INCOMPLETE (fail-closed)."""
        trail: list[dict[str, Any]] = []
        for d in datasets or []:
            if not isinstance(d, dict):
                continue
            # Tokenize unconditionally (no forgeable passthrough). pre_process (S-1) already tokenized the
            # dataset_id; re-tokenizing the surrogate here is a deterministic label-only no-op (dataset_id is
            # a per-invocation display key in this trail, never rejoined against the S-1 value), so a second
            # hash is harmless.
            dataset_id = safe_identifier(d.get("dataset_id") or d.get("id") or "")
            provenance = d.get("source")  # already resolved (src:<sha8> or None) at S-1
            data_types = [str(t) for t in d.get("data_types", []) if isinstance(t, (str, int))]
            trail.append(
                {
                    "dataset_id": dataset_id,
                    "data_types": data_types,
                    "provenance": provenance,
                    "provenance_status": "documented" if provenance else "unverified",
                    # grounded provenance citation only when resolved; None → S-3 fail-closed
                    "citation": provenance,
                }
            )
        return trail

    @staticmethod
    def synthesize(
        query_domains: list[str],
        technique: str | None,
        checklist: list[dict[str, Any]],
        audit_trail: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the governance-advisory deliverable (checklist + dataset audit trail + citations)."""
        # distinct statutory citations (grounded, KB-owned)
        seen: set[str] = set()
        citations: list[dict[str, str]] = []
        for item in checklist:
            if item["citation"] in seen:
                continue
            seen.add(item["citation"])
            citations.append(
                {
                    "law": item["law"],
                    "article": item["article"],
                    "citation": item["citation"],
                    "citation_label": item["citation_label"],
                }
            )
        laws = sorted({item["law"] for item in checklist})
        return {
            "status_kind": "governance_advisory",
            "query_domains": query_domains,
            "training_technique": technique,
            "laws_covered": laws,
            "compliance_checklist": checklist,
            "dataset_audit_trail": audit_trail,
            "citations": citations,
            "confidence": "medium" if checklist else "low",
            "message": (
                "参考: 以下は本問い合わせに該当する公開法令クローズに基づく DRAFT のデータガバナンス"
                "チェックリストです。最終的な適法性判断は法務 / DPO にご確認ください。"
            ),
        }
