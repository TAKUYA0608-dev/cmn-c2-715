# Test Specification — CMN-C2-715

## Test Strategy
- Coverage target: **80%+** (achieved **94%**, `--cov=src`)
- Test types: Unit (pre/post + inner nodes + services) / Unit (Cat 2 graph wiring + real invoke) / Integration / Proof-of-Boundary
- Determinism: **no LLM** — legal-domain classification, BM25-lite clause retrieval, clause-to-context
  mapping, and checklist / audit-trail synthesis are pure tokenization + keyed KB composition (reproducible,
  auditable, grounded in a cited public-law article). No model is declared in `config/agent.yaml` and no LLM
  dependency in `pyproject.toml`.

## Framework Compliance Tests (Mandatory)

| TC-ID | Test | Expected Result | Result |
|-------|------|----------------|--------|
| TC-01 | State contract: flat TypedDict | `State(AgentState)`, NotRequired primitives + JSON strings; no PII/credential fields | ✅ PASS |
| TC-02 | S-2 rejection is degraded, never `status=ERROR` | `_extra_security_gate_input` sets `error_code` (`INJECTION_REJECTED`/`INPUT_TOO_LONG`) and returns `dict(state)`; never raises, never sets `status=ERROR` | ✅ PASS |
| TC-03 | No JWT/Credential in `src/` | `gate-credential-scan`: 0 violations | ✅ PASS |
| TC-04 | InvocationContext read-only | never stored in State | ✅ PASS |
| TC-05 | S-4: no duplicate lifecycle events | only domain events emitted (never node_start/complete) | ✅ PASS |
| TC-06 | S-2 `_security_gate_input()` not overridden | `@final`; only `_extra_*` extended | ✅ (real SDK on CI; local-stub env-diff) |
| TC-07 | S-3 `_security_gate_output()` not overridden | `@final`; may raise via `_extra_*` | ✅ (real SDK on CI; local-stub env-diff) |
| TC-08 | `required_trust_level` explicit on every FunctionNode | `VERIFIED_EXTERNAL` (gate-trust-level-check) | ✅ PASS |
| TC-09 | Cat consistency | Template ID / config / README all Cat 2 (gate-cat-consistency) | ✅ PASS |
| TC-10 | Exact dependency pins | `==` in all sections incl. `[build-system]` setuptools==68.0.0 (gate-dep-pinning) | ✅ PASS |
| TC-11 | S-4: ≥1 domain `emit_trace_event()` per `execute()` | emitted on every path (incl. skip / safe branches) | ✅ PASS |
| TC-12 | Degraded path never `status=ERROR` | injection / oversize / empty / 0-clause → `SUCCESS.value + error_code`; `post_process` still runs | ✅ PASS |
| TC-13 | Input hygiene: PII / secrets never persist in State | credential / My-Number / email / phone redacted from every supplied free-text field (query + dataset descriptors); dataset ids tokenized | ✅ PASS |
| TC-14 | Real `Graph().invoke()` degraded path | injection & oversize → SUCCESS + out-of-scope, `PostProcessNode` in node_history, error_code in terminal S-4 audit, rejected body absent, DRAFT disclaimer present | ✅ PASS |
| TC-15 | S-3 fail-closed citation completeness | grounded advisory with no statutory citation, or any supplied dataset with unresolved provenance → `needs_review` degrade (SUCCESS + `CITATION_INCOMPLETE`), deliverable body withheld, disclaimer + audit still run; verified via real `Graph().invoke()` | ✅ PASS |
| TC-16 | Provenance allowlist + output redaction | unverifiable / forged dataset `source` dropped, never in `formatted_output`; query PII (email/phone/credential) redacted in a grounded output | ✅ PASS |
| TC-17 | Opaque-id boundary — privacy tokenize (dataset_id) | `dataset_id`/`id` → `ds:<sha8>` UNCONDITIONALLY (a no-space name `Alice`/`John.Smith`/`TaroYamada` is tokenized, not passed through); deterministic; surrogate namespace idempotent; verified via real `Graph().invoke()` | ✅ PASS |
| TC-18 | Provenance validation (separate from privacy) | a caller `source` is a grounded dataset citation ONLY if it names an authorized data-governance system of record; unverifiable source (dataset name / `unknown` / fabricated) **and a forged surrogate (`src:1a2b3c4d` / `ds:deadbeef`)** → `None` → `needs_review` (CITATION_INCOMPLETE), never a citation; authorized `data_catalog:…` / `mlflow:…` → privacy-tokenized `src:<sha8>`; verified via real `Graph().invoke()` | ✅ PASS |

## Proof-of-Boundary Tests (Mandatory)

| PB-ID | Boundary | Expected Result | Result |
|-------|----------|----------------|--------|
| PB-2/5 | Post-invoke State is primitives only; no credential fields | AST scan: 0 violations | ✅ PASS |
| PB-4 | Import isolation — no Level 0 (`agenticstar`) imports | AST scan: 0 violations | ✅ PASS |
| PB-6 | Invoke order S-1 → S-4(start) → S-2 → execute → S-3 → S-4(complete) | Order verified | ✅ (real SDK on CI; local-stub env-diff) |
| PB-7 | HITL interrupt propagation | conditional — SKIPPED (`hitl.enabled` not set for this template) | ✅ (n/a, skip) |
| S-0 | Cat 2 `GraphNode`-in-main wraps inner `BaseGraph` (cached `get_subgraph`) | gate-composition passes | ✅ PASS |

## Business Logic Tests

| BL-ID | Test | Input | Expected Result | Result |
|-------|------|-------|----------------|--------|
| BL-01 | Legal-domain classification | query "APPI + EU AI Act Article 10 …" | `{JP, EU}`; token-based (no `us` false-positive on `customer`) | ✅ PASS |
| BL-02 | BM25-lite retrieval | governance query | ≥1 relevant clause (appi-consent + euaiact…); every clause KB-cited `stat:<sha8>`; descending score, stable tie-break | ✅ PASS |
| BL-03 | Out-of-domain retrieval | "how do I bake a cake" | `[]` → out-of-scope safe answer | ✅ PASS |
| BL-04 | Clause → context mapping | clause + technique + jurisdictions | applicability annotation; general-obligation fallback | ✅ PASS |
| BL-05 | Checklist synthesis | retrieved + mapped clauses | one obligation per clause with requirement + audit step + law+article citation | ✅ PASS |
| BL-06 | Dataset audit trail | resolved vs unresolved provenance | `documented`+citation vs `unverified`+`None`; malformed rows dropped | ✅ PASS |
| BL-07 | Grounded advisory (no datasets) | statutory question only | `governance_advisory`, citations present, `citation_complete=True` | ✅ PASS |
| BL-08 | Grounded advisory (authorized dataset) | dataset with `data_catalog:…` source | grounded; dataset `provenance_status=documented`, id `ds:<sha8>` | ✅ PASS |
| BL-09 | Empty input degrades | "   " | `SUCCESS.value + INPUT_REJECTED`, still audits | ✅ PASS |
| BL-10 | Mandatory DRAFT disclaimer | any output | S-3 gate blocks output missing 参考/DRAFT | ✅ PASS |
| BL-11 | S-3 fail-closed (dataset provenance) | supplied dataset with unresolved/forged source | grounded advisory degrades to `needs_review`, deliverable withheld, `CITATION_INCOMPLETE` | ✅ PASS |
| BL-12 | S-3 fail-closed (no statutory citation) | grounded report with empty citations | `needs_review` (CITATION_INCOMPLETE) | ✅ PASS |
| BL-13 | Privacy tokenize (dataset_id) | `dataset_id` = `Taro Yamada 090-…` **or no-space name** | tokenized to `ds:<sha8>`; original absent from output | ✅ PASS |
| BL-14 | Provenance safety / no leak | unverifiable `source` (name+phone) | dropped/redacted — not in `formatted_output` | ✅ PASS |
| BL-15 | Provenance validation (fail-closed) | `source` = dataset name / `unknown` / fabricated | no citation → `needs_review`; source not in output | ✅ PASS |
| BL-16 | Forged-surrogate defence | `source` = `src:1a2b3c4d` / `ds:deadbeef` | unauthorized namespace → dropped → `needs_review`, never a citation; error_code in terminal S-4 audit | ✅ PASS |
| BL-17 | Authorized provenance accepted | `source` = `data_catalog:…` / `mlflow:…` | grounded; citation = tokenized `src:<sha8>`; raw not in output | ✅ PASS |
| BL-18 | Query PII redaction in grounded output | query with email/phone/credential | redacted from `formatted_output` (S-3 defense-in-depth) | ✅ PASS |

## Test Execution Summary
- Total: 75 (unit-nodes 40 + unit-graph 29 + integration 6) + active PB (import_isolation, state_safety, S-0 via graph wiring)
- Pass: 75 (core) · Skip: server-import (local stub env-diff) + PB-7 ×2 (conditional, HITL off)
- env-diff: `test_pb_invoke_order` + `test_framework_compliance_tc06_tc07` (TC-06/07/PB-6) assert against the
  real SDK on CI (the local SDK stub lacks the `emit_trace_event` surface / `@final` gate enforcement); all pass on CI
- Coverage: **94%** (`--cov=src`)
