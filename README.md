# CMN-C2-715 — LLM Fine-Tuning Data Governance Q&A (APPI 2026 + EU AI Act Art.10)

> **Category**: Cat 2 (domain workflow (a job to be done))
> **Industry**: Common (industry-agnostic)

## Overview

Advisory checklist for the data-governance obligations of fine-tuning an LLM. Given a question with the training technique, jurisdictions and optional dataset descriptors (id, source system, data types), the agent classifies the legal domains involved, retrieves the applicable clauses from a seeded public-law knowledge base (the shipped set cites the Japanese personal-information act, EU AI Act Article 10 and the NIST AI RMF), maps each clause to the stated jurisdictions, technique and data types, and returns a compliance_checklist per obligation, a per-dataset provenance trail, citations and a draft disclaimer. Everything is deterministic; no LLM is used and no training data is ever ingested. Dataset identifiers are replaced by opaque tokens, a dataset whose source is not an authorised system of record loses its citation and causes the checklist to be held back for a person to check, and an out-of-domain question gets an out-of-scope answer. The clause knowledge base shipped here is a small curated sample — replace it with your own.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | 3.11 or later (`requires-python = ">=3.11"`) |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent raises
`PlatformRequired` during graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Known limitations

**Regulation and product names can be masked by the platform before retrieval.** AGENTIC STAR's
personal-data protection runs before this template's code and replaces any run of two or more
Title-Case words with `[MASKED]` — it reads "Act Article" in "EU AI Act Article 10" as a person's
name — and the template cannot switch it off. Measured on AgentCore 1.0.3:

| Request contains | Reaches retrieval as | Result |
|---|---|---|
| `EU AI Act Article 10` | `EU AI [MASKED] 10` | 2 of 4 obligations (Art.10(2) and Art.10(5) missing); `confidence: low`, `limitations: ["QUERY_TERMS_MASKED"]` |
| `EU AI Act, article 10` | unchanged | 4 obligations (APPI + Art.10(2), 10(3), 10(5)); `confidence: medium`, `limitations: []` |
| `Supervised Fine Tuning Consent Requirements` | `[MASKED]` | `status_kind: not_evaluated`, not `out_of_scope` (lower case: the APPI consent obligation) |
| `Does the Personal Information Protection Commission regulate quantum annealing?` | `Does the [MASKED] regulate quantum annealing?` | `not_evaluated` — the unmasked words matching nothing is not treated as "outside the KB" |
| JSON `jurisdictions: ["United States"]`, `training_technique: "Supervised Fine-Tuning"` | `[MASKED]`, `[MASKED]-Tuning` | technique and jurisdiction lost; `confidence: low`, `QUERY_TERMS_MASKED` |
| JSON `training_technique: "Supervised Fine-Tuning (SFT)"` | `[MASKED]-Tuning (SFT)` | still recognised as `sft`; not flagged |
| `What does Taro Yamada recommend for ramen in Osaka?` | `What does [MASKED] recommend for ramen in Osaka?` | `not_evaluated` (an off-topic question that contains a masked name is not called out of scope either) |

A masked request is never reported at normal confidence: `limitations` holds the code
`QUERY_TERMS_MASKED`, and `message` names the masked fields (`query`, `jurisdictions`,
`training_technique`) and how to rephrase. Write regulation names without consecutive capitalised
words (for example `EU AI Act, article 10`, `supervised fine-tuning`, `us`) to get the full checklist.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and operational documentation
```

See `docs/02_design.md` for the design and `docs/03_test_spec.md` for the test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
