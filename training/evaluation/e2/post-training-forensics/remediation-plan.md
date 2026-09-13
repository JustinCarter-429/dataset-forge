# Phase E2.5 Minimal Remediation Plan

## Decision

**A. MODEL_ERROR.** The parser and expected label are correct. The model emitted the semantically correct reason code with the prohibited `reject` decision.

## Smallest defensible change

Create a development-only contrast set for decision severity while leaving the parser, ontology, held-out cases, and thresholds unchanged:

- 24 new records: four dataset types (`scenario_expected_result`, `question_answer`, `instruction_response`, and `custom`) × three paired distinctions × two render variants.
- Pair `REQUIRED_FIELD_MISSING`/`revise` against `MISSING_REQUIRED_CONSTRAINT`/`revise` and genuinely non-repairable dataset-spec failures using an already-approved reject-compatible code.
- Include both supported schema spellings (`required` and `required_fields`) without copying fictional identifiers, wording, field combinations, or source facts from the failed held-out case.
- Add at least four independent `scenario_expected_result` missing-field records to validation; never move validation or held-out records into training.
- Re-run exact, canonical-input, token-Jaccard, source-family, template-family, controlled-overfit, E2.2 smoke, E2.4 validation, and E1 held-out overlap gates before any GPU request.

## Experiment proposal—not authorized

Prefer a small continuation experiment from the safely preserved final E2 adapter, because the evidence shows a narrow decision/code compatibility lapse after an otherwise completed run. Freeze a short step ceiling and evaluate a development-only contrast set plus the same 15-case smoke. A new full 2,382-step run is not presently justified. Request explicit GPU authorization only after the new corpus and overlap evidence are reviewed.

Do not change the strict parser, compatibility table, five-field contract, frozen certification thresholds, or original smoke result.
