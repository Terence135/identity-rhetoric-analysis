# Project status

## Selected RoBERTa experiment
- User selected FacebookAI/roberta-base and tdavidson/hate_speech_offensive.
- Initial CPU fine-tuning was stopped after GPU access diagnosis; an incomplete adapter was saved locally.
- GPU-only launcher is ready for a normal Terminal process with Apple Metal or CUDA access.
- RoBERTa test results and selected adapter are pending completion.

## Implemented and locally verifiable
- Trained hate/offensive/neither reference baseline (legacy task) and saved measured metrics.
- Approved-corpus CSV import, limited identifier redaction, exact duplicate/group-component joining,
  deterministic train/dev/test split, source/target audit and two blind human annotation exports.
- OpenAI Responses structured-output annotator, exact evidence validation, abstention, prompt provenance,
  codebook-only/few-shot conditions, input-change checks, bounded calls and resumable outputs.
- Independent-human reference evaluation, subgroup coverage, two-rater agreement, and descriptive
  group-bootstrap strategy/co-occurrence comparison.
- Research protocol, source audit and dissertation working draft.

## Not completed; no results implied
- University ethics position and source approval.
- Final corpus selection/coverage audit on real project data.
- Human rhetorical annotations and independent agreement assessment.
- Live LLM calls (no key or local LLM configured).
- Held-out rhetorical evaluation, substantive four-group findings, final discussion and dissertation.

Synthetic test fixtures only verify software. They are not a research corpus or empirical evidence.
