# Candidate corpus audit - 8 October 2026

## Recommended for investigation
UC Berkeley D-Lab, Measuring Hate Speech:
https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech
Project: https://hatespeech.berkeley.edu/

The dataset card lists English comments, comment_id, platform and repeated annotator_id rows.
It documents target race, gender, transgender and sexual-orientation fields, plus ordinal
attributes including dehumanize. The card lists CC BY 4.0. This is a candidate, not a claim
that the four discrimination categories are balanced or that target mentions imply hostility.

Before analysis, aggregate annotation rows by comment_id; report agreement/disagreement rather
than counting repeated annotations as independent posts. Preserve original target labels as sampling
metadata. Do not turn target_gender_women automatically into sexism or a transgender target flag
into transphobia. Manually verify hostile stance and targeted dimension.
The ordinal dehumanize item is NOT interchangeable with the project's binary dehumanisation codebook.
A mapping requires documented item wording/thresholds and independent validation.

## Data import contract
Convert an approved source to data/corpus_template.csv's columns:
- post_id: source's stable comment ID (one row per post).
- text: original text needed for annotation; strip unnecessary identifiers.
- source: platform/source, ideally with matched period metadata in a separate local provenance file.
- group_id: author/thread connected component if available; otherwise post ID and disclose missing grouping.
- target_identities: semicolon-separated candidate dimensions; multiple allowed, blank permitted.

Run `python -m rhetoric.prepare --input data/local/corpus.csv --output data/processed`.
This merges exact redacted-text duplicates and connected group components before a deterministic
60/20/20 split. It emits two blind annotation sheets and an aggregate audit. Inspect near duplicates
and source/target coverage manually. Candidate target labels are not shown in blind sheets or LLM prompts.
Redaction is limited to URLs, email patterns and @handles; it is not a guarantee of anonymisation.

## Remaining checks
- Actual unique-post counts by source and each target, including intersections and hostile stance.
- Platform code mappings, collection dates, source permissions and university ethics position.
- Minimum samples per target/source; narrow scope if transphobic posts are too sparse.
- Near duplicates and missing author/thread metadata; discuss effects on evaluation.
- Actual dataset file SHA-256 and a pinned source revision.

Raw posts, human annotations, API output and annotator demographic fields must stay out of the public repo.
Only reviewed aggregate results belong in reports/. No corpus has been downloaded in this upgrade.
