# Experiment protocol - draft for supervisor review
Somto Terence Ndu | DAIM_A_105

## Questions
RQ1: Which endorsed discriminatory rhetorical strategies recur across the four target dimensions?
RQ2: How do strategy use, combinations and linguistic realisations differ within the sampled sources?
RQ3: How accurately and consistently can an LLM support annotation against independent human reference labels?

## Scope and sampling
English text comments; aim 1,200-2,000 unique posts if coverage supports it. Pilot 100-150 posts.
Sample target dimensions within matched sources/time periods where possible and retain intersections.
Use corpus-level identity metadata only to sample candidates; human hostile-stance assessment determines
whether a post belongs in discrimination comparisons. Neutral and counter-speech controls remain in evaluation.
Report sampling probabilities or purposive selection rules. Do not infer population prevalence from quotas.

## Annotation and independence
Use the definitions and exclusion rules in ANNOTATION_GUIDE.md. Extend the guide through pilot review,
then freeze it before evaluating. Keep quoted/counter-speech separate from endorsement. Labels allow uncertainty.
Humans annotate blind to model labels. Double-annotate at least 20% of the human sample if resources permit.
Compute raw agreement, kappa and positive-label counts per strategy before adjudicating discrepancies.
Adjudicated human reference labels remain conceptually distinct from LLM predictions.
Do not report one LLM's agreement with another LLM as human validation.

## Splits and experiments
Split connected author/thread/duplicate groups before selecting prompt examples or fitting a vectoriser.
Use training rows for prompt examples and TF-IDF model fitting, development rows for prompt revision,
and a frozen test subset for the single final comparison. The prepared split is deterministic (seed 42).
The hash split does not guarantee sufficient target coverage: audit before freezing it.

Compare codebook-only zero-shot with human-training-example few-shot, using one pinned provider model
version and the same test rows. Fix example selection and retain full prompt/schema/config hashes.
Default call limit is 10; increase deliberately after inspecting pilot outputs and estimating cost.
This is prompted LLM inference, not LLM fine-tuning. Fine-tuning is optional only if warranted by data.
The TF-IDF rhetoric model is a genuine supervised comparator once human training labels exist.
The old toxicity baseline addresses a different task and is not an RQ3 comparator.

## Metrics and failure handling
Per-strategy precision/recall/F1, annotation coverage, gold positives, missing calls and abstentions;
per-target breakdowns use HUMAN target labels. Report coverage beside conditional F1; abstentions
must not silently become absent labels. Track refusals, incomplete outputs, invalid evidence and API errors.
A failed response stops the run; successful prior responses are resumable. Do not replace failures with negatives.
LLM target and stance validation requires separate confusion/error analysis; current scoring CLI focuses on strategies.
Repeat a fixed development subset to assess model stability if resources allow. Temperature is not assumed
supported or deterministic; record returned model identifiers and acknowledge provider variability.

## Comparisons
Use independently human-labelled targets and strategy labels for confirmatory comparisons.
Report proportions, intersection counts, co-occurrence and close-read representative/disconfirming examples.
Cluster bootstrap connected group IDs for sample uncertainty. Sparse categories may yield unstable estimates.
Multi-target posts occur in multiple descriptive groups; comparisons are dependent. No naive independent-group tests.
Model-only expanded-corpus comparisons are exploratory and must be labelled as such, with error sensitivity.
Do not infer causal mechanisms, author motivation or equivalence between systems of discrimination.

## Preregistered decision points
Freeze codebook version, sample rules, group definition, model snapshot, prompts, example IDs and metrics
before held-out analysis. Document any deviations with dates. Results and conclusions remain pending until
an approved corpus, independent reference labels and actual inference are available.
