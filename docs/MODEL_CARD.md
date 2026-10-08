# Baseline model card
Owner: Somto Terence Ndu. Project DAIM_A_105.

## Scope
TF-IDF word unigrams/bigrams with balanced logistic regression on Davidson et al.'s
crowd-labelled English tweets. Outputs hate_speech, offensive_language or neither.
This trained artifact does NOT classify racism, sexism, homophobia, transphobia or rhetorical strategies.
The separate supervised rhetoric mode requires genuine manually annotated data; none has been invented.

## Evaluation
See models/baseline/metrics.json for measured held-out results, per-class scores, confusion matrix,
majority-class comparison, dataset SHA-256 and sample counts. Vocabulary is fitted only on training data.
Exact duplicates after normalisation are removed; conflicting labels are excluded. Seed 42, 80/20 split.
No hyperparameter search was performed. The artifact is fitted only on the training portion.
The original dataset lacks conversation/author metadata in this pipeline. Near-duplicate and source
leakage remain possible. Random held-out results are not estimates for new platforms or identities.

## Limitations and intended use
Academic baseline only; not an automated moderation or decision system. Historic Twitter sampling,
class imbalance, annotator assumptions, dialect bias, missing context and implicit abuse can affect results.
Scores are uncalibrated model probabilities, not factual assessments of a person or intent.
No subgroup fairness claims are supported. Evaluate rhetorical and target labels separately once available.
Raw tweets and usernames are excluded from Git. Model vocabularies may retain offensive tokens from
training data; review before sharing beyond the research context. Load only trusted joblib files.

## Data provenance
Davidson, T., Warmsley, D., Macy, M. and Weber, I. (2017). Automated Hate Speech Detection
and the Problem of Offensive Language. ICWSM 11(1), 512-515. https://arxiv.org/abs/1703.04009
Source: https://github.com/t-davidson/hate-speech-and-offensive-language
Review source access/licensing conditions before reuse; this project grants no new rights to source data.
