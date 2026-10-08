# Shared Rhetorical Strategies Across Identity-Based Discrimination in Public Online Discourse
Somto Terence Ndu | Master's project DAIM_A_105

**Working draft: introduction and proposed methods. Empirical chapters remain pending.**

## Abstract
This project investigates rhetorical similarities and differences across racism, sexism, homophobia
and transphobia in a bounded corpus of English-language public online discourse. It proposes a
mixed-methods framework combining independent human annotation, structured LLM-assisted annotation,
quantitative comparison and qualitative close reading. Five candidate strategies are examined:
dehumanisation, threat framing, moral condemnation, essentialisation and exclusion/hierarchy.
The central methodological question is whether an LLM can identify these strategies sufficiently
reliably to support interpretation across different target dimensions. No rhetorical findings are
reported in this draft because data approval, human reference annotation and live model evaluation
have not yet been completed.

## 1. Introduction
A post-level toxicity label can identify content for further scrutiny, but it cannot by itself explain
how an identity group is represented or why unequal treatment is presented as acceptable. This
project therefore distinguishes the detection of broadly abusive content from the interpretation
of rhetorical strategies. It asks whether discriminatory discourse directed at different identities
uses recurring forms of argument and representation, while retaining the social specificity of each
form of discrimination.

The rationale does not assume that racism, sexism, homophobia and transphobia are interchangeable.
A claim of structural similarity is narrower: speakers may employ comparable argumentative forms
against different targets. For instance, threat construction can be investigated as a strategy while
the object of the alleged threat, the proposed response and the relevant social history differ.
Whether such similarities occur in the selected material is an empirical question.

The research questions are: (1) which strategies recur across the four discriminatory dimensions;
(2) how their distributions, combinations and linguistic expression differ within the corpus; and
(3) how reliably LLM annotations align with an independent human reference. The anticipated
contribution is a transparent analytical framework and an evaluated workflow, rather than a
universal detector or a causal explanation of prejudice.

## 2. Literature and conceptual positioning
Davidson et al. (2017) distinguish hate speech from offensive language in a classification task.
Their work provides a useful computational reference but does not supply the rhetorical labels
required here. Matamoros-Fernández and Farkas (2021) call attention to the theoretical,
methodological and contextual limitations of research on racism and social media. Almagro et al.
(2026) examine hate narratives through corpus-based work, providing motivation for studying
less explicit forms of harmful discourse.

These sources motivate three distinctions: discriminatory targeting is not equivalent to an identity
mention; rhetorical interpretation is not equivalent to toxicity classification; and computational
patterns require contextual interpretation. The five strategy definitions in this project remain
operational candidates to be refined through pilot annotation. This section requires a fuller,
systematically documented literature review on each strategy and on LLM annotation evaluation
before submission. A broad gap claim that no prior cross-identity comparison exists is not made.

## 3. Methodology
### 3.1 Design and corpus
The study will combine content analysis, computational annotation and qualitative interpretation.
One unique post is the unit of analysis. Repeated annotation rows in a source dataset will not be
counted as independent posts. Sources and collection periods will be matched across target
categories where feasible. Target identity fields in an existing dataset will be used as sampling
metadata, with discriminatory stance determined separately. Multiple targeted identities will be
retained to preserve intersectional cases.

The Berkeley Measuring Hate Speech corpus is a candidate because its documented schema includes
multiple identity dimensions and comment identifiers. Its suitability remains conditional on an
actual audit of target coverage, source characteristics and access/ethics requirements. The Davidson
corpus supports a preliminary toxicity baseline but cannot supply the main rhetorical reference.

### 3.2 Annotation
The preliminary codebook defines dehumanisation, threat framing, moral condemnation,
essentialisation and exclusion/hierarchy. Each strategy may be present, absent or uncertain.
A positive label requires endorsement of identity-based hostility; quoted or rejected material is
not automatically treated as the author's stance. Counter-speech, neutral discussion and uncertain
stance will be separately recorded. Positive model annotations must include an exact text span,
which supports traceability without treating the model's explanation as proof.

A pilot will refine boundaries and ambiguous cases. Two human raters, where available, will annotate
independently before adjudication. Raw agreement, per-strategy kappa, label counts and uncertainty
will be reported. The reference sample will be reserved independently of prompt development.
If independent annotation cannot be obtained, that limitation will be stated explicitly.

### 3.3 RoBERTa classification experiment
The selected encoder is FacebookAI/roberta-base, with low-rank adapters on attention query/value
projections and a trained classification head. The selected Davidson data provide three labels:
hate speech, offensive language and neither. Exact normalised duplicates and conflicting-label
duplicates are removed before a stratified 70/15/15 train/validation/test split. The best validation
macro-F1 checkpoint is selected across three epochs, followed by one held-out test evaluation.
A TF-IDF comparator is trained on the identical partition. This experiment assesses hate/offensive
classification; it does not answer the rhetorical questions without additional reference labels.
RoBERTa is an encoder pretrained language model, rather than a generative chat model.

### 3.4 Optional generative LLM experiment
The implementation supports a codebook-only condition and a few-shot condition using human-labelled
training examples. Candidate target labels are withheld from prompts. Posts are treated as untrusted
input, and the output is constrained to a structured schema. Model identifiers, prompt/configuration
hashes, input hashes and token usage are recorded locally. Refusals, incomplete outputs and invalid
evidence stop a run instead of being converted into negative annotations.

Train/development/test partitions are assigned after connecting duplicate and provided grouping
relationships. Near-duplicate inspection and missing author/thread metadata require additional audit.
The LLM is used through prompting; no claim of fine-tuning is made. A supervised TF-IDF rhetorical
baseline can be fitted after human training labels are available.

### 3.5 Evaluation and comparative analysis
Evaluation will report per-strategy precision, recall and F1 alongside the coverage of non-abstained
predictions. Human target labels determine subgroup breakdowns. Excluding uncertain outputs from
conditional metrics requires explicit reporting of the excluded proportion. Rater agreement and
model agreement answer different questions and will be presented separately.

Human-labelled strategy frequencies and combinations will form the principal comparative evidence.
Group-bootstrap intervals will describe sample uncertainty while retaining multi-target membership.
Close reading will examine representative and disconfirming cases, including instances where an
apparently shared label obscures differences in meaning. Source imbalance, sparse categories and
annotation uncertainty will constrain the conclusions.

### 3.6 Ethics and reproducibility
The university's requirements for secondary online-data research and external model processing
must be confirmed before substantive collection and inference. Raw posts and annotations remain
outside the public repository. Automated redaction of handles, email patterns and URLs reduces
some identifiers but does not guarantee anonymity. Publication examples require separate review
for searchability and potential harm. Annotation sessions will be limited to support researcher wellbeing.
Code, configurations and reviewed aggregate outputs will support reproducibility. AI assistance in
software and writing should be disclosed according to the university's assessment rules.

## 4. Results - pending
Only a separate preliminary hate/offensive/neither classifier has been evaluated: macro F1 0.733
on 4,909 held-out Davidson posts. This result answers neither the rhetorical comparison questions
nor the LLM validation question. No table of rhetorical performance or shared-strategy findings is
provided until independent reference labels and actual experiments exist.

Required empirical outputs: corpus/source/target audit; annotation counts and agreement; LLM
coverage and strategy metrics by condition and target; strategy proportions and co-occurrence;
qualitative examples reviewed for publication; uncertainty and error analyses.

## 5. Discussion and conclusion - pending
Interpret the observed results only after the analyses above. Address whether shared strategies
were found, which apparent differences may reflect sampling or model errors, how interpretation
changes across contexts, and what the evidence cannot establish. Do not insert hypothetical findings.

## References
Almagro, M., Vieites, C., Chakkour, H. and Sancho, A. (2026). How can hate narratives be tracked
in online environments? A corpus-based study. Frontiers in Communication, 11, 1743196.
https://doi.org/10.3389/fcomm.2026.1743196

Davidson, T., Warmsley, D., Macy, M. and Weber, I. (2017). Automated Hate Speech Detection and
the Problem of Offensive Language. ICWSM, 11(1), 512-515. https://arxiv.org/abs/1703.04009

Matamoros-Fernández, A. and Farkas, J. (2021). Racism, Hate Speech, and Social Media:
A Systematic Review and Critique. Television & New Media, 22(2), 205-224.
https://doi.org/10.1177/1527476420982230

UC Berkeley D-Lab (n.d.). Measuring Hate Speech [dataset].
https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech

OpenAI (n.d.). Structured model outputs [API documentation].
https://developers.openai.com/api/docs/guides/structured-outputs
