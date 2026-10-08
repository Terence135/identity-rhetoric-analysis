# RoBERTa for identity-based hate-speech research
**Somto Terence Ndu | Master's project DAIM_A_105**

The selected model is **FacebookAI/roberta-base**, trained with LoRA adapters and a three-class
classification head on the **Davidson hate_speech_offensive dataset**. RoBERTa is a pretrained
encoder transformer; this project uses supervised classification, not a conversational generative LLM.

## Current status
The initial restricted-process CPU run was stopped. A replacement run has started successfully
using Apple Metal (`device: mps`) from the normal Terminal, with output in models/roberta_gpu.
The partial CPU adapter remains local and is excluded from Git. **Do not claim measured RoBERTa performance until
models/roberta/metrics.json exists and the run is complete.** The previously published 0.733 macro F1
belongs to the older TF-IDF model, not RoBERTa. See docs/PROJECT_STATUS.md.

## Install
Python 3.11-3.12 recommended. CPU, Apple MPS and CUDA are detected automatically.
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-roberta.txt
python pipeline.py download
python roberta_model.py train
```
The initial run downloads the pretrained checkpoint. Training defaults: 3 epochs, batch size 8,
gradient accumulation 4, maximum sequence length 96, learning rate 2e-4, seed 42.
LoRA rank 8 is applied to attention query/value projections; the classifier is also trained.
The pretrained base is frozen. This is parameter-efficient fine-tuning, not full-weight fine-tuning.

The CSV dataset is deduplicated using normalised text, conflicting duplicate labels are removed,
and unique posts are stratified into 70% train, 15% validation, 15% test. The checkpoint with the
best validation macro F1 is selected. Test evaluation happens after selection. A TF-IDF comparator
is fitted to the exact same split. Original case and punctuation are preserved for RoBERTa.

```bash
python roberta_model.py predict --text "Everyone deserves equal treatment."
```
Prediction requires the trained adapter under models/roberta/adapter and its sibling training.json,
which records the pinned base-model revision. The large original base weights are downloaded from
Hugging Face; only the small trained adapters need to be shared on GitHub.

## GPU notebook
notebooks/roberta_experiment.ipynb contains a reproducible notebook workflow for Colab or another
GPU environment. The notebook includes dataset audit, training, evaluation and limitations.

## How this relates to the research question
The source labels are **hate_speech, offensive_language and neither**. They do not identify the
four discrimination dimensions or the five rhetorical strategies. Classification results alone
cannot answer which rhetorical structures racism, sexism, homophobia and transphobia share.

For that part of the dissertation, independently annotate an approved corpus using
[the annotation guide](docs/ANNOTATION_GUIDE.md) and [experiment protocol](docs/EXPERIMENT_PROTOCOL.md).
The rhetoric package supports blind annotation sheets, optional structured generative-LLM annotation,
human-reference evaluation, agreement, and group-bootstrap comparisons. This supplementary pipeline
is separate from RoBERTa; no paid API service is needed to train or run the RoBERTa classifier.
See docs/RHETORIC_WORKFLOW.md for commands. No human labels or rhetorical findings are fabricated.

## Validation
```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
Tests cover duplicate leakage, split reproducibility, adapter persistence, evidence grounding,
refusal/abstention handling, resumability and human-reference evaluation using explicit software fixtures.
A passing fixture test is not model performance evidence.

## Research and writing
- docs/ROBERTA_MODEL_CARD.md: model scope, training and limitations.
- docs/DATA_AUDIT.md: candidate data for future rhetorical comparison.
- dissertation/working_draft.md: introduction and proposed methods, with empirical chapters pending.
- models/baseline/metrics.json: older TF-IDF evaluation on a different split; not a direct comparison.

## References
- RoBERTa documentation: https://huggingface.co/docs/transformers/model_doc/roberta
- Base checkpoint: https://huggingface.co/FacebookAI/roberta-base
- Selected dataset: https://huggingface.co/datasets/tdavidson/hate_speech_offensive
- Davidson et al. (2017): https://arxiv.org/abs/1703.04009
- Almagro et al. (2026): https://doi.org/10.3389/fcomm.2026.1743196
- Matamoros-Fernández and Farkas (2021): https://doi.org/10.1177/1527476420982230

## Results report
After a run completes:
```bash
python -m pip install -r requirements-report.txt
python make_report.py --run models/roberta_gpu
```
This generates aggregate result tables, a confusion matrix and a same-split per-class F1 comparison.
A missing final metrics file stops report generation rather than filling in hypothetical results.
