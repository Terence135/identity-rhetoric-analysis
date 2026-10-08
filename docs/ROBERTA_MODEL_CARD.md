# RoBERTa model card
Somto Terence Ndu | DAIM_A_105

## Intended use and task
Academic three-class analysis of the selected Davidson English tweet dataset: hate_speech,
offensive_language, neither. The pretrained base is FacebookAI/roberta-base (RoBERTa encoder).
Its trained LoRA adapters plus classifier constitute the downstream model. This is neither a
conversation model nor a validated detector of the five proposed rhetorical strategies.

## Training design
- Base revision resolved and stored in training.json; safetensors loading only.
- LoRA r=8, alpha=16, dropout=.1, query/value attention projections; classifier trainable.
- AdamW, learning rate 2e-4, weight decay .01, 10% warmup then linear decay.
- Batch 8, accumulation 4, 3 epochs, length 96, seed 42, weighted cross-entropy from train counts.
- Exact normalised duplicates removed; conflicting-label duplicates excluded.
- Stratified 70/15/15; checkpoint selected on validation macro F1; held-out test used afterward.
- Same-split TF-IDF comparator and majority-class baseline reported in final metrics.
- Token truncation may omit context. No hyperparameter search or calibration is claimed.
- Training is nondeterministic on some backends despite seed control; record actual software versions.

## Artifacts and evaluation
training.json and split_hashes.json document provenance and split membership without publishing posts.
history.json records validation scores by epoch. adapter/ contains only the selected trained adapter,
classification head and tokenizer files. metrics.json is produced only when final evaluation succeeds.
Do not describe the adapter as complete or quote test scores before that file is present.
The earlier models/baseline metrics use a different split and are not a controlled RoBERTa comparison.

## Limitations
Dataset annotator judgments are a reference, not objective truth. Historical Twitter selection,
class imbalance, dialect bias and missing context limit external validity. Source metadata is
insufficient for reliable author/thread separation; near duplicates can remain after exact deduplication.
There is no balanced four-way target annotation and no direct rhetoric labels in the selected dataset.
No subgroup fairness, causal or general online-prevalence claims are supported.
Model outputs should not determine sanctions or decisions about individuals.

## Attribution
Base: https://huggingface.co/FacebookAI/roberta-base (check base model card/license).
Data: Davidson et al. (2017), ICWSM 11(1), 512-515. https://arxiv.org/abs/1703.04009
Dataset: https://huggingface.co/datasets/tdavidson/hate_speech_offensive
The repository grants no additional rights over source data or pretrained model weights.
