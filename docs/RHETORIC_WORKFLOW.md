# Supplementary rhetorical research workflow

RoBERTa training on Davidson is implemented separately in roberta_model.py.
This workflow requires independent human annotations to answer the original rhetoric questions.

## 1. Prepare an approved corpus
Create data/local/corpus.csv using data/corpus_template.csv's columns, then run:
```bash
python -m rhetoric.prepare --input data/local/corpus.csv --output data/processed
```
Inspect audit.json. Two blind sheets, rater_a.csv and rater_b.csv, have blank strategy labels.
Complete targets (semicolon-separated), stance and each strategy (0, 1 or uncertain).
Do not expose raters to model predictions. Resolve discrepancies into a separate adjudicated.csv.

## 2. Assess human agreement
```bash
python -m rhetoric.evaluate --output reports/agreement.json agreement --rater-a data/processed/rater_a.csv --rater-b data/processed/rater_b.csv
```

## 3. Optional generative LLM condition
Select an explicit provider model/snapshot available to your account. Configure OPENAI_API_KEY
privately in your shell. No API key is needed for RoBERTa. This command validates without inference:
```bash
python -m rhetoric.llm --input data/processed/corpus.jsonl --output runs/zero_shot.jsonl --model YOUR_MODEL_SNAPSHOT --dry-run
```
After the posts are approved for external processing, replace --dry-run with
--allow-external-processing and set --limit deliberately (default 10 requests).
Responses use the official structured-output API:
https://developers.openai.com/api/docs/guides/structured-outputs

Few-shot examples use JSONL records with text, group_id, split="train", origin="human", and a
result matching rhetoric.core.schema(). Positive evidence must be exact substrings.
Pass --examples data/local/examples.jsonl and a different output path. Never use test rows or
related group IDs in examples. A missing key, refusal, invalid evidence or API failure stops a run.
Completed rows are resumed only if prompt configuration and text hashes match.

## 4. Evaluate and compare
```bash
python -m rhetoric.evaluate --output reports/zero_shot.json model --gold data/processed/adjudicated.csv --predictions runs/zero_shot.jsonl
python -m rhetoric.compare --gold data/processed/adjudicated.csv --output reports/human_strategy_comparison.json
```
F1 is conditional on non-abstained labels; report coverage and missing responses alongside it.
Comparisons use human labels. Bootstrap intervals reflect connected-group resampling in the
sample, not representativeness of internet discourse. Review aggregate outputs before publishing.
Raw data, human sheets and API outputs are gitignored. No live generative LLM run has occurred.
