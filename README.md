# Identity rhetoric analysis
**Somto Terence Ndu | Master's project DAIM_A_105**

Research question: which rhetorical strategies are shared across racism, sexism, homophobia and
transphobia, and how do they vary across targets?

## What is implemented
- A genuinely trained, reproducible hate/offensive/neither baseline using the recommended Davidson dataset.
- Saved model, measured held-out metrics, majority baseline and command-line predictions.
- A separate multi-label rhetorical training mode and annotation guide for the actual research question.

**The trained baseline is not a validated rhetorical model.** Rhetorical training requires manually
labelled posts. No synthetic results or invented rhetorical annotations are used.

## Setup and run (Python 3.11-3.12)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python pipeline.py predict --text "Everyone deserves equal treatment."
pytest -q
```
To reproduce training (downloads harmful-language research data locally):
```bash
python pipeline.py download
python pipeline.py train
```
The source URL tracks upstream master; metrics record the exact downloaded SHA-256. If upstream
changes, compare hashes before comparing experiments. Raw data is excluded from version control.

## Rhetorical experiment
Copy data/annotations/template.csv to data/annotations/labelled.csv and complete genuine annotations
using docs/ANNOTATION_GUIDE.md. Reserve independent author/conversation groups in group_id.
```bash
python pipeline.py train --rhetoric --data data/annotations/labelled.csv --output models/rhetoric
python pipeline.py predict --model models/rhetoric/model.joblib --text "Text to analyse"
```
Five independent rhetorical labels are supported; there must be positive and negative training
examples for each. This command has not been run on real rhetorical data yet. Compare strategies
across target_identities only after annotation and evaluation; do not use toxicity labels as substitutes.

## Research workflow
1. Confirm source access and university ethics requirements.
2. Audit coverage of all four discrimination categories and comparable sampling contexts.
3. Pilot the codebook; measure human annotation agreement and freeze held-out data.
4. Train the rhetorical baseline; evaluate per-label and per-target errors.
5. Add and evaluate LLM annotation using the same reserved evaluation set.
6. Interpret shared patterns using quantitative comparison and close reading.

See docs/MODEL_CARD.md for limitations and models/baseline/metrics.json for measured performance.
The current dataset does not establish balanced coverage across the four discrimination types.

## References
- Davidson et al. (2017): https://arxiv.org/abs/1703.04009
- Almagro et al. (2026): https://doi.org/10.3389/fcomm.2026.1743196
- Matamoros-Fernández and Farkas (2021): https://doi.org/10.1177/1527476420982230

## Measured baseline results

| Measure | Result |
|---|---:|
| Training posts | 19,635 |
| Held-out posts | 4,909 |
| Macro F1 | 0.733 |
| Accuracy | 0.867 |
| Majority baseline macro F1 | 0.291 |

These are random-split dataset results, not rhetorical annotation performance.
