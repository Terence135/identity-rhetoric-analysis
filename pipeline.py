"""Reproducible baseline and supervised rhetoric experiments."""
import argparse, csv, hashlib, html, json, re
from pathlib import Path
from urllib.request import urlopen
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier
URL = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
LABELS = ["hate_speech", "offensive_language", "neither"]
STRATEGIES = ["dehumanisation", "threat_framing", "moral_condemnation", "essentialisation", "exclusion_hierarchy"]
def clean(text):
    text = html.unescape(text).lower()
    text = re.sub(r"https?://\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    return " ".join(text.split())
def dataset(path, rhetoric=False):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"text", "group_id", *STRATEGIES} if rhetoric else {"tweet", "class"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV requires columns: {sorted(required)}")
    buckets = {}
    for row in rows:
        text = clean(row["text" if rhetoric else "tweet"])
        if not text: continue
        if rhetoric:
            if any(row[k] not in ("0", "1") for k in STRATEGIES):
                raise ValueError("Rhetoric labels must be 0 or 1; adjudicate uncertain rows first")
            if not row["group_id"].strip(): raise ValueError("group_id is required")
            label = tuple(int(row[k]) for k in STRATEGIES)
        else:
            label = int(row["class"])
            if label not in (0,1,2): raise ValueError("Invalid class")
        buckets.setdefault(text, []).append((label, row.get("group_id", text)))
    # Conflicting duplicate annotations are excluded; do not choose one arbitrarily.
    records = [(text, entries[0][0], entries[0][1]) for text,entries in buckets.items()
               if len({e[0] for e in entries}) == 1]
    if not records: raise ValueError("No usable rows")
    return records, len(rows)
def make_model(rhetoric=False):
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1500, random_state=42)
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=.98, max_features=50000, sublinear_tf=True)),
                     ("classifier", OneVsRestClassifier(clf) if rhetoric else clf)])
def train(path, output, rhetoric=False):
    records, raw_count = dataset(path, rhetoric)
    x = np.array([r[0] for r in records]); y = np.array([r[1] for r in records])
    if rhetoric:
        from sklearn.model_selection import GroupShuffleSplit
        groups = np.array([r[2] for r in records])
        # Repeated posts can join threads; require unique cleaned texts already deduplicated.
        tr,te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(x,y,groups))
        if np.any(y[tr].sum(axis=0)==0) or np.any(y[tr].sum(axis=0)==len(tr)):
            raise ValueError("Each strategy needs positive and negative training examples; collect more annotations")
        names=STRATEGIES
    else:
        tr,te = train_test_split(np.arange(len(x)),test_size=.2,stratify=y,random_state=42)
        names=LABELS
    assert not set(x[tr]) & set(x[te])
    model=make_model(rhetoric); model.fit(x[tr],y[tr]); pred=model.predict(x[te])
    report=classification_report(y[te],pred,target_names=names,output_dict=True,zero_division=0)
    report.update({"raw_rows":raw_count,"usable_unique_rows":len(x),"train_rows":len(tr),"test_rows":len(te),
        "seed":42,"dataset_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "split":"group-held-out" if rhetoric else "stratified random; exact cleaned duplicates removed",
        "task":"rhetorical_strategies" if rhetoric else "hate_offensive_neither"})
    if not rhetoric:
        dummy=DummyClassifier(strategy="most_frequent").fit(x[tr].reshape(-1,1),y[tr])
        report["dummy_macro_f1"]=f1_score(y[te],dummy.predict(x[te].reshape(-1,1)),average="macro")
        report["confusion_matrix"]=confusion_matrix(y[te],pred,labels=[0,1,2]).tolist()
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    joblib.dump({"pipeline":model,"labels":names,"rhetoric":rhetoric},output/"model.joblib",compress=3)
    (output/"metrics.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"macro_f1":report["macro avg"]["f1-score"],"train_rows":len(tr),"test_rows":len(te)},indent=2))
def predict(path,text):
    if not text.strip(): raise ValueError("Provide non-empty text")
    # Only load trusted artifacts: joblib deserialization can execute code.
    bundle=joblib.load(path); model=bundle["pipeline"]; text=clean(text)
    scores=model.predict_proba([text])[0]
    result={"task":"rhetorical_strategies" if bundle["rhetoric"] else "hate_offensive_neither",
            "scores":dict(zip(bundle["labels"],map(float,scores)))}
    if not bundle["rhetoric"]: result["prediction"]=bundle["labels"][int(model.predict([text])[0])]
    return result
def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="command",required=True)
    d=sub.add_parser("download"); d.add_argument("--output",default="data/raw/davidson.csv")
    t=sub.add_parser("train"); t.add_argument("--data",default="data/raw/davidson.csv"); t.add_argument("--output",default="models/baseline"); t.add_argument("--rhetoric",action="store_true")
    p=sub.add_parser("predict"); p.add_argument("--model",default="models/baseline/model.joblib"); p.add_argument("--text",required=True)
    a=ap.parse_args()
    if a.command=="download":
        target=Path(a.output); target.parent.mkdir(parents=True,exist_ok=True)
        with urlopen(URL,timeout=60) as response: target.write_bytes(response.read())
        print(f"Downloaded to {target}. Dataset contains harmful language; raw data is gitignored.")
    elif a.command=="train": train(a.data,a.output,a.rhetoric)
    else: print(json.dumps(predict(a.model,a.text),indent=2))
if __name__=="__main__": main()
