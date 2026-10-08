"""Compare held-out LLM results against independent, adjudicated human labels."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from .core import STRATEGIES, TARGETS, STANCES, digest, read_jsonl, validate

def human_rows(path):
    with open(path,encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    if not rows or len({r["post_id"] for r in rows})!=len(rows):raise ValueError("Human rows require unique IDs")
    for row in rows:
        if any(row.get(k) not in ("0","1","uncertain") for k in STRATEGIES):raise ValueError("Complete all human labels with 0, 1 or uncertain")
        ts=set(filter(None,row.get("targets","").split(";")))
        if not ts.issubset(TARGETS):raise ValueError("Unknown target")
        if row.get("stance") not in STANCES:raise ValueError("Complete stance labels")
        if any(row[k]=="1" for k in STRATEGIES) and (row["stance"]!="endorsed_hostility" or not ts):raise ValueError("Positive human strategies require endorsed identity hostility")
    return rows

def evaluate(gold,predictions):
    rows=[r for r in human_rows(gold) if r.get("split")=="test"]
    if not rows: raise ValueError("No held-out test rows")
    preds=read_jsonl(predictions)
    if len({r["post_id"] for r in preds})!=len(preds):raise ValueError("Duplicate predictions")
    pmap={r["post_id"]:r for r in preds}
    configs={r.get("config_hash") for r in preds}
    if len(configs)!=1:raise ValueError("Mixed prompt configurations; evaluate separately")
    for row in rows:
        if row["post_id"] not in pmap:continue
        pr=pmap[row["post_id"]]
        if pr.get("origin")!="llm" or pr.get("text_hash")!=digest(row["text"]) or pr.get("split")!="test":raise ValueError("Prediction provenance/input mismatch")
        validate(pr["result"],row["text"])
    def scores(subset):
        result={}
        for name in STRATEGIES:
            eligible=[r for r in subset if r[name] in ("0","1")]
            covered=[r for r in eligible if r["post_id"] in pmap and pmap[r["post_id"]]["result"]["strategies"][name]["label"]!="uncertain"]
            item={"eligible_gold":len(eligible),"covered":len(covered),"coverage":len(covered)/len(eligible) if eligible else None,
                  "missing_predictions":sum(r["post_id"] not in pmap for r in eligible)}
            if covered:
                y=[int(r[name]) for r in covered]; pred=[int(pmap[r["post_id"]]["result"]["strategies"][name]["label"]=="present") for r in covered]
                precision,recall,f1,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
                item.update(precision=float(precision),recall=float(recall),f1=float(f1),positive_gold=sum(y))
            result[name]=item
        return result
    return {"test_posts":len(rows),"config_hash":next(iter(configs)),"per_strategy":scores(rows),
        "per_target":{t:scores([r for r in rows if t in r["targets"].split(";")]) for t in TARGETS},
        "interpretation":"Precision/recall/F1 are conditional on non-abstained predictions. Always report coverage; missing/uncertain are not negative labels."}

def agreement(a,b):
    aa=human_rows(a);bb={r["post_id"]:r for r in human_rows(b)}
    if set(bb)!={r["post_id"] for r in aa}:raise ValueError("Rater ID sets must match")
    if any(r["text"]!=bb[r["post_id"]]["text"] for r in aa):raise ValueError("Raters annotated different text")
    out={}
    for name in STRATEGIES:
        paired=[(r[name],bb[r["post_id"]][name]) for r in aa if r[name]!="uncertain" and bb[r["post_id"]][name]!="uncertain"]
        if not paired:out[name]={"paired":0};continue
        x,y=zip(*paired);k=float(cohen_kappa_score(x,y)) if len(set(x)|set(y))>1 else None
        out[name]={"paired":len(paired),"excluded_uncertain":len(aa)-len(paired),"raw_agreement":sum(i==j for i,j in paired)/len(paired),"kappa":k if k is None or np.isfinite(k) else None}
    return out
if __name__=="__main__":
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    e=sub.add_parser("model");e.add_argument("--gold",required=True);e.add_argument("--predictions",required=True)
    a=sub.add_parser("agreement");a.add_argument("--rater-a",required=True);a.add_argument("--rater-b",required=True)
    p.add_argument("--output",required=True);args=p.parse_args()
    result=evaluate(args.gold,args.predictions) if args.command=="model" else agreement(args.rater_a,args.rater_b)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False))
