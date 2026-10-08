"""Prepare an approved corpus; join duplicate/group components before splitting."""
import argparse, csv, json
from collections import Counter
from pathlib import Path
from .core import TARGETS, STRATEGIES, digest, normalise, redact, write_jsonl

def prepare(path,output):
    with open(path,encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    required={"post_id","text","source","group_id","target_identities"}
    if not rows or not required.issubset(rows[0]): raise ValueError(f"Require {sorted(required)}")
    parent={}
    def root(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=root(parent[x])
        return parent[x]
    def union(a,b):
        aa,bb=root(a),root(b)
        if aa!=bb: parent[max(aa,bb)]=min(aa,bb)
    seen={}; kept={}; ids=set()
    for row in rows:
        if not all(row[k].strip() for k in ("post_id","text","source","group_id")): raise ValueError("Blank required field")
        if row["post_id"] in ids: raise ValueError("Duplicate post_id; aggregate repeated annotators first")
        ids.add(row["post_id"])
        targets=set(filter(None,row["target_identities"].split(";")))
        if not targets.issubset(TARGETS): raise ValueError("Unknown candidate target")
        text=redact(row["text"]); key=normalise(text)
        group=digest([row["source"],row["group_id"]])
        root(group)
        if key in seen: union(group,seen[key])
        seen[key]=group
        if key not in kept:
            kept[key]={"post_id":digest([row["source"],row["post_id"]])[:24],"text":text,"source":row["source"],"group_id":group,"candidate_targets":sorted(targets)}
        else: kept[key]["candidate_targets"]=sorted(set(kept[key]["candidate_targets"])|targets)
    result=[]
    for row in kept.values():
        row["group_id"]=root(row["group_id"])
        bucket=int(digest([42,row["group_id"]])[:8],16)%10
        row["split"]="train" if bucket<6 else "dev" if bucket<8 else "test"
        result.append(row)
    result.sort(key=lambda x:x["post_id"])
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write_jsonl(out/"corpus.jsonl",result)
    audit={"input_sha256":__import__('hashlib').sha256(Path(path).read_bytes()).hexdigest(),"input_rows":len(rows),"unique_redacted_posts":len(result),
        "groups":len({x["group_id"] for x in result}),"split_counts":dict(Counter(x["split"] for x in result)),
        "source_counts":dict(Counter(x["source"] for x in result)),
        "candidate_target_counts":{t:sum(t in x["candidate_targets"] for x in result) for t in TARGETS},
        "note":"Candidate targets are sampling metadata, not confirmed discrimination labels. No prevalence claims."}
    (out/"audit.json").write_text(json.dumps(audit,indent=2))
    for rater in ("rater_a","rater_b"):
        with (out/f"{rater}.csv").open("w",newline="",encoding="utf-8") as f:
            fields=["post_id","text","source","group_id","split","targets","stance",*STRATEGIES,"notes"]
            w=csv.DictWriter(f,fields);w.writeheader()
            for row in result: w.writerow({k:row[k] for k in ["post_id","text","source","group_id","split"]})
    return audit
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",default="data/processed")
    a=p.parse_args();print(json.dumps(prepare(a.input,a.output),indent=2))
