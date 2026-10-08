"""Descriptive comparisons of HUMAN annotations; no causal or population claims."""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .core import STRATEGIES, TARGETS
from .evaluate import human_rows

def compare(path,replicates=1000):
    if replicates<100:raise ValueError("Use at least 100 bootstrap replicates")
    rows=human_rows(path)
    if any(not r.get("group_id") for r in rows):raise ValueError("group_id required for clustered uncertainty")
    groups=defaultdict(list)
    for r in rows:groups[r["group_id"]].append(r)
    keys=sorted(groups);rng=np.random.default_rng(42)
    def rate(subset,target,strategy):
        eligible=[r for r in subset if target in r["targets"].split(";") and r[strategy] in ("0","1")]
        return sum(int(r[strategy]) for r in eligible)/len(eligible) if eligible else None
    draws=[]
    if len(keys)>1:
        for _ in range(replicates):
            sample=[r for k in rng.choice(keys,len(keys),replace=True) for r in groups[k]]
            draws.append({(t,s):rate(sample,t,s) for t in TARGETS for s in STRATEGIES})
    out={"posts":len(rows),"groups":len(keys),"origin":"human","bootstrap_replicates":len(draws),"by_target":{}}
    for t in TARGETS:
        subset=[r for r in rows if t in r["targets"].split(";")];out["by_target"][t]={"posts":len(subset),"strategies":{}}
        for s in STRATEGIES:
            valid=[d[t,s] for d in draws if d[t,s] is not None]
            eligible=sum(r[s] in ("0","1") for r in subset)
            out["by_target"][t]["strategies"][s]={"eligible":eligible,"proportion":rate(rows,t,s),
                "cluster_bootstrap_95_interval":list(map(float,np.quantile(valid,[.025,.975]))) if valid else None}
    out["cooccurrence"]={t:{f"{a}+{b}":sum(r[a]=="1" and r[b]=="1" for r in rows if t in r["targets"].split(";"))
        for i,a in enumerate(STRATEGIES) for b in STRATEGIES[i+1:]} for t in TARGETS}
    out["interpretation"]="Sample-conditional descriptions; overlapping target groups are retained. Intervals bootstrap group IDs, not independent posts. Sparse groups and sampling bias limit inference."
    return out
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--gold",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(compare(a.gold),indent=2))
