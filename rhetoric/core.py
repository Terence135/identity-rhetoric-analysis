"""Shared schema and strict local validation. No network calls."""
import csv, hashlib, json, re
from pathlib import Path
STRATEGIES = ("dehumanisation", "threat_framing", "moral_condemnation", "essentialisation", "exclusion_hierarchy")
TARGETS = ("racism", "sexism", "homophobia", "transphobia")
STANCES = ("endorsed_hostility", "quotation", "counter_speech", "neutral", "uncertain")
DEFINITIONS = {
 "dehumanisation": "Represents identity-group members as less than human, objects, animals or contamination; an insult alone is insufficient.",
 "threat_framing": "Constructs an identity group as a danger to safety, culture, institutions or social order; a personal disagreement alone is insufficient.",
 "moral_condemnation": "Attributes moral corruption or inherent wrongdoing to an identity group; criticism of a specific act alone is insufficient.",
 "essentialisation": "Presents negative characteristics as fixed, inherent or universal to an identity group.",
 "exclusion_hierarchy": "Justifies identity-based exclusion, subordinate status or unequal rights or treatment."
}
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def read_jsonl(path):
    with open(path,encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
def write_jsonl(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
def redact(text):
    text=re.sub(r"https?://\S+", "[URL]", text)
    text=re.sub(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", "[EMAIL]", text)
    return re.sub(r"(?<!\w)@[\w]+", "[USER]", text)
def normalise(text): return " ".join(text.casefold().split())
def schema():
    strategy={"type":"object","properties":{"label":{"type":"string","enum":["present","absent","uncertain"]},
              "evidence":{"type":"array","items":{"type":"string"}}},"required":["label","evidence"],"additionalProperties":False}
    props={"stance":{"type":"string","enum":list(STANCES)},
           "targets":{"type":"array","items":{"type":"string","enum":list(TARGETS)}},
           "strategies":{"type":"object","properties":{k:strategy for k in STRATEGIES},"required":list(STRATEGIES),"additionalProperties":False}}
    return {"type":"object","properties":props,"required":list(props),"additionalProperties":False}
def validate(result,text):
    if not isinstance(result,dict) or set(result)!={"stance","targets","strategies"}: raise ValueError("Invalid result fields")
    if result["stance"] not in STANCES: raise ValueError("Invalid stance")
    if not isinstance(result["targets"],list) or any(t not in TARGETS for t in result["targets"]): raise ValueError("Invalid targets")
    if len(set(result["targets"]))!=len(result["targets"]): raise ValueError("Duplicate targets")
    if not isinstance(result["strategies"],dict) or set(result["strategies"])!=set(STRATEGIES): raise ValueError("Missing strategy")
    for name,item in result["strategies"].items():
        if not isinstance(item,dict) or set(item)!={"label","evidence"}: raise ValueError("Invalid strategy fields")
        if item["label"] not in ("present","absent","uncertain"): raise ValueError("Invalid label")
        if not isinstance(item["evidence"],list) or any(not isinstance(s,str) or not s or s not in text for s in item["evidence"]): raise ValueError("Evidence must be exact nonempty substrings")
        if item["label"]=="present":
            if not item["evidence"]: raise ValueError("Positive labels require evidence")
            if result["stance"]!="endorsed_hostility" or not result["targets"]: raise ValueError("Positive strategy requires endorsed identity hostility")
    return result
