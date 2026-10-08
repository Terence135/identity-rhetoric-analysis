import csv
import numpy as np
import pytest
from pipeline import clean, dataset, make_model, predict
import joblib

def test_normalised_duplicate_conflicts_are_excluded(tmp_path):
    p=tmp_path/"rows.csv"
    with p.open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["tweet","class"])
        w.writerows([["Hello @alice",0],["hello @bob",1],["Distinct message",2]])
    rows,n=dataset(p)
    assert n==3 and len(rows)==1 and rows[0][1]==2

def test_round_trip_predictions(tmp_path):
    x=["support equal rights today", "support equal rights always", "friendly welcome everyone", "friendly welcome neighbours"]
    y=[0,0,1,1]
    model=make_model(); model.fit(x,y)
    p=tmp_path/"model.joblib"
    joblib.dump({"pipeline":model,"labels":["a","b"],"rhetoric":False},p)
    result=predict(p,x[0]); assert result["prediction"]=="a"
    assert sum(result["scores"].values())==pytest.approx(1)
    with pytest.raises(ValueError): predict(p," ")

def test_rhetoric_rejects_uncertain_labels(tmp_path):
    from pipeline import STRATEGIES
    p=tmp_path/"rows.csv"
    with p.open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["text","group_id",*STRATEGIES]); w.writerow(["sample","thread1","uncertain",0,0,0,0])
    with pytest.raises(ValueError,match="0 or 1"): dataset(p,True)
