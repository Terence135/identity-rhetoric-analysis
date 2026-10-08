import csv,json
import pytest
from rhetoric.core import STRATEGIES,digest,validate,write_jsonl,read_jsonl
from rhetoric.prepare import prepare
from rhetoric.llm import annotate,parse_response,payload
from rhetoric.evaluate import evaluate,agreement
from rhetoric.compare import compare

def result(positive=False):
    return {'stance':'endorsed_hostility' if positive else 'neutral','targets':['sexism'] if positive else [],'strategies':{s:{'label':'present' if positive and s=='essentialisation' else 'absent','evidence':['Text'] if positive and s=='essentialisation' else []} for s in STRATEGIES}}
def human(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,['post_id','text','source','group_id','split','targets','stance',*STRATEGIES,'notes']);w.writeheader();w.writerows(rows)
def hrow(id='a',positive='0',group='g'):
    return {'post_id':id,'text':'Text','source':'forum','group_id':group,'split':'test','targets':'sexism','stance':'endorsed_hostility' if positive=='1' else 'neutral',**{s:positive for s in STRATEGIES}}
def test_evidence_validation_rejects_hallucinated_span():
    r=result(True);r['strategies']['essentialisation']['evidence']=['Invented']
    with pytest.raises(ValueError,match='exact'):validate(r,'Text')
def test_positive_counter_speech_rejected():
    r=result(True);r['stance']='counter_speech'
    with pytest.raises(ValueError,match='endorsed'):validate(r,'Text')
def test_duplicate_components_are_split_together(tmp_path):
    p=tmp_path/'input.csv'
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['post_id','text','source','group_id','target_identities']);w.writerows([['1','Same text','forum','A','sexism'],['2','Same text','forum','B','transphobia'],['3','Another text','forum','B','']])
    audit=prepare(p,tmp_path/'prepared');rows=read_jsonl(tmp_path/'prepared/corpus.jsonl')
    assert len(rows)==2 and len({r['group_id'] for r in rows})==1 and len({r['split'] for r in rows})==1
    assert audit['candidate_target_counts']['transphobia']==1
    assert 'candidate_targets' not in (tmp_path/'prepared/rater_a.csv').read_text()
def test_dry_run_never_makes_api_calls(tmp_path,monkeypatch):
    write_jsonl(tmp_path/'corpus.jsonl',[{'post_id':'a','text':'Text','group_id':'g','split':'test'}])
    monkeypatch.setattr('rhetoric.llm.call_openai',lambda *a:pytest.fail('No network expected'))
    out=annotate(tmp_path/'corpus.jsonl',tmp_path/'out.jsonl','specified-model',dry_run=True)
    assert out['live_inference'] is False and out['pending_requests']==1
def test_few_shot_cannot_leak_test_groups():
    row={'post_id':'a','text':'Text','group_id':'g','split':'test'}
    example={'text':'Different','group_id':'g','origin':'human','split':'train','result':result()}
    with pytest.raises(ValueError,match='leakage'):payload(row,'model',[example])
def test_refusals_and_incomplete_outputs_not_negatives():
    with pytest.raises(ValueError,match='refusal'):parse_response({'status':'completed','output':[{'type':'message','content':[{'type':'refusal'}]}]},'Text')
    with pytest.raises(ValueError,match='Incomplete'):parse_response({'status':'incomplete'},'Text')
def test_resumption_skips_success_and_records_provenance(tmp_path,monkeypatch):
    write_jsonl(tmp_path/'corpus.jsonl',[{'post_id':'a','text':'Text','group_id':'g','split':'test'}]);monkeypatch.setenv('OPENAI_API_KEY','test-not-a-real-key');calls=[]
    def fake(body,key):
        calls.append(body)
        return {'status':'completed','id':'test-fixture','model':'test-model','usage':{},'output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(result())}]}]}
    monkeypatch.setattr('rhetoric.llm.call_openai',fake);args=(tmp_path/'corpus.jsonl',tmp_path/'out.jsonl','test-model')
    assert annotate(*args)['completed_requests']==1 and annotate(*args)['completed_requests']==0
    assert len(calls)==1 and calls[0]['store'] is False
    assert read_jsonl(tmp_path/'out.jsonl')[0]['origin']=='llm'
def test_missing_and_abstained_predictions_report_coverage(tmp_path):
    human(tmp_path/'gold.csv',[hrow('a'),hrow('b')]);rr=result();rr['strategies']['essentialisation']['label']='uncertain'
    write_jsonl(tmp_path/'pred.jsonl',[{'post_id':'a','origin':'llm','split':'test','text_hash':digest('Text'),'config_hash':'fixture','result':rr}])
    metrics=evaluate(tmp_path/'gold.csv',tmp_path/'pred.jsonl')
    assert metrics['per_strategy']['essentialisation']['coverage']==0 and 'f1' not in metrics['per_strategy']['essentialisation']
    assert metrics['per_strategy']['dehumanisation']['coverage']==.5
def test_agreement_and_bootstrap_on_explicit_software_fixtures(tmp_path):
    rows=[hrow('a','0','g1'),hrow('b','1','g2')];human(tmp_path/'a.csv',rows);human(tmp_path/'b.csv',rows)
    assert agreement(tmp_path/'a.csv',tmp_path/'b.csv')['dehumanisation']['kappa']==1
    metrics=compare(tmp_path/'a.csv',replicates=100)
    assert metrics['by_target']['sexism']['strategies']['dehumanisation']['proportion']==.5 and metrics['by_target']['transphobia']['posts']==0
