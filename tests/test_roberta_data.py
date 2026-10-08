import csv
import pytest
pytest.importorskip('torch')
pytest.importorskip('peft')
from roberta_model import load_records,split_records

def test_roberta_dedup_keeps_case_but_removes_conflicts(tmp_path):
    p=tmp_path/'data.csv'
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['tweet','class']);w.writerows([['Hello @a',0],['hello @b',1],['Preserve CASE!',2]])
    rows,meta=load_records(p)
    assert rows==[('Preserve CASE!',2)] and meta['raw_rows']==3

def test_roberta_splits_disjoint_and_reproducible():
    rows=[(f'Example {i}',i%3) for i in range(120)];splits=split_records(rows,42)
    assert [list(x) for x in splits]==[list(x) for x in split_records(rows,42)]
    assert sum(map(len,splits))==120
    assert all(set(splits[i]).isdisjoint(splits[j]) for i in range(3) for j in range(i+1,3))

def test_lora_selected_checkpoint_preserves_predictions(tmp_path):
    import torch
    from transformers import RobertaConfig,RobertaForSequenceClassification
    from peft import LoraConfig,TaskType,get_peft_model
    config=RobertaConfig(vocab_size=40,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=24,num_labels=3)
    model=get_peft_model(RobertaForSequenceClassification(config),LoraConfig(task_type=TaskType.SEQ_CLS,r=2,lora_alpha=4,target_modules=['query','value'],modules_to_save=['classifier']))
    inputs={'input_ids':torch.tensor([[0,5,6,2]]),'attention_mask':torch.ones(1,4,dtype=torch.long)}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    model.train();loss=model(**inputs,labels=torch.tensor([1])).loss;loss.backward();optimizer.step()
    model.eval()
    with torch.no_grad():before=model(**inputs).logits
    model.save_pretrained(tmp_path/'adapter',safe_serialization=True)
    model.load_adapter(tmp_path/'adapter',adapter_name='selected');model.set_adapter('selected');model.eval()
    with torch.no_grad():after=model(**inputs).logits
    torch.testing.assert_close(before,after)
