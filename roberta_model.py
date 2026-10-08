"""RoBERTa + LoRA supervised classification of the Davidson dataset.

The base encoder is pretrained; adapters and a classification head are fine-tuned.
This is NOT a generative LLM or a rhetorical classifier.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from pipeline import clean, LABELS, make_model


def load_records(path):
    """Keep RoBERTa's case/punctuation; normalise for duplicate grouping only."""
    import html, re
    buckets={}
    with open(path,encoding='utf-8',newline='') as f:
        raw=list(csv.DictReader(f))
    for row in raw:
        label=int(row['class'])
        if label not in (0,1,2):raise ValueError('Unexpected class')
        text=html.unescape(row['tweet'])
        text=re.sub(r'https?://\S+','[URL]',text)
        text=re.sub(r'(?<!\w)@\w+','[USER]',text)
        text=' '.join(text.split())
        if not text:continue
        key=clean(text)
        buckets.setdefault(key,[]).append((text,label))
    # Drop label conflicts rather than assigning arbitrary majority labels.
    rows=[v[0] for v in buckets.values() if len({x[1] for x in v})==1]
    return rows, {'raw_rows':len(raw),'usable_unique_rows':len(rows),
                  'dataset_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def split_records(rows,seed):
    labels=[r[1] for r in rows]; ids=np.arange(len(rows))
    train,hold=train_test_split(ids,test_size=.3,stratify=labels,random_state=seed)
    dev,test=train_test_split(hold,test_size=.5,stratify=np.array(labels)[hold],random_state=seed)
    for a,b in [(train,dev),(train,test),(dev,test)]:
        if set(clean(rows[i][0]) for i in a)&set(clean(rows[i][0]) for i in b):raise ValueError('Duplicate leakage')
    return train,dev,test


class EncodedDataset(Dataset):
    def __init__(self,rows,indices,tokenizer,max_length):
        self.encoding=tokenizer([rows[i][0] for i in indices],truncation=True,padding='max_length',max_length=max_length)
        self.labels=[rows[i][1] for i in indices]
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):
        return {**{k:torch.tensor(v[i],dtype=torch.long) for k,v in self.encoding.items()},
                'labels':torch.tensor(self.labels[i],dtype=torch.long)}


def choose_device():
    if torch.cuda.is_available():return torch.device('cuda')
    if torch.backends.mps.is_available():return torch.device('mps')
    return torch.device('cpu')


def evaluate(model,loader,device):
    model.eval(); truth=[];pred=[]
    with torch.inference_mode():
        for batch in loader:
            truth.extend(batch.pop('labels').tolist())
            logits=model(**{k:v.to(device) for k,v in batch.items()}).logits
            pred.extend(logits.argmax(-1).cpu().tolist())
    return truth,pred


def train(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'training.json').exists():raise ValueError('Output already contains a run; choose a new output directory')
    rows,provenance=load_records(args.data);tr,dev,te=split_records(rows,args.seed)
    device=choose_device(); print(f'Device: {device}; train={len(tr)}, dev={len(dev)}, test={len(te)}',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(args.base_model,revision=args.revision)
    base=AutoModelForSequenceClassification.from_pretrained(args.base_model,revision=args.revision,num_labels=3,
        id2label=dict(enumerate(LABELS)),label2id={v:i for i,v in enumerate(LABELS)},attn_implementation='eager',use_safetensors=True)
    revision=base.config._commit_hash
    model=get_peft_model(base,LoraConfig(task_type=TaskType.SEQ_CLS,r=8,lora_alpha=16,lora_dropout=.1,
                        target_modules=['query','value'],modules_to_save=['classifier']))
    model.to(device);model.print_trainable_parameters()
    loaders=[]
    for ids,shuffle in [(tr,True),(dev,False),(te,False)]:
        ds=EncodedDataset(rows,ids,tokenizer,args.max_length)
        loaders.append(DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle,num_workers=0))
    train_loader,dev_loader,test_loader=loaders
    counts=np.bincount([rows[i][1] for i in tr],minlength=3)
    weights=torch.tensor(len(tr)/(3*counts),dtype=torch.float32,device=device)
    loss_fn=torch.nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=.01)
    total_steps=math.ceil(len(train_loader)/args.accumulation)*args.epochs
    warmup=max(1,int(.1*total_steps))
    def lr_factor(step):
        return (step+1)/warmup if step<warmup else max(0.,(total_steps-step)/max(1,total_steps-warmup))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    config={**vars(args),'resolved_base_revision':revision,'device':str(device),'train_rows':len(tr),'dev_rows':len(dev),'test_rows':len(te),
        'class_weights':weights.cpu().tolist(),'split':'stratified 70/15/15 after exact cleaned deduplication',**provenance,
        'method':'LoRA rank 8 on attention query/value plus trained classifier; base weights frozen',
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (out/'training.json').write_text(json.dumps(config,indent=2))
    (out/'split_hashes.json').write_text(json.dumps({name:[hashlib.sha256(clean(rows[i][0]).encode()).hexdigest() for i in ids]
        for name,ids in [('train',tr),('dev',dev),('test',te)]},indent=2))
    best=-1.;history=[];start=time.monotonic();step=0
    try:
        for epoch in range(args.epochs):
            model.train();optimizer.zero_grad();total_loss=0
            for idx,batch in enumerate(train_loader):
                labels=batch.pop('labels').to(device)
                logits=model(**{k:v.to(device) for k,v in batch.items()}).logits
                loss=loss_fn(logits,labels)
                # Mean over the actual microbatch count in the final accumulation group.
                group_size=min(args.accumulation,len(train_loader)-(idx//args.accumulation)*args.accumulation)
                (loss/group_size).backward();total_loss+=loss.item()
                if (idx+1)%args.accumulation==0 or idx+1==len(train_loader):
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                    optimizer.step();scheduler.step();optimizer.zero_grad();step+=1
                if (idx+1)%25==0:
                    print(json.dumps({'epoch':epoch+1,'batch':idx+1,'batches':len(train_loader),'loss':total_loss/(idx+1),'elapsed_minutes':round((time.monotonic()-start)/60,2)}),flush=True)
            y,pred=evaluate(model,dev_loader,device);score=f1_score(y,pred,average='macro')
            item={'epoch':epoch+1,'training_loss':total_loss/len(train_loader),'dev_macro_f1':float(score)}
            history.append(item);print(json.dumps(item),flush=True)
            if score>best:
                best=score;model.save_pretrained(out/'adapter',safe_serialization=True);tokenizer.save_pretrained(out/'adapter')
            (out/'history.json').write_text(json.dumps(history,indent=2))
        # Reload selected adapter; the held-out test is used once after selection.
        model.load_adapter(out/'adapter',adapter_name='selected');model.set_adapter('selected')
        y,pred=evaluate(model,test_loader,device)
        report=classification_report(y,pred,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0)
        report['confusion_matrix']=confusion_matrix(y,pred,labels=[0,1,2]).tolist()
        # Fit a task-matched TF-IDF baseline on the IDENTICAL split.
        baseline=make_model();baseline.fit([clean(rows[i][0]) for i in tr],[rows[i][1] for i in tr])
        bpred=baseline.predict([clean(rows[i][0]) for i in te])
        report['tfidf_same_split']=classification_report(y,bpred,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0)
        majority=int(np.argmax(counts))
        report['majority_macro_f1']=float(f1_score(y,[majority]*len(y),average='macro'))
        report.update(best_dev_macro_f1=best,selected_epoch=max(history,key=lambda x:x['dev_macro_f1'])['epoch'],
                      elapsed_minutes=(time.monotonic()-start)/60,task='hate_offensive_neither',**provenance)
        (out/'metrics.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'status':'complete','test_macro_f1':report['macro avg']['f1-score'],'test_accuracy':report['accuracy']}),flush=True)
    except BaseException:
        # Save a recoverable adapter even if interrupted before the first epoch completes.
        model.save_pretrained(out/'interrupted_adapter',safe_serialization=True);tokenizer.save_pretrained(out/'interrupted_adapter')
        raise


def predict(args):
    if not args.text.strip():raise ValueError('Non-empty text required')
    adapter=Path(args.adapter);cfg=json.loads((adapter/'adapter_config.json').read_text())
    training_path=adapter.parent/'training.json'
    revision=json.loads(training_path.read_text()).get('resolved_base_revision') if training_path.exists() else None
    if not revision:raise ValueError('training.json with pinned base revision is required beside adapter')
    tokenizer=AutoTokenizer.from_pretrained(adapter)
    base=AutoModelForSequenceClassification.from_pretrained(cfg['base_model_name_or_path'],revision=revision,num_labels=3,
        id2label=dict(enumerate(LABELS)),label2id={v:i for i,v in enumerate(LABELS)},use_safetensors=True)
    model=PeftModel.from_pretrained(base,adapter);model.eval()
    maxlen=json.loads(training_path.read_text())['max_length']
    with torch.inference_mode():probs=model(**tokenizer(args.text,return_tensors='pt',truncation=True,max_length=maxlen)).logits.softmax(-1)[0].tolist()
    print(json.dumps({'prediction':LABELS[int(np.argmax(probs))],'scores':dict(zip(LABELS,probs))},indent=2))


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    t=s.add_parser('train');t.add_argument('--data',default='data/raw/davidson.csv');t.add_argument('--output',default='models/roberta')
    t.add_argument('--base-model',default='FacebookAI/roberta-base');t.add_argument('--revision',default='main')
    t.add_argument('--epochs',type=int,default=3);t.add_argument('--batch-size',type=int,default=8);t.add_argument('--accumulation',type=int,default=4)
    t.add_argument('--max-length',type=int,default=96);t.add_argument('--learning-rate',type=float,default=2e-4)
    t.add_argument('--seed',type=int,default=42);t.add_argument('--cpu-threads',type=int,default=4)
    q=s.add_parser('predict');q.add_argument('--adapter',default='models/roberta/adapter');q.add_argument('--text',required=True)
    args=p.parse_args()
    if args.command=='train':
        if min(args.epochs,args.batch_size,args.accumulation,args.max_length,args.cpu_threads)<1:p.error('Training counts must be positive')
        train(args)
    else:predict(args)
if __name__=='__main__':main()
