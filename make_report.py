"""Generate reviewed-ready aggregate tables/plots after a completed RoBERTa run."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pipeline import LABELS

def report(run,output):
    run=Path(run);output=Path(output)
    if not (run/'metrics.json').exists():raise ValueError('Training has not produced final metrics; no results report generated')
    metrics=json.loads((run/'metrics.json').read_text());training=json.loads((run/'training.json').read_text())
    if metrics.get('task')!='hate_offensive_neither':raise ValueError('Unexpected task')
    output.mkdir(parents=True,exist_ok=True)
    matrix=np.array(metrics['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    image=ax.imshow(matrix,cmap='Blues');fig.colorbar(image,ax=ax)
    ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted label',ylabel='Reference label',title='RoBERTa: held-out confusion matrix')
    ax.tick_params(axis='x',labelrotation=20)
    for i in range(3):
        for j in range(3):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    fig.savefig(output/'confusion_matrix.png',dpi=200);plt.close(fig)
    x=np.arange(3);fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    ax.bar(x-.18,[metrics[k]['f1-score'] for k in LABELS],.36,label='RoBERTa LoRA')
    ax.bar(x+.18,[metrics['tfidf_same_split'][k]['f1-score'] for k in LABELS],.36,label='TF-IDF logistic regression')
    ax.set(xticks=x,xticklabels=LABELS,ylim=(0,1),ylabel='F1 score',title='Same-split held-out comparison');ax.legend()
    fig.savefig(output/'per_class_f1.png',dpi=200);plt.close(fig)
    lines=['# RoBERTa classification results','Somto Terence Ndu | DAIM_A_105','',
           '**Scope:** hate speech, offensive language and neither. These results do not measure rhetorical strategies.','',
           f"Training / validation / test posts: {training['train_rows']:,} / {training['dev_rows']:,} / {training['test_rows']:,}.",
           f"Selected epoch: {metrics['selected_epoch']}; validation macro F1: {metrics['best_dev_macro_f1']:.3f}.",'',
           '| Model | Test macro F1 | Test accuracy |','|---|---:|---:|',
           f"| RoBERTa LoRA | {metrics['macro avg']['f1-score']:.3f} | {metrics['accuracy']:.3f} |",
           f"| TF-IDF, identical split | {metrics['tfidf_same_split']['macro avg']['f1-score']:.3f} | {metrics['tfidf_same_split']['accuracy']:.3f} |",
           f"| Majority class | {metrics['majority_macro_f1']:.3f} | Not recorded |",'',
           '| Class | Precision | Recall | F1 | Test support |','|---|---:|---:|---:|---:|']
    for k in LABELS:
        m=metrics[k];lines.append(f"| {k} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1-score']:.3f} | {m['support']:.0f} |")
    lines += ['', '![Held-out confusion matrix](confusion_matrix.png)','', '![Per-class comparison](per_class_f1.png)','',
        '## Interpretation limits','These are single-seed random-split results on one historical English tweet dataset. Exact duplicates were removed, but near duplicates and author/thread overlap can remain. Scores do not establish performance on new platforms, fairness across identity groups, or shared rhetoric. No significance or causal claim is made.', '',
        f"Base revision: `{training['resolved_base_revision']}`.",f"Dataset SHA-256: `{training['dataset_sha256']}`."]
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return output/'RESULTS.md'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='models/roberta_gpu');p.add_argument('--output',default='reports/roberta_results');a=p.parse_args();print(report(a.run,a.output))
