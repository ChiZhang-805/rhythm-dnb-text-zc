"""Read frozen partitions without changing the master or exposing answers as inputs."""

from collections import Counter
import math

from .corpus import ROOT,DATABASE,connect,read_json,dumps,sha256
from .chinese_data import examples,model_pair,verify
from .contract import CATEGORIES,METRICS,CONTRACT_ID

DEFAULT_CONFIG=ROOT/'train_config.json'
BASE=ROOT/'models/macbert/base'
MODEL_ROOT=ROOT/'models/macbert'


def configuration(path=DEFAULT_CONFIG):
    config=read_json(path)
    expected={'model_id','revision','seed','max_length','batch_size','gradient_accumulation',
              'epochs','patience','encoder_lr','head_lr','weight_decay','warmup_ratio',
              'huber_delta','dropout','max_grad_norm','threads','device'}
    if set(config)!=expected or config['model_id']!='hfl/chinese-macbert-base':
        raise ValueError('Unsupported model or unexpected training configuration fields.')
    revision=config['revision']
    if not isinstance(revision,str) or len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Pin an immutable model revision.')
    for key in ('seed','max_length','batch_size','gradient_accumulation','epochs','patience','threads'):
        if type(config[key]) is not int or config[key]<1:
            raise ValueError('Invalid positive integer: '+key)
    if not 8<=config['max_length']<=512:
        raise ValueError('MacBERT supports at most 512 tokens including the category.')
    for key in ('encoder_lr','head_lr','weight_decay','warmup_ratio','huber_delta','dropout','max_grad_norm'):
        value=config[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid numeric setting: '+key)
    if any(config[k]<=0 for k in ('encoder_lr','head_lr','huber_delta','max_grad_norm')):
        raise ValueError('Learning rates and loss/gradient thresholds must be positive.')
    if not 0<=config['dropout']<1 or not 0<=config['warmup_ratio']<1:
        raise ValueError('Dropout and warmup must be in [0,1).')
    if config['device'] not in ('auto','cpu','cuda'):
        raise ValueError('Device must be auto, cpu or cuda.')
    return config


def load_partitions(database=DATABASE):
    verify(database=database)
    with connect(database,readonly=True) as db:
        rows=examples(db)
    if not rows:
        raise ValueError('The master dataset is empty.')
    for row in rows:
        model_pair(row)
    partitions={split:[r for r in rows if r['split']==split] for split in ('train','validation','test')}
    for split,items in partitions.items():
        if set(r['category'] for r in items)!=set(CATEGORIES):
            raise ValueError('Each partition must cover every category: '+split)
    manifest={'contract_id':CONTRACT_ID,'counts':{s:len(rs) for s,rs in partitions.items()},
              'origins':dict(Counter(r['origin'] for r in rows)),
              'category_counts':{s:dict(Counter(r['category'] for r in rs)) for s,rs in partitions.items()},
              'dataset_sha256':sha256(dumps([(r['example_id'],r['category'],r['text'],r['scores'],r['split'],r['group_id']) for r in rows])),
              'partitions':{s:[r['example_id'] for r in rs] for s,rs in partitions.items()}}
    return partitions,manifest


def encode(tokenizer,category,text,max_length):
    result=tokenizer(CATEGORIES[category][0],text,add_special_tokens=True,truncation=False)
    if len(result['input_ids'])>max_length:
        raise ValueError(f'文本与类别共{len(result["input_ids"])}个token，超过{max_length}；不会静默截断，请缩短文本。')
    return result


class ScoreDataset:
    def __init__(self,rows,tokenizer,max_length):
        self.rows=rows
        self.features=[encode(tokenizer,r['category'],r['text'],max_length) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        return self.features[index],row['category'],row['scores']


class Collator:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer

    def __call__(self,items):
        import torch
        features,categories,scores=zip(*items)
        batch=dict(self.tokenizer.pad(list(features),padding=True,return_tensors='pt'))
        batch['categories']=list(categories)
        batch['labels']=torch.tensor([[score.get(m[0],0)/100 for m in METRICS] for score in scores],dtype=torch.float32)
        return batch
