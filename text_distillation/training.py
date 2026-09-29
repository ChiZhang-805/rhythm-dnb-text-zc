"""Full-data MacBERT regression fine-tuning; test labels never select the checkpoint."""

from contextlib import nullcontext
from datetime import datetime,timezone
import math
from pathlib import Path
import random
import tempfile
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer,get_linear_schedule_with_warmup
from safetensors.torch import save_file,load_file

from .artifacts import save_json,file_hash
from .contract import CATEGORIES,CONTRACT_ID,schema
from .corpus import ROOT,read_json
from .learning_data import configuration,load_partitions,ScoreDataset,Collator,BASE,MODEL_ROOT
from .model import ScoringModel,regression_loss
from .evaluation import mean_baseline,report


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def device_for(name):
    if name=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable.')
    return torch.device('cuda' if name=='cuda' or name=='auto' and torch.cuda.is_available() else 'cpu')


def base_tokenizer(config):
    receipt=read_json(BASE/'download.json')
    if receipt['model_id']!=config['model_id'] or receipt['revision']!=config['revision']:
        raise ValueError('Base model revision differs; run prepare with the intended config.')
    for name,digest in receipt['files'].items():
        if file_hash(BASE/name)!=digest:
            raise ValueError('Base file checksum mismatch: '+name)
    return BertTokenizer.from_pretrained(BASE,local_files_only=True)


def prepare(config_path=None, *, download=True):
    from huggingface_hub import snapshot_download
    config=configuration(config_path) if config_path else configuration()
    partitions,manifest=load_partitions()
    if download:
        snapshot_download(config['model_id'],revision=config['revision'],local_dir=BASE,
                          allow_patterns=['config.json','pytorch_model.bin','vocab.txt','tokenizer.json',
                                          'tokenizer_config.json','special_tokens_map.json','added_tokens.json'])
        files={p.name:file_hash(p) for p in BASE.iterdir() if p.is_file() and p.name!='download.json'}
        save_json(BASE/'download.json',{'model_id':config['model_id'],'revision':config['revision'],'files':files})
    tokenizer=base_tokenizer(config)
    lengths={}
    for split,rows in partitions.items():
        dataset=ScoreDataset(rows,tokenizer,config['max_length'])
        lengths[split]={'records':len(rows),'max_tokens':max(len(f['input_ids']) for f in dataset.features)}
    result={'status':'prepared','model':config['model_id'],'revision':config['revision'],
            'dataset':manifest,'tokenization':lengths,'config':config,'cuda':torch.cuda.is_available(),
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'trained':False}
    save_json(ROOT/'results/macbert/preparation.json',result)
    return {**result,'dataset':{k:v for k,v in manifest.items() if k!='partitions'}}


def optimizer_for(model,config):
    return torch.optim.AdamW([
        {'params':model.encoder.parameters(),'lr':config['encoder_lr']},
        {'params':model.heads.parameters(),'lr':config['head_lr']}],weight_decay=config['weight_decay'])


def loader(rows,tokenizer,config, *, shuffle=False):
    return DataLoader(ScoreDataset(rows,tokenizer,config['max_length']),batch_size=config['batch_size'],
                      shuffle=shuffle,collate_fn=Collator(tokenizer),num_workers=0,
                      generator=torch.Generator().manual_seed(config['seed']))


def forward_batch(model,batch,device):
    return model(**{k:v.to(device) for k,v in batch.items() if k not in ('labels','categories')})


@torch.inference_mode()
def predict_rows(model,tokenizer,rows,config,device):
    model.eval()
    predictions=[]
    for batch in loader(rows,tokenizer,config):
        outputs=forward_batch(model,batch,device)
        for i,category in enumerate(batch['categories']):
            values=(outputs[category][i].float().cpu()*100).tolist()
            predictions.append(dict(zip(CATEGORIES[category][1],values)))
    return predictions


def save_checkpoint(folder,model,tokenizer,config,manifest):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    temporary=folder/'model.pending.safetensors'
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(temporary))
    temporary.replace(folder/'model.safetensors')
    model.encoder.config.save_pretrained(folder/'encoder')
    tokenizer.save_pretrained(folder/'tokenizer')
    files={str(p.relative_to(folder)).replace('\\','/'):file_hash(p) for p in folder.rglob('*')
           if p.is_file() and p.name!='manifest.json'}
    save_json(folder/'manifest.json',{**manifest,'contract':schema(),'config':config,'files':files})


def load_checkpoint(folder, *, allow_smoke=False):
    folder=Path(folder)
    manifest=read_json(folder/'manifest.json')
    if manifest['contract']!=schema():
        raise ValueError('Checkpoint output contract differs from the current API.')
    if manifest.get('purpose')!='full_dataset_finetune' and not allow_smoke:
        raise ValueError('测试权重不能用于正式预测，请先运行完整训练。')
    if manifest.get('status')!='complete' and not allow_smoke:
        raise ValueError('训练尚未完成，不能加载中间检查点。')
    for name,digest in manifest['files'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or file_hash(path)!=digest:
            raise ValueError('Checkpoint file checksum mismatch.')
    model=ScoringModel.from_config(folder/'encoder',manifest['config']['dropout'])
    model.load_state_dict(load_file(str(folder/'model.safetensors')),strict=True)
    tokenizer=BertTokenizer.from_pretrained(folder/'tokenizer',local_files_only=True)
    return model,tokenizer,manifest


def train(config_path=None):
    config=configuration(config_path) if config_path else configuration()
    torch.set_num_threads(config['threads'])
    seed_all(config['seed'])
    partitions,data_manifest=load_partitions()
    tokenizer=base_tokenizer(config)
    train_loader=loader(partitions['train'],tokenizer,config,shuffle=True)
    # Validate every row before allocating the optimizer, including held-out token lengths.
    for split in ('validation','test'):
        ScoreDataset(partitions[split],tokenizer,config['max_length'])
    device=device_for(config['device'])
    model=ScoringModel.pretrained(BASE,config['dropout']).to(device)
    optimizer=optimizer_for(model,config)
    steps_per_epoch=math.ceil(len(train_loader)/config['gradient_accumulation'])
    total_steps=steps_per_epoch*config['epochs']
    scheduler=get_linear_schedule_with_warmup(optimizer,int(total_steps*config['warmup_ratio']),total_steps)
    dtype=torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and dtype==torch.float16)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    run=MODEL_ROOT/'runs'/run_id
    run.mkdir(parents=True,exist_ok=False)
    result_file=ROOT/'results/macbert'/run_id/'training.json'
    baseline=mean_baseline(partitions['train'])
    metadata={'purpose':'full_dataset_finetune','status':'training','run_id':run_id,
              'dataset':data_manifest,'mean_baseline':baseline,'history':[],
              'selection_metric':'validation.macro_category_mae','test_used_for_selection':False}
    best=float('inf')
    stale=0
    started=time.monotonic()
    try:
        for epoch in range(1,config['epochs']+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss=0.0
            seen=0
            for index,batch in enumerate(train_loader):
                group_start=(index//config['gradient_accumulation'])*config['gradient_accumulation']
                # Scale by sample count, including an incomplete final accumulation group.
                group_samples=min(len(partitions['train'])-group_start*config['batch_size'],
                                  config['batch_size']*config['gradient_accumulation'])
                count=len(batch['categories'])
                context=torch.autocast('cuda',dtype=dtype) if device.type=='cuda' else nullcontext()
                with context:
                    outputs=forward_batch(model,batch,device)
                    loss=regression_loss(outputs,batch['labels'].to(device),batch['categories'],config['huber_delta'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss; checkpoint not published.')
                scaler.scale(loss*count/group_samples).backward()
                epoch_loss+=float(loss.detach())*count
                seen+=count
                if (index+1)%config['gradient_accumulation']==0 or index+1==len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'],error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if (index+1)%25==0:
                    print(f'Epoch {epoch}: {seen}/{len(partitions["train"])} training records',flush=True)
            if seen!=len(partitions['train']):
                raise ValueError('An epoch did not consume the complete training partition.')
            validation=report(partitions['validation'],predict_rows(model,tokenizer,partitions['validation'],config,device),baseline)
            metadata['history'].append({'epoch':epoch,'training_records_seen':seen,'loss':epoch_loss/seen,'validation':validation})
            if validation['macro_category_mae']<best:
                best=validation['macro_category_mae']
                metadata['best_epoch']=epoch
                save_checkpoint(run/'checkpoint',model,tokenizer,config,metadata)
                stale=0
            else:
                stale+=1
            save_json(result_file,{**metadata,'config':config,'elapsed_seconds':time.monotonic()-started})
            print(f'Epoch {epoch}: validation MAE={validation["macro_category_mae"]:.3f}',flush=True)
            if stale>=config['patience']:
                break
        saved=read_json(run/'checkpoint/manifest.json')
        saved.update(status='complete',history=metadata['history'],elapsed_seconds=time.monotonic()-started)
        save_json(run/'checkpoint/manifest.json',saved)
        save_json(result_file,saved)
        save_json(MODEL_ROOT/'current.json',{'checkpoint':str((run/'checkpoint').relative_to(MODEL_ROOT)),
                                          'run_id':run_id,'contract_id':CONTRACT_ID})
        return {'status':'complete','run_id':run_id,'training_records':len(partitions['train']),
                'best_epoch':metadata['best_epoch'],'validation_mae':best,
                'checkpoint':str(run/'checkpoint'),'report':str(result_file),'test_evaluated':False}
    except BaseException as error:
        save_json(result_file,{**metadata,'status':'failed','error':str(error)})
        raise


def smoke(config_path=None):
    """Real pretrained encoder forward/backward/save/load, never a deployable checkpoint."""
    config=configuration(config_path) if config_path else configuration()
    seed_all(config['seed'])
    torch.set_num_threads(config['threads'])
    partitions,_=load_partitions()
    rows=[next(r for r in partitions['train'] if r['category']==category) for category in CATEGORIES]
    tokenizer=base_tokenizer(config)
    device=device_for(config['device'])
    model=ScoringModel.pretrained(BASE,config['dropout']).to(device)
    optimizer=optimizer_for(model,config)
    dataset=ScoreDataset(rows,tokenizer,config['max_length'])
    batch=Collator(tokenizer)([dataset[i] for i in range(len(rows))])
    dtype=torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and dtype==torch.float16)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats()
    before={name:head.weight.detach().clone() for name,head in model.heads.items()}
    losses=[]
    started=time.monotonic()
    for _ in range(2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        context=torch.autocast('cuda',dtype=dtype) if device.type=='cuda' else nullcontext()
        with context:
            loss=regression_loss(forward_batch(model,batch,device),batch['labels'].to(device),batch['categories'],config['huber_delta'])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite smoke loss.')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters()):
            raise ValueError('The encoder did not receive gradients.')
        torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'],error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    updated={c:not torch.equal(before[c],model.heads[c].weight) for c in CATEGORIES}
    if not all(updated.values()):
        raise ValueError('A scoring head did not train.')
    all_train=ScoreDataset(partitions['train'],tokenizer,config['max_length'])
    longest=sorted(range(len(all_train)),key=lambda i:len(all_train.features[i]['input_ids']),reverse=True)[:config['batch_size']]
    longest_batch=Collator(tokenizer)([all_train[i] for i in longest])
    optimizer.zero_grad(set_to_none=True)
    context=torch.autocast('cuda',dtype=dtype) if device.type=='cuda' else nullcontext()
    with context:
        longest_loss=regression_loss(forward_batch(model,longest_batch,device),longest_batch['labels'].to(device),
                                    longest_batch['categories'],config['huber_delta'])
    if not torch.isfinite(longest_loss):
        raise ValueError('Nonfinite loss on the longest training batch.')
    longest_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    peak_memory=torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else None
    expected=predict_rows(model,tokenizer,rows,config,device)
    parent=ROOT/'results/macbert'
    parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='smoke-',dir=parent) as folder:
        save_checkpoint(folder,model,tokenizer,config,{'purpose':'smoke_only','status':'test_only'})
        restored,tok,_=load_checkpoint(folder,allow_smoke=True)
        restored.to(device)
        actual=predict_rows(restored,tok,rows,config,device)
        max_error=max(abs(actual[i][k]-expected[i][k]) for i in range(len(rows)) for k in actual[i])
        if max_error>1e-4:
            raise ValueError('Checkpoint roundtrip changed predictions.')
        try:
            load_checkpoint(folder)
        except ValueError:
            pass
        else:
            raise ValueError('A smoke checkpoint was accepted for deployment.')
    result={'passed':True,'kind':'engineering_smoke_not_model_accuracy','records':len(rows),'optimizer_steps':2,
            'all_heads_updated':updated,'encoder_gradients_verified':True,'roundtrip_max_error':max_error,
            'losses':losses,'device':str(device),'elapsed_seconds':time.monotonic()-started,
            'mixed_precision':str(dtype) if device.type=='cuda' else None,
            'longest_batch_shape':list(longest_batch['input_ids'].shape),'peak_allocated_gpu_gib':peak_memory,
            'deployment_checkpoint_created':False}
    save_json(parent/'smoke.json',result)
    return result
