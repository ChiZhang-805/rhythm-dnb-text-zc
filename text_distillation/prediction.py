"""One trained model for CLI, batch, HTTP and subsequent text-feature integration."""

from pathlib import Path
import threading
import time
import math

from .contract import CATEGORIES,METRICS,validate_input
from .corpus import ROOT,read_json
from .artifacts import save_json,file_hash
from .learning_data import MODEL_ROOT,encode,load_partitions


class ModelNotReady(RuntimeError):
    pass


def current_checkpoint():
    pointer=MODEL_ROOT/'current.json'
    if not pointer.is_file():
        raise ModelNotReady('尚未完成MacBERT正式训练；接口已就绪，暂不提供评分。')
    info=read_json(pointer)
    path=(MODEL_ROOT/info['checkpoint']).resolve()
    if not path.is_relative_to(MODEL_ROOT.resolve()) or not (path/'manifest.json').is_file():
        raise ModelNotReady('已发布模型路径无效，请检查训练结果。')
    return path


class Predictor:
    def __init__(self,checkpoint=None,device='cpu'):
        self.checkpoint=Path(checkpoint) if checkpoint else None
        self.device_name=device
        self.loaded=None
        self.lock=threading.Lock()

    def health(self):
        try:
            path=self.checkpoint or current_checkpoint()
            info=read_json(path/'manifest.json')
            if info.get('purpose')!='full_dataset_finetune' or info.get('status')!='complete':
                raise ModelNotReady('当前权重不是已完成的正式训练模型。')
            return {'ready':True,'status':'trained_checkpoint_available','run_id':info['run_id']}
        except (OSError,ValueError,KeyError,ModelNotReady) as error:
            return {'ready':False,'status':'needs_training','message':str(error)}

    def _load(self):
        from .training import load_checkpoint,device_for
        import torch
        path=self.checkpoint or current_checkpoint()
        identity=(str(path),file_hash(path/'manifest.json'))
        if self.loaded!=identity:
            model,tokenizer,manifest=load_checkpoint(path)
            torch.set_num_threads(manifest['config']['threads'])
            device=device_for(self.device_name)
            model.to(device).eval()
            self.model,self.tokenizer,self.manifest,self.device=model,tokenizer,manifest,device
            self.loaded=identity

    def predict(self,category,text):
        import torch
        category,text=validate_input(category,text)
        started=time.monotonic()
        with self.lock:
            self._load()
            encoded=encode(self.tokenizer,category,text,self.manifest['config']['max_length'])
            batch={k:torch.tensor([v],device=self.device) for k,v in encoded.items()}
            with torch.inference_mode():
                values=(self.model(**batch)[category][0].float().cpu()*100).tolist()
            if any(not math.isfinite(value) for value in values):
                raise RuntimeError('模型输出了无效数值，未返回任何评分。')
            keys=CATEGORIES[category][1]
            scores={k:int(math.floor(v+0.5)) for k,v in zip(keys,values)}
            from .chinese_data import validate_prediction
            validate_prediction(category,scores)
            return {'category':category,'category_label':CATEGORIES[category][0],
                    'scores':scores,'normalized':{k:v/100 for k,v in scores.items()},
                    'metrics':[{'key':k,'label':next(m[2] for m in METRICS if m[0]==k),'score':scores[k]} for k in keys],
                    'model_id':self.manifest['config']['model_id'],'run_id':self.manifest['run_id'],
                    'elapsed_ms':round((time.monotonic()-started)*1000),
                    'score_kind':'text_semantic_estimate_not_probability'}


def evaluate(split='test',checkpoint=None):
    if split not in ('validation','test'):
        raise ValueError('Evaluate validation or test, not training data.')
    from .training import load_checkpoint,predict_rows,device_for
    from .evaluation import report
    path=Path(checkpoint) if checkpoint else current_checkpoint()
    model,tokenizer,manifest=load_checkpoint(path)
    partitions,data_manifest=load_partitions()
    if data_manifest['dataset_sha256']!=manifest['dataset']['dataset_sha256']:
        raise ValueError('The master dataset changed since training; frozen evaluation is required.')
    device=device_for(manifest['config']['device'])
    model.to(device)
    rows=partitions[split]
    predictions=predict_rows(model,tokenizer,rows,manifest['config'],device)
    result={'split':split,'run_id':manifest['run_id'],
            **report(rows,predictions,manifest['mean_baseline']),
            'rounded_scores':report(rows,[{k:int(math.floor(v+0.5)) for k,v in p.items()} for p in predictions],manifest['mean_baseline'])}
    folder=ROOT/'results/macbert'/manifest['run_id']
    save_json(folder/(split+'.json'),result)
    save_json(folder/(split+'_predictions.json'),[
        {'example_id':r['example_id'],'category':r['category'],'reference':r['scores'],'predicted':p}
        for r,p in zip(rows,predictions)])
    return result
