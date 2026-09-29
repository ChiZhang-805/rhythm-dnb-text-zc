"""Isolated regression, checkpoint and end-to-end training checks; no master writes."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import BertConfig,BertModel,BertTokenizer

from text_distillation.contract import CATEGORIES,METRICS,normalize_category,validate_input,schema
from text_distillation.corpus import ROOT
from text_distillation.learning_data import configuration,ScoreDataset,Collator,encode,load_partitions
from text_distillation.model import ScoringModel,regression_loss
from text_distillation.evaluation import errors,mean_baseline,report
from text_distillation.training import save_checkpoint,load_checkpoint,train
from text_distillation.prediction import Predictor,ModelNotReady,current_checkpoint


def tiny_model():
    return ScoringModel(BertModel(BertConfig(vocab_size=80,hidden_size=24,num_hidden_layers=1,
        num_attention_heads=4,intermediate_size=32,max_position_embeddings=512)),dropout=0)


def fixture_rows(split='train'):
    return [{'example_id':split+':'+c,'category':c,'text':'今天的感受是很累但还可以。',
             'scores':{k:13+i*14 for i,k in enumerate(keys)},'split':split,
             'group_id':split+':'+c,'origin':'translated_source'} for c,(_,keys) in CATEGORIES.items()]


class ScoringChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        (ROOT/'results').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'results')
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]']+list(dict.fromkeys('今天的感受是很累但还可以情绪压力饮食睡眠社交。'))
        (self.folder/'vocab.txt').write_text('\n'.join(vocab),encoding='utf-8')
        self.tokenizer=BertTokenizer(vocab=str(self.folder/'vocab.txt'))
        self.config=configuration()
        self.config.update(device='cpu',epochs=2,batch_size=3,gradient_accumulation=2,dropout=0,threads=2)

    def test_category_contract_and_aliases(self):
        self.assertEqual(normalize_category('睡眠'),'sleep')
        self.assertEqual(len(METRICS),17)
        self.assertEqual([len(v[1]) for v in CATEGORIES.values()],[5,1,3,4,4])
        self.assertTrue(next(x for x in schema()['categories'] if x['id']=='stress')['auxiliary'])
        for category,text in [('unknown','今天很累'),(['sleep'],'今天很累'),('sleep',' '),('sleep','English only'),('sleep','中'*10001)]:
            with self.assertRaises(ValueError):
                validate_input(category,text)

    def test_prompt_encodes_category_and_never_answers(self):
        rows=fixture_rows()
        before=ScoreDataset(rows,self.tokenizer,512).features
        changed=deepcopy(rows)
        changed[0]['scores']={k:99 for k in changed[0]['scores']}
        changed[0]['origin']='authored_simulation'
        changed[0]['split']='test'
        self.assertEqual(before,ScoreDataset(changed,self.tokenizer,512).features)
        self.assertNotEqual(encode(self.tokenizer,'sleep',rows[0]['text'],512),encode(self.tokenizer,'emotion',rows[0]['text'],512))

    def test_oversized_text_is_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError,'截断'):
            encode(self.tokenizer,'sleep','今天'*100,32)

    def test_heads_have_fixed_shapes_and_bounds(self):
        rows=fixture_rows()
        batch=Collator(self.tokenizer)(list(ScoreDataset(rows,self.tokenizer,512)))
        outputs=tiny_model()(**{k:v for k,v in batch.items() if k not in ('labels','categories')})
        for category,(_,keys) in CATEGORIES.items():
            self.assertEqual(outputs[category].shape,(5,len(keys)))
            self.assertTrue(torch.isfinite(outputs[category]).all())
            self.assertTrue(((outputs[category]>=0)&(outputs[category]<=1)).all())

    def test_loss_ignores_out_of_category_targets(self):
        outputs={c:torch.full((1,len(keys)),.4,requires_grad=True) for c,(_,keys) in CATEGORIES.items()}
        labels=torch.zeros((1,17))
        first=regression_loss(outputs,labels,['stress'])
        labels[0,:5]=float('nan')
        second=regression_loss(outputs,labels,['stress'])
        self.assertEqual(first.item(),second.item())
        second.backward()
        self.assertIsNotNone(outputs['stress'].grad)
        self.assertIsNone(outputs['emotion'].grad)

    def test_loss_balances_samples_not_number_of_outputs(self):
        outputs={c:torch.full((2,len(keys)),.5) for c,(_,keys) in CATEGORIES.items()}
        labels=torch.zeros((2,17))
        combined=regression_loss(outputs,labels,['emotion','stress'])
        expected=torch.nn.functional.huber_loss(torch.tensor([.5]),torch.tensor([0.]),delta=.1)
        self.assertAlmostEqual(combined.item(),expected.item())

    def test_unknown_training_category_is_rejected(self):
        with self.assertRaises(ValueError):
            regression_loss({},torch.zeros((1,17)),['unknown'])

    def test_metric_error_and_constant_rank(self):
        result=errors([10,20,30],[14,30,50])
        self.assertAlmostEqual(result['mae'],34/3)
        self.assertAlmostEqual(result['within_5'],1/3)
        self.assertAlmostEqual(result['within_10'],2/3)
        self.assertIsNone(errors([10,10],[20,20])['spearman'])

    def test_baseline_uses_training_scores_only(self):
        rows=fixture_rows()
        baseline=mean_baseline(rows)
        evaluation=fixture_rows('test')
        evaluation[0]['scores']['mood_valence']=99
        self.assertEqual(baseline['mood_valence'],13)
        result=report(evaluation,[r['scores'] for r in evaluation],baseline)
        self.assertEqual(result['macro_category_mae'],0)
        self.assertGreater(result['mean_baseline']['mood_valence']['mae'],0)

    def test_missing_extra_and_nan_predictions_rejected(self):
        rows=fixture_rows()
        for bad in ({},{'extra':5},{k:float('nan') for k in rows[0]['scores']}):
            predictions=[dict(r['scores']) for r in rows]
            predictions[0]=bad
            with self.assertRaises(ValueError):
                report(rows,predictions,mean_baseline(rows))

    def checkpoint(self,purpose='full_dataset_finetune'):
        folder=self.folder/'checkpoint'
        save_checkpoint(folder,tiny_model(),self.tokenizer,self.config,
                        {'purpose':purpose,'status':'complete','run_id':'isolated-unit-fixture'})
        return folder

    def test_checkpoint_roundtrip_and_fixed_public_output(self):
        folder=self.checkpoint()
        predictor=Predictor(folder)
        for c,(_,keys) in CATEGORIES.items():
            value=predictor.predict(c,'今天的感受是很累但还可以。')
            self.assertEqual(set(value['scores']),set(keys))
            self.assertTrue(all(type(v) is int and 0<=v<=100 for v in value['scores'].values()))
            self.assertEqual(value['normalized'],{k:v/100 for k,v in value['scores'].items()})
        self.assertEqual(predictor.predict('睡眠','今天很累。')['category'],'sleep')

    def test_smoke_and_incomplete_checkpoints_cannot_serve(self):
        folder=self.checkpoint('smoke_only')
        with self.assertRaises(ValueError):
            load_checkpoint(folder)
        self.assertFalse(Predictor(folder).health()['ready'])
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        manifest.update(purpose='full_dataset_finetune',status='training')
        (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        with self.assertRaises(ValueError):
            load_checkpoint(folder)

    def test_corrupted_checkpoint_and_contract_rejected(self):
        folder=self.checkpoint()
        with (folder/'model.safetensors').open('ab') as handle:
            handle.write(b'corrupted')
        with self.assertRaisesRegex(ValueError,'checksum'):
            load_checkpoint(folder)

    def test_missing_current_checkpoint_is_not_a_fake_prediction(self):
        with patch('text_distillation.prediction.MODEL_ROOT',self.folder/'absent'):
            self.assertFalse(Predictor().health()['ready'])
            with self.assertRaises(ModelNotReady):
                Predictor().predict('sleep','今天很累。')

    def test_full_training_loop_consumes_all_rows_and_roundtrips(self):
        partitions={s:fixture_rows(s) for s in ('train','validation','test')}
        manifest={'dataset_sha256':'isolated-fixture','counts':{s:5 for s in partitions}}
        config_path=self.folder/'train_config.json'
        config_path.write_text(json.dumps(self.config),encoding='utf-8')
        with patch('text_distillation.training.load_partitions',return_value=(partitions,manifest)), \
             patch('text_distillation.training.base_tokenizer',return_value=self.tokenizer), \
             patch('text_distillation.training.ScoringModel.pretrained',side_effect=lambda *args:tiny_model()), \
             patch('text_distillation.training.ROOT',self.folder), \
             patch('text_distillation.training.MODEL_ROOT',self.folder/'models'):
            result=train(config_path)
        self.assertEqual(result['training_records'],5)
        self.assertFalse(result['test_evaluated'])
        model,tokenizer,saved=load_checkpoint(result['checkpoint'])
        self.assertTrue(all(epoch['training_records_seen']==5 for epoch in saved['history']))
        self.assertEqual(len(saved['history']),2)
        self.assertTrue((self.folder/'models/current.json').exists())

    def test_invalid_configuration_rejected(self):
        for key,value in [('max_length',513),('dropout',1),('batch_size',True),('encoder_lr',float('nan')),('device','invalid')]:
            config={**self.config,key:value}
            path=self.folder/'bad.json'
            path.write_text(json.dumps(config),encoding='utf-8')
            with self.assertRaises(ValueError):
                configuration(path)

    def test_early_stop_and_failed_run_do_not_publish_wrong_weights(self):
        partitions={s:fixture_rows(s) for s in ('train','validation','test')}
        manifest={'dataset_sha256':'isolated-fixture','counts':{s:5 for s in partitions}}
        config={**self.config,'epochs':6,'patience':1}
        config_path=self.folder/'early.json'
        config_path.write_text(json.dumps(config),encoding='utf-8')
        with patch('text_distillation.training.load_partitions',return_value=(partitions,manifest)), \
             patch('text_distillation.training.base_tokenizer',return_value=self.tokenizer), \
             patch('text_distillation.training.ScoringModel.pretrained',side_effect=lambda *args:tiny_model()), \
             patch('text_distillation.training.ROOT',self.folder), \
             patch('text_distillation.training.MODEL_ROOT',self.folder/'models'), \
             patch('text_distillation.training.report',side_effect=[{'macro_category_mae':10},{'macro_category_mae':11}]):
            result=train(config_path)
            pointer=self.folder/'models/current.json'
            previous=pointer.read_bytes()
            with patch('text_distillation.training.regression_loss',side_effect=RuntimeError('isolated failure')):
                with self.assertRaises(RuntimeError):
                    train(config_path)
            self.assertEqual(pointer.read_bytes(),previous)
        _,_,saved=load_checkpoint(result['checkpoint'])
        self.assertEqual(saved['best_epoch'],1)
        self.assertEqual(len(saved['history']),2)

    def test_evaluation_covers_frozen_test_and_rejects_changed_master(self):
        from text_distillation.prediction import evaluate
        folder=self.folder/'evaluation-checkpoint'
        rows=fixture_rows()
        save_checkpoint(folder,tiny_model(),self.tokenizer,self.config,{
            'purpose':'full_dataset_finetune','status':'complete','run_id':'evaluation-fixture',
            'dataset':{'dataset_sha256':'frozen'},'mean_baseline':mean_baseline(rows)})
        partitions={s:fixture_rows(s) for s in ('train','validation','test')}
        with patch('text_distillation.prediction.ROOT',self.folder), \
             patch('text_distillation.prediction.load_partitions',return_value=(partitions,{'dataset_sha256':'frozen'})):
            result=evaluate('test',folder)
            self.assertEqual(result['records'],5)
            self.assertEqual(len(result['per_metric']),17)
        with patch('text_distillation.prediction.load_partitions',return_value=(partitions,{'dataset_sha256':'changed'})):
            with self.assertRaisesRegex(ValueError,'changed'):
                evaluate('test',folder)

    def test_pointer_cannot_escape_model_directory(self):
        folder=self.folder/'models'
        folder.mkdir()
        (folder/'current.json').write_text(json.dumps({'checkpoint':'../elsewhere'}),encoding='utf-8')
        with patch('text_distillation.prediction.MODEL_ROOT',folder):
            with self.assertRaises(ModelNotReady):
                current_checkpoint()

    def test_master_full_partitions_read_only_and_no_leakage(self):
        from text_distillation.artifacts import file_hash
        from text_distillation.corpus import DATABASE
        if not DATABASE.exists():
            self.skipTest('Private master dataset is not included in the public code repository.')
        before=file_hash(DATABASE)
        partitions,manifest=load_partitions()
        self.assertEqual({s:len(rs) for s,rs in partitions.items()},manifest['counts'])
        self.assertGreaterEqual(sum(manifest['counts'].values()),3000)
        self.assertEqual(file_hash(DATABASE),before)
        groups={s:{r['group_id'] for r in rs} for s,rs in partitions.items()}
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['train'] & groups['validation'])


if __name__=='__main__':
    unittest.main()
