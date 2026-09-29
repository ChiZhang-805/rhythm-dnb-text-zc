"""Disposable fixtures for Chinese-only, category-fixed score examples."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from text_distillation import chinese_data as zh,corpus,batch


class ChineseDataChecks(unittest.TestCase):
    def setUp(self):
        (corpus.ROOT/'results').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=corpus.ROOT/'results')
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.database=self.folder/'master.sqlite'
        self.packet=self.folder/'authoring.json'
        with corpus.connect(self.database) as db:
            corpus.initialize(db)
            for key,text,split in [('short','one word','train'),('real','original source '*50,'test')]:
                corpus.insert(db,{'record_id':key,'text':text,'split':split,'dataset':'fixture_only',
                    'group_id':key,'context':{},'provenance':{},'native_labels':{}})
        self.data={'method':'individually_authored','synthetic':[
            ['sim','emotion','仅供隔离测试使用的虚构中文段落。'*10,[72,63,0,9,4]]],
            'translations':[['real','stress','仅供隔离测试使用的中文翻译占位段落。'*10,[71]]]}

    def apply(self):
        self.packet.write_text(json.dumps(self.data,ensure_ascii=False),encoding='utf-8')
        return zh.import_authored(self.packet,database=self.database)

    def test_fixed_category_positions_and_integer_values(self):
        self.apply()
        with corpus.connect(self.database) as db:
            rows=zh.examples(db)
        inputs,targets=zh.model_pair(rows[0])
        self.assertEqual(set(inputs),{'category','text'})
        self.assertEqual(set(targets),set(zh.CATEGORIES['emotion'][1]))
        self.assertNotIn('split',targets)
        self.assertNotIn('origin',inputs)
        self.assertEqual(zh.validate_prediction('stress',{'stress_intensity':73}),{'stress_intensity':73})
        for bad in ({'joy_intensity':73},{'stress_intensity':101},{'stress_intensity':True},{'stress_intensity':73.5},{'stress_intensity':None},{}):
            with self.assertRaises(ValueError):
                zh.validate_prediction('stress',bad)

    def test_short_or_english_or_missing_answer_rejected(self):
        for text,scores in [('开心',[3]),('some English words '*100,[3]),('测试文本'*40,[None]),('测试文本'*40,[18,30])]:
            with self.assertRaises(ValueError):
                zh.validate('stress',text,scores,'authored_simulation')

    def test_meaningful_sentences_have_no_old_length_minimum(self):
        for text in ('睡不着。','忙完了，终于能松口气。','这周的事情都能安排过来。'):
            self.assertEqual(zh.validate('stress',text,[23],'authored_simulation'),{'stress_intensity':23})
        for text in ('','。','累','开心。','非常焦虑','stressed'):
            self.assertFalse(zh.sentence_candidate(text))

    def test_reviewed_recovery_preserves_identity_and_is_idempotent(self):
        snapshot=self.folder/'snapshot.sqlite'
        with corpus.connect(snapshot) as db:
            corpus.initialize(db)
            corpus.insert(db,{'record_id':'recovered','text':'I cannot fall asleep tonight.',
                'split':'test','dataset':'fixture_only','group_id':'recovered-person',
                'context':{},'native_labels':{},'provenance':{}})
        self.data['translations']=[['recovered','sleep','今晚我怎么也睡不着。',[18,88,17,46]]]
        self.packet.write_text(json.dumps(self.data,ensure_ascii=False),encoding='utf-8')
        result=zh.restore_reviewed_sources(self.packet,database=self.database,snapshot=snapshot)
        self.assertEqual(result['restored'],1)
        self.assertEqual(zh.restore_reviewed_sources(self.packet,database=self.database,snapshot=snapshot)['restored'],0)
        self.apply()
        with corpus.connect(self.database) as db:
            row=zh.examples(db)[1]
            self.assertEqual((row['split'],row['group_id']),('test','recovered-person'))
            self.assertEqual(db.execute('SELECT source_text_sha256 FROM source_restorations').fetchone()[0],row['source_sha256'])

    def test_real_partition_preserved_and_synthetic_training_only(self):
        self.apply()
        with corpus.connect(self.database) as db:
            rows=zh.examples(db)
        self.assertEqual([r['split'] for r in rows],['train','test'])
        self.assertEqual(rows[1]['source_record_id'],'real')
        self.assertTrue(zh.verify(database=self.database)['passed'])

    def test_import_is_idempotent_and_never_overwrites(self):
        self.assertEqual(self.apply()['active_examples'],2)
        self.assertEqual(self.apply()['active_examples'],2)
        self.data['synthetic'][0][3][0]=74
        with self.assertRaises(ValueError):
            self.apply()

    def test_punctuation_variants_do_not_inflate_sample_count(self):
        self.apply()
        repeated=deepcopy(self.data['synthetic'][0])
        repeated[0]='another-id'
        repeated[2]=repeated[2].replace('。','！')
        self.data['synthetic'].append(repeated)
        with self.assertRaisesRegex(ValueError,'Repeated text'):
            self.apply()

    def test_one_source_cannot_be_retranslated_as_a_second_category(self):
        self.apply()
        self.data['translations']=[['real','emotion','现在我对这些事终于没有那么担心了。',[65,34,12,19,4]]]
        with self.assertRaisesRegex(ValueError,'exactly one category'):
            self.apply()

    def test_fidelity_revision_preserves_split_and_records_previous_text(self):
        self.apply()
        with corpus.connect(self.database) as db:
            row=zh.examples(db)[1]
        item={'source_record_id':'real','category':'stress',
            'previous_payload_sha256':corpus.sha256(corpus.dumps([row['text'],row['scores']])),
            'text':'事情已经安排妥当，我现在不那么紧张了。','scores':[24],'reason':'Fixture-only fidelity revision.'}
        packet=self.folder/'fidelity.json'
        packet.write_text(json.dumps({'method':'individual_fidelity_review','items':[item]},ensure_ascii=False),encoding='utf-8')
        self.assertEqual(zh.review_fidelity(packet,database=self.database)['changed_records'],1)
        self.assertEqual(zh.review_fidelity(packet,database=self.database)['changed_records'],0)
        with corpus.connect(self.database) as db:
            revised=zh.examples(db)[1]
            history=db.execute('SELECT previous_payload FROM chinese_revision_history').fetchone()[0]
        self.assertEqual(revised['split'],'test')
        self.assertEqual(revised['group_id'],row['group_id'])
        self.assertEqual(json.loads(history)[0],row['text'])

    def test_single_sheet_only_text_category_and_scores(self):
        self.apply()
        with corpus.connect(self.database) as db:
            book=corpus.build_view(db)
        self.addCleanup(book.close)
        sheet=book.active
        self.assertEqual(book.sheetnames,['数据'])
        self.assertEqual((sheet.max_row,sheet.max_column),(3,20))
        self.assertEqual([c.value for c in sheet[1]][:3],['序号','文本','输入类别'])
        self.assertEqual(sheet['C2'].value,'情绪')
        self.assertEqual(sheet['D2'].value,72)
        self.assertEqual(sheet['F2'].value,0)
        self.assertEqual(sheet['H2'].value,4)
        self.assertIsNone(sheet['I2'].value)
        self.assertTrue(all(c.fill.fgColor.rgb=='00FFFFFF' for line in sheet for c in line))
        self.assertTrue(all(c.comment is None for line in sheet for c in line))
        self.assertEqual(len(sheet.conditional_formatting),0)
        self.assertEqual(sheet.freeze_panes,'D2')
        self.assertFalse(any(d.hidden for d in sheet.column_dimensions.values()))

    def test_view_roundtrip_never_exposes_answers(self):
        self.apply()
        with patch.object(corpus,'DATABASE',self.database):
            corpus.export()
        with patch.object(batch,'DATABASE',self.database):
            rows=batch.read_input(self.folder/'master_view.xlsx')
        self.assertEqual(rows[0]['category'],'emotion')
        self.assertNotIn('scores',rows[0])
        self.assertNotIn('总体心情',rows[0])
        self.assertNotIn('origin',rows[0])

    def test_short_text_cannot_be_reimported(self):
        self.apply()
        with corpus.connect(self.database) as db:
            with self.assertRaises(ValueError):
                corpus.insert(db,{'record_id':'another-short','text':'short','split':'train'})

    def test_null_in_real_translation_is_rejected_without_filling_zero(self):
        self.data['translations'][0][1]='sleep'
        self.data['translations'][0][3]=[None,87,None,None]
        with self.assertRaisesRegex(ValueError,'no missing scores'):
            self.apply()
        with corpus.connect(self.database) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM chinese_examples').fetchone()[0],0)

    def test_single_category_and_no_cross_category_targets(self):
        self.apply()
        with corpus.connect(self.database) as db:
            row=zh.examples(db)[0]
        row['scores']['stress_intensity']=65
        with self.assertRaisesRegex(ValueError,'category contract'):
            zh.model_pair(row)
        with self.assertRaises(ValueError):
            zh.validate(['emotion','stress'],'隔离测试文本'*30,[50],'translated_source')

    def test_review_preserves_old_scores_text_and_partition(self):
        self.apply()
        with corpus.connect(self.database) as db:
            db.execute("UPDATE chinese_examples SET scores=? WHERE source_record_id='real'",(json.dumps({'stress_intensity':None}),))
        item={'source_record_id':'real','category':'stress','scores':[62],
              'judgments':{'stress_intensity':{'basis':'weak_estimate','reason':'Fixture-only review; not a clinical measurement.'}}}
        review=self.folder/'review.json'
        review.write_text(json.dumps({'method':'individual_semantic_review','items':[item]}),encoding='utf-8')
        self.assertEqual(zh.review_scores(review,database=self.database)['changed_records'],1)
        self.assertEqual(zh.review_scores(review,database=self.database)['changed_records'],0)
        with corpus.connect(self.database) as db:
            history=db.execute('SELECT * FROM chinese_score_history').fetchone()
            row=zh.examples(db)[1]
        self.assertIsNone(json.loads(history['previous_scores'])['stress_intensity'])
        self.assertEqual(row['scores']['stress_intensity'],62)
        self.assertEqual(row['split'],'test')
        self.assertEqual(row['text'],self.data['translations'][0][2])

    def test_category_sorting_preserves_source_identity_and_splits(self):
        self.data['synthetic'].insert(0,['social','social','仅供隔离测试使用的社交描述。'*12,[30,60,10,20]])
        self.apply()
        with corpus.connect(self.database) as db:
            original=[(r['example_id'],r['split']) for r in zh.examples(db)]
            displayed=zh.display_examples(db)
        self.assertEqual([r['category'] for r in displayed],['emotion','stress','social'])
        self.assertEqual([r['view_id'] for r in displayed],[1,2,3])
        self.assertEqual(set(original),{(r['example_id'],r['split']) for r in displayed})
        with patch.object(corpus,'DATABASE',self.database):
            corpus.export()
        with patch.object(batch,'DATABASE',self.database):
            restored=batch.read_input(self.folder/'master_view.xlsx')
        self.assertEqual([r['record_id'] for r in restored],[r['example_id'] for r in displayed])


if __name__=='__main__':
    unittest.main()
