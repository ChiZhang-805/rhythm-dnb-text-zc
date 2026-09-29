"""Check the delivered view and describe coverage; never create text or score labels."""

from collections import Counter
import json
import statistics
import hashlib

from .corpus import ROOT,connect,sha256,dumps
from .chinese_data import CATEGORIES,VIEW_METRICS,display_examples,verify,han_count


def run():
    from openpyxl import load_workbook
    from sklearn.feature_extraction.text import TfidfVectorizer

    report=verify()
    with connect() as db:
        rows=display_examples(db)
        report['restored_real_sources']=db.execute('SELECT COUNT(*) FROM source_restorations').fetchone()[0]
        report['real_source_datasets']=dict(Counter(r[0] for r in db.execute('''
            SELECT r.dataset FROM chinese_examples e JOIN records r ON r.record_id=e.source_record_id''')))
    book=load_workbook(ROOT/'dataset/master_view.xlsx')
    try:
        sheet=book.active
        expected=[['序号','文本','输入类别']+[m[2] for m in VIEW_METRICS]]
        expected += [[r['view_id'],r['text'],CATEGORIES[r['category']][0]]+
                     [r['scores'].get(m[0]) for m in VIEW_METRICS] for r in rows]
        if book.sheetnames!=['数据'] or list(sheet.values)!=[tuple(r) for r in expected]:
            raise ValueError('Workbook differs from the one authoritative dataset.')
        if len(sheet.conditional_formatting) or any(c.comment or c.fill.fgColor.rgb!='00FFFFFF' for line in sheet for c in line):
            raise ValueError('Workbook must be plain white, without comments or conditional formatting.')
        report['workbook_matches_master']=True
    finally:
        book.close()
    report['target_examples']=3000
    report['sample_count_target_met']=len(rows)>=3000
    report['category_target']=600
    report['category_targets_met']=all(sum(r['category']==c for r in rows)>=600 for c in CATEGORIES)
    report['category_by_split']={CATEGORIES[c][0]:dict(Counter(r['split'] for r in rows if r['category']==c)) for c in CATEGORIES}
    report['score_coverage']={}
    for key,_,name,*_ in VIEW_METRICS:
        values=[r['scores'][key] for r in rows if key in r['scores']]
        report['score_coverage'][name]={'n':len(values),'min':min(values),'max':max(values),
            'mean':round(statistics.mean(values),2),'std':round(statistics.pstdev(values),2),
            'bins_0_20_40_60_80_100':[sum(lo<=v<=hi for v in values) for lo,hi in ((0,20),(21,40),(41,60),(61,80),(81,100))]}
    lengths=[han_count(r['text']) for r in rows]
    report['text_han_characters']={'min':min(lengths),'median':statistics.median(lengths),'max':max(lengths)}
    report['dataset_sha256']=sha256(dumps([(r['example_id'],r['text'],r['scores'],r['split']) for r in rows]))
    packets=[ROOT/'results/chinese_authoring.json']+sorted((ROOT/'results/authored_packets').glob('*.json'))
    authored={}
    report['authoring_files']={}
    for path in packets:
        packet=json.loads(path.read_text(encoding='utf-8'))
        report['authoring_files'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for origin,field in (('authored_simulation','synthetic'),('translated_source','translations')):
            for key,category,text,values in packet[field]:
                identifier='zh:'+origin+':'+key+':'+category
                if identifier in authored:
                    raise ValueError('An authored example occurs in multiple input receipts.')
                authored[identifier]=(text,dict(zip(CATEGORIES[category][1],values)),origin)
    persisted={r['example_id']:(r['text'],r['scores'],r['origin']) for r in rows}
    if authored!=persisted:
        raise ValueError('Master labels differ from the literal authored text/score receipts.')
    report['authoring_matches_master']=True
    review=json.loads((ROOT/'results/source_translation_review.json').read_text(encoding='utf-8'))
    corrections=json.loads((ROOT/'results/source_translation_corrections.json').read_text(encoding='utf-8'))
    revised={item['source_record_id'] for item in corrections['items']}
    retained={corrections['neutral_control_adjudication']['source_record_id']}
    if {item['source_record_id'] for item in review['findings']}!=revised|retained:
        raise ValueError('Fidelity review findings have not all been adjudicated.')
    by_source={r['source_record_id']:r for r in rows if r['source_record_id']}
    for item in corrections['items']:
        row=by_source[item['source_record_id']]
        if row['text']!=item['text'] or list(row['scores'].values())!=item['scores']:
            raise ValueError('A fidelity correction is missing from the delivered dataset.')
    report['translation_fidelity_review']={'reviewed':review['summary']['reviewed_count'],
        'findings_adjudicated':len(revised|retained),'corrected_records':len(revised),'retained_control_records':len(retained)}
    # Similar wording is a review candidate, not proof of identical meaning or permission to delete.
    vectors=TfidfVectorizer(analyzer='char',ngram_range=(2,4)).fit_transform([r['text'] for r in rows])
    similarities=(vectors@vectors.T).tocoo()
    candidates=[]
    for left,right,value in zip(similarities.row,similarities.col,similarities.data):
        if left<right and value>=0.84:
            candidates.append({'left':rows[left]['example_id'],'right':rows[right]['example_id'],
                'similarity':round(float(value),4),'cross_split':rows[left]['split']!=rows[right]['split'],
                'left_text':rows[left]['text'],'right_text':rows[right]['text']})
    report['near_duplicate_candidates']=sorted(candidates,key=lambda r:-r['similarity'])
    report['labels_are_semantic_reference_not_clinical_measurement']=True
    report['held_out_labels_independently_validated']=False
    report['model_accuracy_measured_by_this_audit']=False
    output=ROOT/'results/chinese_dataset_audit.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    report=run()
    print(json.dumps({k:v for k,v in report.items() if k not in ('score_coverage','near_duplicate_candidates')},ensure_ascii=False,indent=2))
    print('near_duplicate_candidates:',len(report['near_duplicate_candidates']))
