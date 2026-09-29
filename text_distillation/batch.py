"""Batch inference reads category/text only and never changes the master."""

import csv
import json
from pathlib import Path

from .corpus import DATABASE,connect,dumps
from .contract import validate_input,CATEGORIES
from .artifacts import result_path


def read_input(path):
    path=Path(path)
    if path.suffix.lower()=='.jsonl':
        with path.open(encoding='utf-8-sig') as handle:
            rows=[json.loads(line) for line in handle if line.strip()]
    elif path.suffix.lower()=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as handle:
            reader=csv.DictReader(handle)
            header=reader.fieldnames
            if not header or len(set(header))!=len(header) or any(not k for k in header):
                raise ValueError('CSV requires unique nonblank columns.')
            rows=list(reader)
            if any(None in row or any(v is None for v in row.values()) for row in rows):
                raise ValueError('CSV row width differs from the header.')
    elif path.suffix.lower()=='.xlsx':
        from openpyxl import load_workbook
        workbook=load_workbook(path,read_only=True,data_only=True)
        try:
            values=iter(workbook.active.values)
            header=next(values,())
            if not header or len(set(header))!=len(header) or any(not isinstance(k,str) or not k for k in header):
                raise ValueError('Excel requires unique nonblank columns.')
            rows=[dict(zip(header,line)) for line in values if any(v is not None for v in line)]
        finally:
            workbook.close()
        if path.resolve()==(DATABASE.parent/'master_view.xlsx').resolve():
            from .chinese_data import display_examples
            with connect(DATABASE,readonly=True) as db:
                originals={r['view_id']:r for r in display_examples(db)}
            restored=[]
            for row in rows:
                original=originals.get(row.get('序号'))
                if original is None or original['text']!=row.get('文本') or CATEGORIES[original['category']][0]!=row.get('输入类别'):
                    raise ValueError('Master view differs from the corpus; regenerate with export.')
                restored.append({'record_id':original['example_id'],'category':original['category'],'text':original['text']})
            rows=restored
    else:
        raise ValueError('Use CSV, XLSX or JSONL.')
    result=[]
    seen=set()
    for row in rows:
        if not isinstance(row,dict):
            raise ValueError('Every input row must be an object.')
        category,text=validate_input(row.get('category',row.get('输入类别')),row.get('text',row.get('文本')))
        identifier=str(row.get('record_id',row.get('序号')) or '').strip()
        if not identifier or identifier in seen:
            raise ValueError('Every row needs a unique record_id.')
        seen.add(identifier)
        result.append({'record_id':identifier,'category':category,'text':text})
    if not result:
        raise ValueError('Input contains no rows.')
    return result


def run(args):
    import tempfile
    from .prediction import Predictor,ModelNotReady
    rows=read_input(args.input)
    output=result_path(args.output)
    predictor=Predictor(args.checkpoint)
    if not predictor.health()['ready']:
        raise ModelNotReady('请先完成正式训练。')
    with tempfile.TemporaryDirectory(prefix='batch-',dir=output.parent) as folder:
        pending=Path(folder)/'predictions.jsonl'
        with pending.open('w',encoding='utf-8') as handle:
            for row in rows:
                value={'record_id':row['record_id'],**predictor.predict(row['category'],row['text'])}
                handle.write(dumps(value)+'\n')
        pending.replace(output)
    return {'records':len(rows),'output':str(output),'master_dataset_modified':False}
