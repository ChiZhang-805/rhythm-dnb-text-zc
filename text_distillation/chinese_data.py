"""Import individually authored Chinese examples into the one master database."""

from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .corpus import ROOT,DATABASE,connect,dumps,sha256,read_json
from .contract import CATEGORIES,METRICS

VIEW_METRICS = tuple(next(m for m in METRICS if m[0]==key)
                     for _,keys in CATEGORIES.values() for key in keys)


def han_count(text):
    return len(re.findall(r'[\u3400-\u9fff]',text))


def sentence_candidate(text):
    """Reject obvious isolated words; semantic completeness is reviewed during authoring."""
    if not isinstance(text,str):
        return False
    content=''.join(re.findall(r'[\u3400-\u9fffA-Za-z0-9]',text))
    isolated={'开心','高兴','悲伤','焦虑','烦躁','压力','情绪','饮食','睡眠','社交',
              '孤独','疲劳','失眠','难过','快乐','愤怒','恐惧','抑郁','满意','一般',
              '不开心','不高兴','非常开心','非常难过','非常焦虑','非常疲劳'}
    if han_count(text):
        return han_count(text)>1 and content not in isolated
    return len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*",text))>1


def text_identity(text):
    return ''.join(re.findall(r'[\u3400-\u9fffA-Za-z0-9]',text)).casefold()


def initialize(db):
    db.executescript('''
    CREATE TABLE IF NOT EXISTS chinese_examples (
        serial INTEGER PRIMARY KEY AUTOINCREMENT, example_id TEXT UNIQUE NOT NULL,
        source_record_id TEXT REFERENCES records(record_id), source_sha256 TEXT,
        category TEXT NOT NULL, text TEXT NOT NULL, scores TEXT NOT NULL,
        origin TEXT NOT NULL, split TEXT NOT NULL, group_id TEXT NOT NULL,
        CHECK(origin IN ('authored_simulation','translated_source')),
        CHECK(split IN ('train','validation','test'))
    );
    CREATE TABLE IF NOT EXISTS corpus_policy (key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE UNIQUE INDEX IF NOT EXISTS chinese_one_category_per_source
        ON chinese_examples(source_record_id) WHERE source_record_id IS NOT NULL;
    CREATE TABLE IF NOT EXISTS chinese_score_history (
        change_id INTEGER PRIMARY KEY, example_id TEXT NOT NULL REFERENCES chinese_examples(example_id),
        text_sha256 TEXT NOT NULL, previous_scores TEXT NOT NULL, scores TEXT NOT NULL,
        judgments TEXT NOT NULL, method TEXT NOT NULL,
        changed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS source_restorations (
        record_id TEXT PRIMARY KEY REFERENCES records(record_id),
        snapshot_file TEXT NOT NULL, snapshot_sha256 TEXT NOT NULL,
        source_text_sha256 TEXT NOT NULL,
        restored_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS chinese_revision_history (
        change_id INTEGER PRIMARY KEY, example_id TEXT NOT NULL REFERENCES chinese_examples(example_id),
        previous_payload TEXT NOT NULL, payload TEXT NOT NULL, reason TEXT NOT NULL,
        method TEXT NOT NULL, changed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    ''')


def validate(category,text,scores,origin):
    if (not isinstance(category,str) or category not in CATEGORIES
            or not sentence_candidate(text) or not han_count(text)):
        raise ValueError('Each example needs one known category and meaningful Chinese text, not an isolated word.')
    keys=CATEGORIES[category][1]
    if not isinstance(scores,list) or len(scores)!=len(keys):
        raise ValueError('Provide exactly the fixed output positions for the input category.')
    if any(not(type(v) is int and 0<=v<=100) for v in scores):
        raise ValueError('Every required output must be an integer 0-100; no missing scores.')
    return dict(zip(keys,scores))


def import_authored(path, *, database=DATABASE):
    """Only validate and persist literal text/score decisions; no translation or scoring rules."""
    packet=read_json(path)
    if set(packet)!={'method','synthetic','translations'} or packet['method']!='individually_authored':
        raise ValueError('Use the individually authored packet format.')
    with connect(database) as db:
        initialize(db)
        seen={text_identity(r['text']):r['example_id'] for r in db.execute('SELECT example_id,text FROM chinese_examples')}
        for origin,items in (('authored_simulation',packet['synthetic']),('translated_source',packet['translations'])):
            for item in items:
                key,category,text,scores=item
                values=validate(category,text,scores,origin)
                if origin=='translated_source':
                    source=db.execute('SELECT * FROM records WHERE record_id=?',(key,)).fetchone()
                    if source is None or source['duplicate_of'] or source['exclusion_reason']:
                        raise ValueError('A translation must link to an eligible preserved real source: '+key)
                    source_id,source_hash,split,group=key,source['text_sha256'],source['split'],source['group_id']
                    previous_category=db.execute('SELECT category FROM chinese_examples WHERE source_record_id=?',(key,)).fetchone()
                    if previous_category and previous_category[0]!=category:
                        raise ValueError('A source sample has exactly one category: '+key)
                else:
                    source_id,source_hash,split,group=None,None,'train','authored:'+key
                example_id='zh:'+origin+':'+key+':'+category
                normalized=text_identity(text)
                if normalized in seen and seen[normalized]!=example_id:
                    raise ValueError('Repeated text, including punctuation-only variants: '+key)
                seen[normalized]=example_id
                payload=(source_id,source_hash,category,text,dumps(values),origin,split,group)
                prior=db.execute('SELECT source_record_id,source_sha256,category,text,scores,origin,split,group_id FROM chinese_examples WHERE example_id=?',(example_id,)).fetchone()
                if prior and tuple(prior)!=payload:
                    raise ValueError('Existing authored example differs; review changes explicitly: '+key)
                db.execute('''INSERT OR IGNORE INTO chinese_examples
                    (example_id,source_record_id,source_sha256,category,text,scores,origin,split,group_id)
                    VALUES (?,?,?,?,?,?,?,?,?)''',(example_id,*payload))
        db.execute("INSERT OR REPLACE INTO corpus_policy VALUES ('active_task','chinese_category_degree')")
        db.execute("DELETE FROM corpus_policy WHERE key='long_text_only'")
        db.execute("INSERT OR REPLACE INTO corpus_policy VALUES ('text_admission','meaningful_sentence_no_length_minimum')")
        return summary(db)


def restore_reviewed_sources(path, *, database=DATABASE, snapshot=None):
    """Recover only individually selected translations; do not bulk-restore a word backlog."""
    packet=read_json(path)
    if set(packet)!={'method','synthetic','translations'} or packet['method']!='individually_authored':
        raise ValueError('Use the individually authored packet format.')
    snapshot=Path(snapshot or ROOT/'archive/before_chinese_long_text/master.sqlite').resolve()
    if not snapshot.is_file() or snapshot==Path(database).resolve():
        raise ValueError('A distinct existing recovery snapshot is required.')
    with snapshot.open('rb') as handle:
        digest=hashlib.file_digest(handle,'sha256').hexdigest()
    with closing(sqlite3.connect(snapshot.as_uri()+'?mode=ro',uri=True)) as recovery, connect(database) as db:
        recovery.row_factory=sqlite3.Row
        initialize(db)
        restored=[]
        for key,category,text,scores in packet['translations']:
            validate(category,text,scores,'translated_source')
            row=recovery.execute('SELECT * FROM records WHERE record_id=?',(key,)).fetchone()
            if (row is None or row['duplicate_of'] or row['exclusion_reason']
                    or row['split'] not in ('train','validation','test') or not sentence_candidate(row['text'])):
                raise ValueError('Source is not eligible for reviewed recovery: '+key)
            prior=db.execute('SELECT text_sha256,split,group_id FROM records WHERE record_id=?',(key,)).fetchone()
            identity=(row['text_sha256'],row['split'],row['group_id'])
            if prior:
                if tuple(prior)!=identity:
                    raise ValueError('Preserved source identity differs from snapshot: '+key)
                continue
            names=row.keys()
            db.execute(f"INSERT INTO records ({','.join(names)}) VALUES ({','.join('?' for _ in names)})",tuple(row))
            db.execute('INSERT INTO source_restorations (record_id,snapshot_file,snapshot_sha256,source_text_sha256) VALUES (?,?,?,?)',
                       (key,str(snapshot),digest,row['text_sha256']))
            restored.append(key)
        if db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Recovery broke source links.')
        return {'restored':len(restored),'record_ids':restored,'snapshot_sha256':digest}


def active(db):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='corpus_policy'").fetchone():
        return False
    return bool(db.execute("SELECT 1 FROM corpus_policy WHERE key='active_task' AND value='chinese_category_degree'").fetchone())


def review_fidelity(path, *, database=DATABASE):
    """Apply explicit fidelity corrections without changing source identities or partitions."""
    packet=read_json(path)
    if packet.get('method')!='individual_fidelity_review':
        raise ValueError('Expected individually adjudicated fidelity corrections.')
    with connect(database) as db:
        initialize(db)
        changed=0
        for item in packet['items']:
            row=db.execute('SELECT * FROM chinese_examples WHERE source_record_id=?',(item['source_record_id'],)).fetchone()
            if row is None or row['category']!=item['category']:
                raise ValueError('Fidelity review cannot create a sample or change its category.')
            values=validate(row['category'],item['text'],item['scores'],row['origin'])
            before=dumps([row['text'],json.loads(row['scores'])])
            after=dumps([item['text'],values])
            if before==after:
                continue
            if sha256(before)!=item['previous_payload_sha256'] or not item['reason'].strip():
                raise ValueError('Stale or unexplained fidelity revision.')
            if any(text_identity(r['text'])==text_identity(item['text']) for r in
                   db.execute('SELECT text FROM chinese_examples WHERE example_id!=?',(row['example_id'],))):
                raise ValueError('Fidelity revision would duplicate another text.')
            db.execute('INSERT INTO chinese_revision_history (example_id,previous_payload,payload,reason,method) VALUES (?,?,?,?,?)',
                       (row['example_id'],before,after,item['reason'],packet['method']))
            db.execute('UPDATE chinese_examples SET text=?,scores=? WHERE example_id=?',
                       (item['text'],dumps(values),row['example_id']))
            changed+=1
        return {'changed_records':changed,**summary(db)}


def examples(db):
    return [{**dict(r),'scores':json.loads(r['scores'])} for r in db.execute('SELECT * FROM chinese_examples ORDER BY serial')]


def display_examples(db):
    """Sort the view only; database identities and frozen partitions never change."""
    order={name:index for index,name in enumerate(CATEGORIES)}
    rows=sorted(examples(db),key=lambda r:(order[r['category']],r['serial']))
    return [{**row,'view_id':index} for index,row in enumerate(rows,1)]


def model_pair(row):
    """Only category and text enter the model; provenance and split are never targets."""
    keys=CATEGORIES[row['category']][1]
    if set(row['scores'])!=set(keys):
        raise ValueError('Output keys must equal the specified category contract.')
    validate(row['category'],row['text'],[row['scores'][k] for k in keys],row['origin'])
    return {'category':row['category'],'text':row['text']},{k:row['scores'][k] for k in keys}


def validate_prediction(category,value):
    if not isinstance(category,str) or category not in CATEGORIES or not isinstance(value,dict) or set(value)!=set(CATEGORIES[category][1]):
        raise ValueError('Return all and only this category\'s fixed score fields.')
    if any(not(type(v) is int and 0<=v<=100) for v in value.values()):
        raise ValueError('Every predicted score must be an integer 0-100.')
    return value


def review_scores(path, *, database=DATABASE):
    """Persist literal, individually reviewed numbers, with their inference limits."""
    packet=read_json(path)
    if set(packet)!={'method','items'} or packet['method']!='individual_semantic_review':
        raise ValueError('Expected individual semantic review decisions.')
    with connect(database) as db:
        initialize(db)
        changed=0
        seen=set()
        for item in packet['items']:
            if set(item)!={'source_record_id','category','scores','judgments'}:
                raise ValueError('Unexpected review fields.')
            identifier='zh:translated_source:'+item['source_record_id']+':'+item['category']
            if identifier in seen:
                raise ValueError('Repeated example in review packet.')
            seen.add(identifier)
            row=db.execute('SELECT * FROM chinese_examples WHERE example_id=?',(identifier,)).fetchone()
            if row is None:
                raise ValueError('Read and translate the real source before reviewing its scores.')
            values=validate(row['category'],row['text'],item['scores'],row['origin'])
            before=json.loads(row['scores'])
            judgments=item['judgments']
            if before==values:
                previous=db.execute('SELECT judgments FROM chinese_score_history WHERE example_id=? ORDER BY change_id DESC LIMIT 1',(identifier,)).fetchone()
                if previous and json.loads(previous[0])==judgments:
                    continue
                raise ValueError('A changed rationale requires an explicit new review.')
            if set(judgments)!={k for k in values if before[k]!=values[k]}:
                raise ValueError('Every revised score requires an individual rationale.')
            for judgment in judgments.values():
                if (set(judgment)!={'basis','reason'} or judgment['basis'] not in ('contextual_estimate','weak_estimate')
                        or not isinstance(judgment['reason'],str) or not judgment['reason'].strip()):
                    raise ValueError('Record semantic reasoning and distinguish weak estimates from explicit facts.')
            db.execute('''INSERT INTO chinese_score_history
                (example_id,text_sha256,previous_scores,scores,judgments,method) VALUES (?,?,?,?,?,?)''',
                (identifier,sha256(row['text']),row['scores'],dumps(values),dumps(judgments),packet['method']))
            db.execute('UPDATE chinese_examples SET scores=? WHERE example_id=?',(dumps(values),identifier))
            changed+=1
        return {'changed_records':changed,**summary(db)}


def summary(db):
    rows=examples(db)
    pending=db.execute('''SELECT COUNT(*) FROM records r WHERE NOT EXISTS
        (SELECT 1 FROM chinese_examples e WHERE e.source_record_id=r.record_id)''').fetchone()[0]
    return {'task':'chinese_category_degree','active_examples':len(rows),
        'fully_scored_examples':sum(all(v is not None for v in r['scores'].values()) for r in rows),
        'partially_scored_examples':sum(any(v is None for v in r['scores'].values()) for r in rows),
        'by_category':dict(Counter(CATEGORIES[r['category']][0] for r in rows)),
        'by_origin':dict(Counter(r['origin'] for r in rows)),
        'by_split':dict(Counter(r['split'] for r in rows)),
        'numeric_targets':sum(type(v) is int for r in rows for v in r['scores'].values()),
        'min_han_characters':min((han_count(r['text']) for r in rows),default=0),
        'untranslated_source_backlog':pending,'old_model_evaluates_this_dataset':False}


def verify(*,database=DATABASE):
    if not Path(database).is_file():
        raise FileNotFoundError('The private dataset/master.sqlite is not included in the code repository.')
    with connect(database,readonly=True) as db:
        if not active(db):
            raise ValueError('Chinese dataset is not active.')
        rows=examples(db)
        seen={}
        text_ids={}
        for row in rows:
            validate(row['category'],row['text'],[row['scores'][k] for k in CATEGORIES[row['category']][1]],row['origin'])
            model_pair(row)
            normalized=text_identity(row['text'])
            if normalized in text_ids:
                raise ValueError('Repeated Chinese text: '+row['example_id'])
            text_ids[normalized]=row['example_id']
            if row['origin']=='authored_simulation' and row['split']!='train':
                raise ValueError('Authored simulations cannot be reported as real held-out tests.')
            if row['source_record_id']:
                source=db.execute('SELECT text_sha256,split,group_id FROM records WHERE record_id=?',(row['source_record_id'],)).fetchone()
                if tuple(source)!=(row['source_sha256'],row['split'],row['group_id']):
                    raise ValueError('Translation has changed the source identity or frozen partition.')
            for key in (row['group_id'],sha256(row['text'])):
                if key in seen and seen[key]!=row['split']:
                    raise ValueError('Same group or text occurs across partitions.')
                seen[key]=row['split']
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok' or db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Master database integrity failed.')
        return {'passed':True,**summary(db)}


def build_view(db):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment,Border,Font,PatternFill,Side
    from openpyxl.utils import get_column_letter
    book=Workbook()
    sheet=book.active
    sheet.title='数据'
    sheet.append(['序号','文本','输入类别']+[m[2] for m in VIEW_METRICS])
    for row in display_examples(db):
        model_pair(row)
        values=[row['scores'].get(m[0]) for m in VIEW_METRICS]
        sheet.append([row['view_id'],row['text'],CATEGORIES[row['category']][0]]+values)
    white=PatternFill('solid',fgColor='FFFFFF')
    grid=Side(style='hair',color='D5D5D5')
    for cells in sheet:
        for cell in cells:
            cell.fill=white
            cell.font=Font(name='Microsoft YaHei',size=11,color='000000',bold=cell.row==1)
            cell.border=Border(bottom=grid,right=grid)
    for cell in sheet[1]:
        cell.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
    for cells in sheet.iter_rows(min_row=2):
        sheet.row_dimensions[cells[0].row].height=min(240,max(42,24+len(cells[1].value)/2.8))
        for cell in cells:
            if isinstance(cell.value,str):
                cell.data_type='s'
            cell.alignment=Alignment(wrap_text=True,vertical='top' if cell.column==2 else 'center',horizontal='left' if cell.column==2 else 'center')
            if cell.column>=4:
                cell.number_format='0'
    for col in range(1,21):
        sheet.column_dimensions[get_column_letter(col)].width={1:8,2:88,3:12}.get(col,14)
    sheet.freeze_panes='D2'
    sheet.auto_filter.ref=sheet.dimensions
    sheet.row_dimensions[1].height=48
    sheet.sheet_view.showGridLines=False
    sheet.sheet_view.zoomScale=85
    sheet.print_title_rows='1:1'
    return book
