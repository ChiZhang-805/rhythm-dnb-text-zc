"""The single master corpus and its white, single-sheet Excel view."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

ROOT=Path(__file__).resolve().parents[1]
DATABASE=ROOT/'dataset/master.sqlite'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def dumps(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))


@contextmanager
def connect(path=None, *, readonly=False):
    path=Path(path or DATABASE).resolve()
    if readonly:
        connection=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        connection=sqlite3.connect(path)
    connection.row_factory=sqlite3.Row
    connection.execute('PRAGMA foreign_keys=ON')
    try:
        yield connection
        if not readonly:
            connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize(db):
    """Keep source identities compatible with the preserved canonical database."""
    db.executescript('''
    CREATE TABLE IF NOT EXISTS records (
        record_id TEXT PRIMARY KEY, dataset TEXT NOT NULL, text TEXT NOT NULL,
        text_sha256 TEXT NOT NULL, split TEXT NOT NULL, group_id TEXT NOT NULL,
        subject_id TEXT, observed_at TEXT, context TEXT NOT NULL,
        native_labels TEXT NOT NULL, provenance TEXT NOT NULL, target TEXT,
        annotation_scope TEXT NOT NULL DEFAULT 'none', annotation_origin TEXT,
        exclusion_reason TEXT, duplicate_of TEXT REFERENCES records(record_id)
    );
    CREATE TABLE IF NOT EXISTS sources (name TEXT PRIMARY KEY,metadata TEXT NOT NULL);
    ''')


def insert(db,row):
    """Append real source material without rewriting a source or its partition."""
    from .chinese_data import active,sentence_candidate
    if active(db) and not sentence_candidate(row['text']):
        raise ValueError('Isolated words are not admitted to the active corpus.')
    text_hash=sha256(row['text'])
    previous=db.execute('SELECT text_sha256,split,group_id FROM records WHERE record_id=?',(row['record_id'],)).fetchone()
    if previous:
        if tuple(previous)!=(text_hash,row['split'],row['group_id']):
            raise ValueError('Source text or frozen partition changed.')
        return False
    names=('record_id','dataset','text','text_sha256','split','group_id','subject_id',
           'observed_at','context','native_labels','provenance','exclusion_reason')
    values={**row,'text_sha256':text_hash}
    for key in ('context','native_labels','provenance'):
        values[key]=dumps(row.get(key,{}))
    db.execute(f"INSERT INTO records ({','.join(names)}) VALUES ({','.join('?' for _ in names)})",
               [values.get(key) for key in names])
    return True


def summary(db):
    from .chinese_data import summary as chinese_summary
    return {'master_file':str(DATABASE),**chinese_summary(db)}


def build_view(db):
    from .chinese_data import build_view as chinese_view
    return chinese_view(db)


def export():
    from .artifacts import save_json
    with connect(readonly=True) as db:
        stats=summary(db)
        book=build_view(db)
    with tempfile.TemporaryDirectory(prefix='excel-export-',dir=DATABASE.parent) as folder:
        pending=Path(folder)/'master_view.xlsx'
        try:
            book.save(pending)
            pending.replace(DATABASE.parent/'master_view.xlsx')
        except PermissionError as error:
            raise PermissionError('请关闭 WPS/Excel 中的 master_view.xlsx；原文件未被删除。') from error
        finally:
            book.close()
    save_json(DATABASE.parent/'summary.json',stats)
    return stats
