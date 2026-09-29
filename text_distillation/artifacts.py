"""Atomic experiment artifacts; never overwrite the dataset or a previous run."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from .corpus import ROOT


def save_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,pending=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as handle:
            json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        os.replace(pending,path)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def result_path(path):
    path=Path(path).resolve()
    if not path.is_relative_to((ROOT/'results').resolve()):
        raise ValueError('Output must stay inside this project results directory.')
    if path.exists():
        raise FileExistsError('Existing results are preserved; select a new output path.')
    path.parent.mkdir(parents=True,exist_ok=True)
    return path
