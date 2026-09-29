"""Publish only allowlisted static assets and the public score contract, never research data."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from text_distillation.contract import schema

ASSETS=('index.html','app.js','style.css','favicon.svg')
GENERATED=('schema.json','config.json','.nojekyll')


def validate_api_url(value):
    value=value.strip() if isinstance(value,str) else ''
    if not value:
        return None
    parsed=urlsplit(value)
    if (parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or any(c.isspace() for c in value)):
        raise ValueError('A public API URL must be HTTPS without credentials, query or fragment.')
    if parsed.hostname in ('localhost','127.0.0.1','::1'):
        raise ValueError('A GitHub Pages site must not send private text to a loopback address.')
    return value.rstrip('/')+'/'


def build(destination=None,api_url=None):
    destination=Path(destination or ROOT/'build/github-pages').resolve()
    if not destination.is_relative_to((ROOT/'build').resolve()):
        raise ValueError('Static output must stay in this project build directory.')
    endpoint=validate_api_url(api_url)
    allowed=set(ASSETS+GENERATED)
    if destination.exists() and any(p.name not in allowed or not p.is_file() or p.is_symlink() for p in destination.iterdir()):
        raise ValueError('Unexpected files in the publish directory; refusing to deploy them.')
    destination.mkdir(parents=True,exist_ok=True)
    for name in ASSETS:
        source=ROOT/'web'/name
        if source.is_symlink():
            raise ValueError('Static assets cannot be symlinks.')
        shutil.copyfile(source,destination/name)
    (destination/'schema.json').write_text(json.dumps(schema(),ensure_ascii=False,indent=2),encoding='utf-8')
    (destination/'config.json').write_text(json.dumps({'mode':'static','api_base_url':endpoint},indent=2),encoding='utf-8')
    (destination/'.nojekyll').touch()
    return {'output':str(destination),'files':sorted(allowed),'api_configured':endpoint is not None,
            'dataset_included':False,'model_weights_included':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--api-url',default=os.environ.get('DNB_PUBLIC_API_URL',''))
    args=parser.parse_args()
    print(json.dumps(build(args.output,args.api_url),ensure_ascii=False,indent=2))
