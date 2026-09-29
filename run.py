"""Single entry point for the Chinese category-conditioned scoring project."""

import argparse
import json
from pathlib import Path


def main():
    from text_distillation.corpus import ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('summary','export','verify','audit','test'):
        commands.add_parser(name)
    for name in ('prepare','train','smoke'):
        command=commands.add_parser(name)
        command.add_argument('--config',type=Path,default=ROOT/'train_config.json')
        if name=='prepare':
            command.add_argument('--offline',action='store_true')
    command=commands.add_parser('evaluate')
    command.add_argument('--split',choices=['validation','test'],default='test')
    command.add_argument('--checkpoint',type=Path)
    command=commands.add_parser('predict')
    command.add_argument('--category',required=True)
    command.add_argument('--text',required=True)
    command.add_argument('--checkpoint',type=Path)
    command.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    command=commands.add_parser('predict-file')
    command.add_argument('--input',required=True,type=Path)
    command.add_argument('--output',required=True,type=Path)
    command.add_argument('--checkpoint',type=Path)
    command=commands.add_parser('serve')
    command.add_argument('--port',type=int,default=8765)
    command.add_argument('--checkpoint',type=Path)
    command.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    command=commands.add_parser('import')
    command.add_argument('--input',required=True,type=Path)
    args=parser.parse_args()
    try:
        if args.command=='test':
            import unittest
            from text_distillation.artifacts import save_json
            outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
            result={'tests_run':outcome.testsRun,'failures':len(outcome.failures),'errors':len(outcome.errors),
                    'skipped':len(outcome.skipped),'passed':outcome.wasSuccessful()}
            save_json(ROOT/'results/macbert/unit_tests.json',result)
            if not outcome.wasSuccessful():
                raise SystemExit(1)
        elif args.command=='summary':
            from text_distillation.corpus import connect,summary
            with connect(readonly=True) as db:
                result=summary(db)
        elif args.command=='export':
            from text_distillation.corpus import export
            result=export()
        elif args.command=='verify':
            from text_distillation.chinese_data import verify
            result=verify()
        elif args.command=='audit':
            from text_distillation.chinese_audit import run
            result=run()
        elif args.command=='import':
            from text_distillation.chinese_data import import_authored
            result=import_authored(args.input)
        elif args.command in ('prepare','train','smoke'):
            from text_distillation import training
            function=getattr(training,args.command)
            result=function(args.config,download=not args.offline) if args.command=='prepare' else function(args.config)
        elif args.command=='evaluate':
            from text_distillation.prediction import evaluate
            result=evaluate(args.split,args.checkpoint)
        elif args.command=='predict':
            from text_distillation.prediction import Predictor
            result=Predictor(args.checkpoint,args.device).predict(args.category,args.text)
        elif args.command=='predict-file':
            from text_distillation.batch import run
            result=run(args)
        elif args.command=='serve':
            from text_distillation.server import serve
            serve(args.port,args.checkpoint,args.device)
            return
    except (ValueError,OSError,RuntimeError) as error:
        parser.exit(1,json.dumps({'error':str(error)},ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
