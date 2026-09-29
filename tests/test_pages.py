"""The public deployment never includes datasets, weights, credentials or local API requests."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_pages import ROOT,ASSETS,GENERATED,build,validate_api_url
from text_distillation.contract import schema


class PagesChecks(unittest.TestCase):
    def setUp(self):
        (ROOT/'build').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='pages-test-',dir=ROOT/'build')
        self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name)/'site'

    def test_only_public_assets_and_matching_contract_are_deployed(self):
        result=build(self.output)
        self.assertEqual(set(p.name for p in self.output.iterdir()),set(ASSETS+GENERATED))
        self.assertEqual(json.loads((self.output/'schema.json').read_text(encoding='utf-8')),schema())
        self.assertEqual(json.loads((self.output/'config.json').read_text()),{'mode':'static','api_base_url':None})
        self.assertFalse(result['dataset_included'])
        self.assertFalse(result['model_weights_included'])

    def test_github_project_subpaths_have_relative_asset_urls(self):
        build(self.output)
        html=(self.output/'index.html').read_text(encoding='utf-8')
        self.assertNotIn('href="/',html)
        self.assertNotIn('src="/',html)
        for name in ('favicon.svg','app.js','style.css'):
            self.assertIn('./'+name,html)

    def test_publish_rejects_contaminated_directories_and_wrong_destination(self):
        self.output.mkdir(parents=True)
        (self.output/'master.sqlite').write_bytes(b'isolated fixture')
        with self.assertRaises(ValueError):
            build(self.output)
        with self.assertRaises(ValueError):
            build(ROOT/'dataset')

    def test_api_url_rejects_http_credentials_and_loopback(self):
        for value in ('http://api.example.org','https://name:password@example.org',
                      'https://example.org?token=x','https://example.org#fragment','https://localhost:8765'):
            with self.assertRaises(ValueError):
                validate_api_url(value)
        self.assertIsNone(validate_api_url(''))
        self.assertEqual(validate_api_url('https://api.example.org/scoring'),'https://api.example.org/scoring/')


if __name__=='__main__':
    unittest.main()
