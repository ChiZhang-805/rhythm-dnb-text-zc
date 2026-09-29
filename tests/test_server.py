"""HTTP contract tests use a marked fixture; production never fabricates scores."""

from http.client import HTTPConnection
import json
import threading
import unittest

from text_distillation.server import create_server
from text_distillation.contract import CATEGORIES
from text_distillation.prediction import ModelNotReady


class FixturePredictor:
    def __init__(self):
        self.ready=True

    def health(self):
        return {'ready':self.ready,'run_id':'isolated-fixture'}

    def predict(self,category,text):
        if not self.ready:
            raise ModelNotReady('Test fixture has no weights.')
        return {'category':category,'scores':{k:23 for k in CATEGORIES[category][1]}}


class ServerChecks(unittest.TestCase):
    def setUp(self):
        self.predictor=FixturePredictor()
        self.server=create_server(0,self.predictor)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self,path,body=None,headers=None,method=None):
        connection=HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        try:
            connection.request(method or ('POST' if body is not None else 'GET'),path,body=body,
                               headers=headers or {'Content-Type':'application/json'})
            response=connection.getresponse()
            return response.status,response.read(),dict(response.getheaders())
        finally:
            connection.close()

    def test_schema_and_static_page(self):
        status,body,headers=self.request('/api/schema')
        self.assertEqual(status,200)
        self.assertEqual(len(json.loads(body)['categories']),5)
        self.assertEqual(json.loads(self.request('/config.json')[1])['mode'],'local')
        self.assertEqual(json.loads(self.request('/schema.json')[1]),json.loads(body))
        for path in ('/','/style.css','/app.js','/favicon.svg'):
            status,body,headers=self.request(path)
            self.assertEqual(status,200)
            self.assertTrue(body)
            self.assertIn('Content-Security-Policy',headers)

    def test_category_routes_only_its_fields(self):
        for category,(_,keys) in CATEGORIES.items():
            status,body,_=self.request('/api/predict',json.dumps({'category':category,'text':'今天很累但还可以。'},ensure_ascii=False).encode())
            self.assertEqual(status,200)
            self.assertEqual(set(json.loads(body)['scores']),set(keys))

    def test_no_model_returns_503_without_scores(self):
        self.predictor.ready=False
        status,body,_=self.request('/api/predict',json.dumps({'category':'sleep','text':'昨晚睡得不好。'}).encode())
        self.assertEqual(status,503)
        self.assertNotIn('scores',json.loads(body))

    def test_bad_inputs_extra_targets_and_wrong_content_type(self):
        for value in ({'category':'wrong','text':'今天很累'}, {'category':'sleep','text':''},
                      {'category':'sleep','text':'今天很累','scores':[99]},[],None):
            status,_,_=self.request('/api/predict',json.dumps(value).encode())
            self.assertEqual(status,400)
        self.assertEqual(self.request('/api/predict',b'{broken')[0],400)
        self.assertEqual(self.request('/api/predict',b'{}',{'Content-Type':'text/plain'})[0],415)

    def test_cross_origin_dns_rebinding_and_path_traversal_blocked(self):
        self.assertEqual(self.request('/api/health',headers={'Host':'evil.example'})[0],403)
        self.assertEqual(self.request('/api/health',headers={'Origin':'https://evil.example'})[0],403)
        self.assertEqual(self.request('/../dataset/master.sqlite')[0],404)
        self.assertEqual(self.request('/api/predict',b'x'*50001)[0],413)


if __name__=='__main__':
    unittest.main()
