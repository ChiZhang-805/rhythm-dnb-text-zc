"""Loopback-only scoring API and local UI; request text is never logged or stored."""

from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

from .contract import schema,validate_input
from .prediction import Predictor,ModelNotReady

WEB=Path(__file__).resolve().parents[1]/'web'


def create_server(port=8765,predictor=None):
    predictor=predictor or Predictor()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self,*args):
            pass

        def respond(self,status,value,content_type='application/json; charset=utf-8'):
            body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8') if 'json' in content_type else value
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('X-Frame-Options','DENY')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def trusted(self):
            hosts={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            host=self.headers.get('Host','')
            origin=self.headers.get('Origin')
            if host not in hosts or origin is not None and origin!='http://'+host:
                self.respond(403,{'error':'仅接受本机同源请求。'})
                return False
            return True

        def do_GET(self):
            if not self.trusted():
                return
            path=urlsplit(self.path).path
            if path in ('/api/schema','/schema.json'):
                self.respond(200,schema())
            elif path=='/config.json':
                self.respond(200,{'mode':'local','api_base_url':None})
            elif path=='/api/health':
                self.respond(200,predictor.health())
            elif path in ('/','/app.js','/style.css','/favicon.svg'):
                names={'/':('index.html','text/html'),'/app.js':('app.js','text/javascript'),
                       '/style.css':('style.css','text/css'),'/favicon.svg':('favicon.svg','image/svg+xml')}
                name,kind=names[path]
                self.respond(200,(WEB/name).read_bytes(),kind+'; charset=utf-8')
            else:
                self.respond(404,{'error':'Not found'})

        def do_POST(self):
            if not self.trusted():
                return
            if self.path!='/api/predict':
                self.respond(404,{'error':'Not found'})
                return
            if self.headers.get('Content-Type','').split(';')[0].strip()!='application/json':
                self.respond(415,{'error':'Use application/json.'})
                return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=50000:
                    self.respond(413,{'error':'请求为空或过大。'})
                    return
                value=json.loads(self.rfile.read(length))
                if not isinstance(value,dict) or set(value)!={'category','text'}:
                    raise ValueError('请求必须且只能包含 category 和 text。')
                category,text=validate_input(value['category'],value['text'])
                self.respond(200,predictor.predict(category,text))
            except ModelNotReady as error:
                self.respond(503,{'error':str(error)})
            except (ValueError,UnicodeError) as error:
                self.respond(400,{'error':str(error)})
            except Exception:
                self.respond(500,{'error':'模型加载或预测失败，请检查模型文件及服务环境。'})

    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def serve(port=8765,checkpoint=None,device='cpu'):
    server=create_server(port,Predictor(checkpoint,device))
    print(f'http://127.0.0.1:{server.server_port}',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
