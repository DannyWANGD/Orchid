"""Serve the Orchid comparison page and one resident model on localhost."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from threading import Lock
from urllib.parse import urlsplit

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.generation import Sampling
from orchid.inference import load_poet
from orchid.io import ROOT, json_text
from orchid.metrics import judge_format_keywords, rhyme_status
from orchid.prosody_decode import RANKINGS


ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/examples.js": ("examples.js", "text/javascript; charset=utf-8"),
}


class DemoServer(ThreadingHTTPServer):
    def __init__(self, port, poet):
        self.poet = poet
        self.generation_lock = Lock()
        super().__init__(("127.0.0.1", port), DemoHandler)

    def generate(self, body):
        if not isinstance(body, dict):
            raise ValueError("请求必须是 JSON 对象。")
        words, seed, ranking = body.get("keywords"), body.get("seed"), body.get("ranking", "original")
        if (not isinstance(words, list) or len(words) > 3
                or any(not isinstance(word, str) for word in words)):
            raise ValueError("请输入一至三个不同的关键词。")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("种子须为 0 至 4294967295 的整数。")
        if ranking not in RANKINGS:
            raise ValueError("请选择页面提供的排序方式。")
        words = self.poet.task.validate_keywords(words)
        self.poet.tokenizer.encode("".join(words), strict=True)
        sampling = Sampling()
        pair = {"keywords": words, "seed": seed, "ranking": ranking}
        with self.generation_lock:
            table = self.poet.rhyme_decoder.table
            for mode in ("raw", "rhymed"):
                constrained = mode == "rhymed"
                result = self.poet.tokens(
                    words, sampling, seed, constrain_rhyme=constrained, constrain_keywords=constrained,
                    ranking=ranking if constrained else "original",
                )
                quality = judge_format_keywords(result["generated"], words)
                rhyme = rhyme_status(quality["lines"], table)
                pair[mode] = {
                    "generated": result["generated"], "seconds": result["seconds"],
                    "format_ok": quality["format_ok"], "keyword_hits": quality["keyword_hits"],
                    **rhyme,
                    "distinct_rhyme": rhyme["rhyme"] == "pass" and not rhyme["repeated_rhyme_char"],
                }
        return pair


class DemoHandler(BaseHTTPRequestHandler):
    def send_bytes(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, body):
        self.send_bytes(status, "application/json; charset=utf-8", json_text(body, indent=None).encode())

    def do_GET(self):
        request_path = urlsplit(self.path).path
        if request_path == "/api/status":
            self.send_json(200, {"device": str(self.server.poet.device)})
        elif request_path in ASSETS:
            name, content_type = ASSETS[request_path]
            self.send_bytes(200, content_type, (ROOT / "demo" / name).read_bytes())
        else:
            self.send_error(404)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/generate":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 4096 or self.headers.get_content_type() != "application/json":
                raise ValueError("请发送包含关键词和种子的 JSON。")
            body = json.loads(self.rfile.read(size))
            response = self.server.generate(body)
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
            return
        self.send_json(200, response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    torch.set_num_threads(2)
    poet = load_poet(args.checkpoint, args.device)
    with DemoServer(args.port, poet) as server:
        print(f"Orchid ({poet.device}) http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
