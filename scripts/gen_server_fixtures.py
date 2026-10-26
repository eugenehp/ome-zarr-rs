"""Record the responses of ome-zarr-py's `view` / `finder` server (its
CORSRequestHandler: RangeHTTPServer + CORS) for a set of requests.

Run with ome-zarr-py installed: python scripts/gen_server_fixtures.py
Serves tests/fixtures/server_root and writes tests/fixtures/server.json.
"""

import http.client
import json
import os
import pathlib
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

from RangeHTTPServer import RangeRequestHandler

ROOT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SERVE = ROOT / "server_root"

# a small deterministic tree: an image group, a chunk, a csv, a nested dir
files = {
    "img.zarr/zarr.json": b'{"zarr_format": 3, "node_type": "group", "attributes": {}}',
    "img.zarr/s0/zarr.json": b'{"zarr_format": 3, "node_type": "array"}',
    "img.zarr/s0/c/0/0": bytes(range(256)) * 4,
    "img.zarr/.zattrs": b"{}",
    "biofile_finder.csv": b"File Path,File Name\nhttp://x,img.zarr\n",
    "dir with space/a b.txt": b"spaced",
    "empty.bin": b"",
}
for rel, data in files.items():
    p = SERVE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
for p in SERVE.rglob("*"):
    os.utime(p, (1700000000, 1700000000))


class CORSRequestHandler(RangeRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        SimpleHTTPRequestHandler.end_headers(self)

    def translate_path(self, path: str) -> str:
        self.directory = str(SERVE)
        return super().translate_path(path)

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", 0), CORSRequestHandler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()

REQUESTS = [
    ("GET", "/img.zarr/zarr.json", {}), ("HEAD", "/img.zarr/zarr.json", {}),
    ("GET", "/img.zarr/s0/c/0/0", {}), ("GET", "/img.zarr/s0/c/0/0", {"Range": "bytes=10-19"}),
    ("GET", "/img.zarr/s0/c/0/0", {"Range": "bytes=1000-"}), ("GET", "/img.zarr/s0/c/0/0", {"Range": "bytes=-5"}),
    ("GET", "/img.zarr/s0/c/0/0", {"Range": "bytes=5000-6000"}), ("GET", "/img.zarr/s0/c/0/0", {"Range": "bytes=20-10"}),
    ("GET", "/img.zarr/s0/c/0/0", {"Range": "bytes=0-2000"}), ("HEAD", "/img.zarr/s0/c/0/0", {"Range": "bytes=0-9"}),
    ("GET", "/img.zarr/.zattrs", {}), ("GET", "/biofile_finder.csv", {}), ("GET", "/empty.bin", {}),
    ("GET", "/empty.bin", {"Range": "bytes=0-1"}),
    ("GET", "/missing.json", {}), ("HEAD", "/missing.json", {}), ("GET", "/img.zarr", {}), ("GET", "/img.zarr/", {}),
    ("GET", "/img.zarr/?x=1", {}), ("GET", "/img.zarr/zarr.json?x=1", {}), ("GET", "/dir%20with%20space/a%20b.txt", {}),
    ("GET", "/../server.json", {}), ("GET", "/img.zarr/s0/../zarr.json", {}), ("OPTIONS", "/img.zarr/zarr.json", {}),
    ("POST", "/img.zarr/zarr.json", {}), ("GET", "/", {}),
]
out = []
for method, path, headers in REQUESTS:
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request(method, path, headers=headers)
    r = c.getresponse()
    body = r.read()
    out.append({"method": method, "path": path, "headers": headers, "status": r.status, "reason": r.reason,
                "response_headers": [[k, v] for k, v in r.getheaders() if k.lower() not in ("date", "server")],
                "server": r.getheader("Server"), "version": r.version,
                "body": body.decode("latin-1")})
    c.close()
server.shutdown()
(ROOT / "server.json").write_text(json.dumps(out, indent=1))
for o in out:
    print(o["method"], o["path"], o["headers"], o["status"], o["response_headers"], len(o["body"]))
