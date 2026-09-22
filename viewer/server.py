"""Static server for the viewer, plus a capture endpoint for the demo GIF.

`python -m http.server` is enough to run the viewer. The extra endpoint exists
only so the browser can hand rendered frames back to disk: the demo GIF has to
be the real renderer, not a matplotlib re-creation of it, and the only way to
get canvas pixels out of a page and into a file is to post them somewhere.

    POST /capture/<name>.png   body = raw PNG bytes

Frames land in viewer/captures/. Nothing else is writable.

    python -m viewer.server 8777
"""

from __future__ import annotations

import gzip
import io
import re
import sys
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
THROTTLE_KBPS = 0   # 0 = unlimited
CAPTURES = ROOT / "captures"
GZIP_MAGIC = bytes([0x1F, 0x8B])
SAFE = re.compile(r"^[A-Za-z0-9_.-]+\.png$")


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        if not self.path.startswith("/capture/"):
            self.send_error(404)
            return
        name = self.path[len("/capture/"):]
        if not SAFE.match(name):
            self.send_error(400, "bad frame name")
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 32 * 1024 * 1024:
            self.send_error(400, "bad length")
            return
        CAPTURES.mkdir(parents=True, exist_ok=True)
        (CAPTURES / name).write_bytes(self.rfile.read(length))
        self.send_response(204)
        self.end_headers()

    def send_head(self):
        """Gzip JSON responses.

        Any real static host does this, and without it the cold-load figure is
        measured against an artificially large manifest -- the JSON compresses
        by roughly 5x, which is the difference between a headline number that
        reflects deployment and one that reflects this dev server.
        """
        path = self.translate_path(self.path)
        accepts = "gzip" in self.headers.get("Accept-Encoding", "")

        # Tiles are written gzipped by viewer.prepare. Without the
        # Content-Encoding header the browser hands the raw deflate stream
        # to decodeTile, which fails on the magic-byte check.
        if path.endswith(".bin") and Path(path).is_file():
            raw = Path(path).read_bytes()
            if raw[:2] == GZIP_MAGIC:
                if not accepts:
                    raw = gzip.decompress(raw)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                if accepts:
                    self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                return io.BytesIO(raw)

        if accepts and path.endswith(".json") and Path(path).is_file():
            body = gzip.compress(Path(path).read_bytes(), 6)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return io.BytesIO(body)
        return super().send_head()

    def copyfile(self, source, outputfile):
        """Copy with an optional bandwidth cap.

        Cold-load numbers measured over localhost are not a claim about
        anything: the transfer is instant and the figure reduces to parse time.
        Throttling here lets the same measurement be taken against a stated
        bandwidth, which is reproducible in a way "it was fast on my machine"
        is not.
        """
        if not THROTTLE_KBPS:
            return super().copyfile(source, outputfile)
        chunk = max(1024, THROTTLE_KBPS * 1024 // 20)   # ~20 writes/second
        while True:
            block = source.read(chunk)
            if not block:
                break
            outputfile.write(block)
            time.sleep(len(block) / (THROTTLE_KBPS * 1024))

    def end_headers(self):
        # the viewer is a set of ES modules fetched from disk; no caching while
        # iterating on them
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "POST" not in (args[0] if args else ""):
            return
        super().log_message(fmt, *args)


def main() -> int:
    global THROTTLE_KBPS
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    if len(sys.argv) > 2:
        THROTTLE_KBPS = int(sys.argv[2])
        print(f"throttling to {THROTTLE_KBPS} KB/s")
    handler = partial(Handler, directory=str(ROOT))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {ROOT} on http://127.0.0.1:{port}")
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
