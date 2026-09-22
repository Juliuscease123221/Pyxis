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

import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAPTURES = ROOT / "captures"
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
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    handler = partial(Handler, directory=str(ROOT))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {ROOT} on http://127.0.0.1:{port}")
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
