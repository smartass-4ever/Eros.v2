"""Serve the Bella surface: index.html + live state.json + Atom feed + webmention receiver."""
import http.server
import os
import socketserver

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("PORT", 8080))
FEED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feed.xml")


class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/feed.xml", "/feed", "/feed.xml/"):
            if os.path.exists(FEED_PATH):
                self.send_response(200)
                self.send_header("Content-Type", "application/atom+xml; charset=utf-8")
                self.end_headers()
                with open(FEED_PATH, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/webmention":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8", errors="replace")
            print(f"[webmention] received: {body[:200]}")
            self.send_response(202)
            self.end_headers()
            return
        self.send_response(405)
        self.end_headers()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    with socketserver.TCPServer(("0.0.0.0", PORT), H) as httpd:
        print(f"Bella surface live at http://0.0.0.0:{PORT}  (serving {os.getcwd()})")
        print(f"  /feed.xml   -> Atom feed (publishes when she forms real positions)")
        print(f"  /webmention -> receives incoming webmentions")
        httpd.serve_forever()
