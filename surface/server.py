"""Serve the Bella surface: index.html + the live state.json she writes each cycle."""
import http.server, socketserver, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("PORT", 8080))


class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
    def log_message(self, *a):
        pass


if __name__ == "__main__":
    with socketserver.TCPServer(("0.0.0.0", PORT), H) as httpd:
        print(f"Bella surface live at http://0.0.0.0:{PORT}  (serving {os.getcwd()})")
        httpd.serve_forever()
