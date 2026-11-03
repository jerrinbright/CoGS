"""Make build/artifact.html from index.html for the private claude.ai preview (the host supplies <html>/<head>/<body>).

    python3 tools/make_artifact.py

The page's img/, media/ and data/ files are published alongside it under the same relative paths
(see BUILD.md). No other changes are needed: all assets are standard web media types.
"""
import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
html = (SITE / "index.html").read_text()
head = re.search(r"<head>(.*?)</head>", html, re.S).group(1)
body = re.search(r"<body>(.*?)</body>", html, re.S).group(1)
head = re.sub(r'<meta charset[^>]*>\s*|<meta name="viewport"[^>]*>\s*', "", head)
out = SITE / "build" / "artifact.html"; out.parent.mkdir(exist_ok=True)
out.write_text(head.strip() + "\n" + body.strip() + "\n")
print("wrote", out, out.stat().st_size // 1024, "KB")
