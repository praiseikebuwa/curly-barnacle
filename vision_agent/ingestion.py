from __future__ import annotations
import hashlib, ipaddress, socket
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from PIL import Image
from .safety import is_public_host

class ImageIngestor:
    def __init__(self, root: str | Path = "data/images", max_bytes: int = 10_000_000):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True); self.max_bytes = max_bytes
    def _validate_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname: raise ValueError("Only HTTPS image URLs are allowed")
        try: addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)}
        except socket.gaierror as exc: raise ValueError("Could not resolve image host") from exc
        if not addresses or not all(is_public_host(a) for a in addresses): raise ValueError("Private, local, or metadata destinations are blocked")
        return url
    def import_url(self, url: str) -> Path:
        self._validate_url(url)
        ingestor = self
        class ValidatingRedirects(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                ingestor._validate_url(newurl)
                return super().redirect_request(req, fp, code, msg, headers, newurl)
        request = Request(url, headers={"User-Agent": "VisionOS-AI/0.3"})
        with build_opener(ValidatingRedirects).open(request, timeout=8) as response:
            data = response.read(self.max_bytes + 1)
            content_type = response.headers.get_content_type()
            if len(data) > self.max_bytes: raise ValueError("Image exceeds size limit")
            if content_type not in {"image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"}: raise ValueError("Unsupported image content type")
        return self._save_verified(data)
    def _save_verified(self, data: bytes) -> Path:
        digest = hashlib.sha256(data).hexdigest()[:24]; target = self.root / f"{digest}.image"; target.write_bytes(data)
        try:
            with Image.open(target) as image: image.verify()
        except Exception as exc:
            target.unlink(missing_ok=True); raise ValueError("Downloaded content is not a supported image") from exc
        return target
    def import_local(self, source: str | Path) -> Path:
        source = Path(source).resolve()
        if not source.is_file(): raise ValueError("Image does not exist")
        data = source.read_bytes()
        if len(data) > self.max_bytes: raise ValueError("Image exceeds size limit")
        return self._save_verified(data)
