from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageGrab

from .models import Element, Observation, now_iso


class ScreenObserver:
    """Capture a screen and optionally extract OCR elements.

    Optional desktop packages are imported lazily so this module remains usable in CI.
    """

    def __init__(self, output_dir: str | Path = "screenshots") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, use_ocr: bool = False) -> Observation:
        image = self._grab()
        path = self.output_dir / f"screen-{now_iso().replace(':', '').replace('+', '-')}.png"
        image.save(path)
        elements = tuple(self._ocr(image)) if use_ocr else ()
        return Observation(now_iso(), image.width, image.height, str(path), elements, self._fingerprint(image))

    def _grab(self) -> Image.Image:
        try:
            import mss
            from PIL import Image as PILImage
            with mss.mss() as sct:
                shot = sct.grab(sct.monitors[0])
                return PILImage.frombytes("RGB", shot.size, shot.rgb)
        except Exception:
            # ImageGrab works on Windows/macOS and is a useful fallback.
            return ImageGrab.grab()

    @staticmethod
    def _fingerprint(image: Image.Image) -> str:
        small = image.convert("L").resize((64, 64))
        return hashlib.sha256(small.tobytes()).hexdigest()

    @staticmethod
    def _ocr(image: Image.Image) -> list[Element]:
        try:
            import pytesseract
            data: dict[str, Any] = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except Exception:
            return []
        results: list[Element] = []
        for i, raw in enumerate(data.get("text", [])):
            text = raw.strip()
            if not text:
                continue
            results.append(Element(text, "text", int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i]), float(data["conf"][i]) / 100))
        return results


class ResultVerifier:
    @staticmethod
    def difference(before: str | Path, after: str | Path) -> float:
        first = Image.open(before).convert("RGB").resize((64, 64))
        second = Image.open(after).convert("RGB").resize((64, 64))
        diff = ImageChops.difference(first, second)
        values = list(diff.convert("L").getdata())
        return sum(values) / (len(values) * 255)

    @classmethod
    def changed(cls, before: str | Path, after: str | Path, threshold: float = 0.02) -> bool:
        return cls.difference(before, after) >= threshold
