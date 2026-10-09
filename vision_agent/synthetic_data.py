from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

LABELS = ['button', 'text_field', 'panel']
def generate_dataset(root: str | Path = 'data/synthetic_ui', samples_per_class: int = 40, seed: int = 7) -> Path:
    root = Path(root); root.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(seed); rows=[]
    for label_id, label in enumerate(LABELS):
        for index in range(samples_per_class):
            image = Image.new('L', (16,16), color=20); draw = ImageDraw.Draw(image); x=int(rng.integers(1,4)); y=int(rng.integers(1,4)); w=int(rng.integers(8,13)); h=int(rng.integers(5,10))
            if label == 'button': draw.rounded_rectangle((x,y,x+w,y+h), radius=2, fill=220)
            elif label == 'text_field': draw.rectangle((x,y,x+w,y+h), outline=220, width=2); draw.line((x+2,y+h//2,x+w-2,y+h//2), fill=150, width=1)
            else: draw.rectangle((x,y,x+w,y+h), fill=100); draw.rectangle((x+2,y+2,x+w-2,y+3), fill=220); draw.rectangle((x+2,y+5,x+w-5,y+6), fill=180)
            filename=f'{label}_{index:04d}.png'; image.save(root/filename); rows.append({'path':filename,'label':label,'label_id':label_id,'source':'synthetic','seed':seed})
    (root/'manifest.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n', encoding='utf-8'); return root

def load_dataset(root: str | Path):
    root=Path(root); rows=[json.loads(line) for line in (root/'manifest.jsonl').read_text().splitlines() if line.strip()]; x=[]; y=[]
    for row in rows:
        with Image.open(root/row['path']) as image: x.append(np.asarray(image.convert('L'), dtype=np.float32).reshape(-1)/255.0); y.append(row['label_id'])
    return np.asarray(x), np.asarray(y, dtype=np.int64), rows
