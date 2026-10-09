from __future__ import annotations
import os, platform, shutil

def assess() -> dict:
    memory_kb = None
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemTotal:'): memory_kb=int(line.split()[1]); break
    except OSError: pass
    return {'os': platform.platform(), 'python': platform.python_version(), 'cpu_count': os.cpu_count(), 'memory_mb': round(memory_kb/1024) if memory_kb else None, 'gpu': 'not detected by dependency-free check', 'disk_free_gb': round(shutil.disk_usage('.').free/1024**3,2)}

def recommendation(info: dict) -> str:
    if info.get('memory_mb') and info['memory_mb'] < 4096: return 'Use the tiny CPU model, small batches, and synthetic datasets.'
    return 'Use the tiny CPU model first; scale only after held-out evaluation is reproducible.'
