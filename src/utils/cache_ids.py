"""Read identifiers without executing downloaded pickle payloads.

Reviewed object arrays are matched by SHA-256 against the trusted code release
and replaced in memory by its verified plain-text row identifiers. Custom
identifier arrays must use NumPy's Unicode or UTF-8 byte-string dtype.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
IDS_ROOT = REPOSITORY / 'camera_ready/output/ids'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_sample_ids(path):
    path = Path(path)
    try:
        ids = np.load(path, allow_pickle=False)
    except ValueError as error:
        if 'Object arrays cannot be loaded' not in str(error):
            raise
        # Neither the cache's manifest nor any pickle supplied by it is trusted.
        manifest = json.loads((IDS_ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
        entry = manifest['splits'].get(path.parent.name)
        if path.name != 'sample_ids.npy' or entry is None or sha256(path) != entry['source_sha256']:
            raise ValueError('Unverified object identifier array; use the pinned reviewed cache or '
                             'a non-pickled Unicode identifier array') from error
        text = IDS_ROOT/f'{path.parent.name}.txt'
        if sha256(text) != entry['txt_sha256']:
            raise ValueError('Trusted plain-text identifier file has changed')
        values = text.read_text(encoding='utf-8').splitlines()
        if len(values) != entry['rows']:
            raise ValueError('Plain-text identifier row count differs from the trusted manifest')
        ids = np.asarray(values, dtype=object)
    if ids.ndim != 1 or (ids.dtype.kind not in ('U', 'S', 'O')):
        raise ValueError('Expected one-dimensional string identifiers')
    if ids.dtype.kind == 'S':
        return np.asarray([value.decode('utf-8') for value in ids], dtype=object)
    return np.asarray([str(value) for value in ids], dtype=object)
