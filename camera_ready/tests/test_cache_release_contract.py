"""Cache validation must separate immutable data from the current notice."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/review/validate_cache_release.py'
spec = importlib.util.spec_from_file_location('cache_release_validator', SCRIPT)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


def bind_contract(folder):
    contract = {'schema_version': 1}
    for key, name in [('data_manifest', 'CHECKSUMS.txt'), ('current_notice', 'LICENSE_NOTICE.txt')]:
        data = (folder/name).read_bytes()
        contract[key] = {'path': name, 'sha256': digest(data), 'size_bytes': len(data)}
    contract['data_manifest']['entries'] = 2
    write_json(folder/'CACHE_PROVENANCE.json', contract)
    trusted = folder / 'trusted_code_metadata'
    trusted.mkdir(exist_ok=True)
    write_json(trusted/'CACHE_PROVENANCE.json', contract)


@pytest.fixture
def cache(tmp_path):
    files = {'FIT/array.bin': b'unchanged numerical data', 'FIT/ids.txt': b'row-0\n'}
    for rel, data in files.items():
        path = tmp_path/rel
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
    (tmp_path/'CHECKSUMS.txt').write_text(''.join(f'{digest(data)}  {rel}\n' for rel, data in files.items()), encoding='utf-8')
    (tmp_path/'LICENSE_NOTICE.txt').write_text('Current notice: CC BY-NC-SA 4.0; DF40 attribution.', encoding='utf-8')
    (tmp_path/'DATASET_CARD.md').write_text('\n'.join('## '+section for section in validator.REQUIRED_CARD_SECTIONS), encoding='utf-8')
    doc = {key: 'fixture' for key in validator.REQUIRED_CROISSANT_FIELDS}
    doc.update({'@context': {}, '@type': 'sc:Dataset', 'conformsTo': 'http://mlcommons.org/croissant/1.0',
                'distribution': [{'@id': rel, 'sha256': digest(data), 'contentSize': str(len(data))} for rel, data in files.items()]})
    write_json(tmp_path/'croissant.json', doc)
    bind_contract(tmp_path)
    return tmp_path


def invoke(cache):
    return subprocess.run([sys.executable, '-B', '-c',
                           'import importlib.util,sys; from pathlib import Path; '
                           's=importlib.util.spec_from_file_location("validator",sys.argv[1]); '
                           'v=importlib.util.module_from_spec(s); s.loader.exec_module(v); '
                           'v.TRUSTED_METADATA=Path(sys.argv[2])/"trusted_code_metadata"; '
                           'sys.argv=[sys.argv[1],"--cache-root",sys.argv[2]]; sys.exit(v.main())',
                           str(SCRIPT), str(cache)],
                          capture_output=True, text=True, timeout=30)


def test_complete_cache_with_separate_current_notice_passes(cache):
    result = invoke(cache)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'VALIDATION PASSED: 2 files, 2 Croissant entries' in result.stdout


@pytest.mark.parametrize('target', ['FIT/array.bin', 'FIT/ids.txt', 'LICENSE_NOTICE.txt', 'CHECKSUMS.txt'])
def test_alteration_is_rejected(cache, target):
    with (cache/target).open('ab') as stream:
        stream.write(b'altered')
    result = invoke(cache)
    assert result.returncode == 1 and 'VALIDATION FAILED' in result.stderr


def test_stale_manifest_cannot_require_a_second_notice(cache):
    notice = cache/'LICENSE_NOTICE.txt'
    with (cache/'CHECKSUMS.txt').open('a', encoding='utf-8') as stream:
        stream.write(f'{digest(b"old notice")}  LICENSE_NOTICE.txt\n')
    result = invoke(cache)
    assert result.returncode == 1 and 'CHECKSUMS.txt does not match' in result.stderr
    assert notice.read_text().startswith('Current notice')


def test_previous_notice_cannot_replace_current_notice(cache):
    (cache/'LICENSE_NOTICE.txt').write_text('DF40 previous notice', encoding='utf-8')
    result = invoke(cache)
    assert result.returncode == 1 and 'LICENSE_NOTICE.txt does not match' in result.stderr


@pytest.mark.parametrize('target', ['CACHE_PROVENANCE.json', 'LICENSE_NOTICE.txt', 'FIT/array.bin'])
def test_missing_required_file_is_rejected(cache, target):
    (cache/target).unlink()
    assert invoke(cache).returncode == 1


@pytest.mark.parametrize('mutation', ['duplicate', 'size', 'extra_notice'])
def test_croissant_cannot_hide_or_misdescribe_data(cache, mutation):
    path = cache/'croissant.json'
    doc = json.loads(path.read_text())
    if mutation == 'duplicate':
        doc['distribution'][1] = doc['distribution'][0].copy()
    elif mutation == 'size':
        doc['distribution'][0]['contentSize'] = '0'
    else:
        doc['distribution'].append({'@id': 'LICENSE_NOTICE.txt', 'sha256': '0'*64})
    write_json(path, doc)
    assert invoke(cache).returncode == 1


@pytest.mark.parametrize(('line', 'diagnostic'), [
    (f"{'0'*64}  ../escape.bin\n", 'unsafe CHECKSUMS path'),
    (f"{'0'*64}  C:/escape.bin\n", 'unsafe CHECKSUMS path'),
    ('broken  FIT/array.bin\n', 'malformed SHA-256'),
])
def test_malformed_or_escaping_manifest_entry_is_rejected(cache, line, diagnostic):
    (cache/'CHECKSUMS.txt').write_text(line, encoding='utf-8')
    bind_contract(cache)
    result = invoke(cache)
    assert result.returncode == 1 and diagnostic in result.stderr, result.stdout + result.stderr


def test_duplicate_manifest_entry_is_rejected(cache):
    path = cache/'CHECKSUMS.txt'
    first = path.read_text().splitlines(True)[0]
    path.write_text(first+first, encoding='utf-8')
    bind_contract(cache)
    result = invoke(cache)
    assert result.returncode == 1 and 'duplicate CHECKSUMS path' in result.stderr, result.stdout + result.stderr
