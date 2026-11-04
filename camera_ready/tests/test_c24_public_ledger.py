"""The public table build and harness comparator share one complete contract."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

CR = Path(__file__).resolve().parents[1]


def harness():
    spec = importlib.util.spec_from_file_location('public_c24_harness', CR/'scripts/review/reproduce_all_v4.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_rows(path, data):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


@pytest.fixture(scope='module')
def public_build(tmp_path_factory):
    output = tmp_path_factory.mktemp('public_build')
    result = subprocess.run([sys.executable, '-B', str(CR/'scripts/authoring/generate_camera_ready_tables_v18.py'),
                             '--inputs', str(CR/'inputs'), '--out', str(output)],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout+result.stderr
    return output


def test_the_real_builder_passes_the_harness_comparator(public_build):
    h = harness()
    rebuilt = public_build/'05_number_provenance.csv'
    assert rows(rebuilt) == rows(CR/'provenance/05_number_provenance.csv')
    result = h.compare_ledgers(new=rebuilt)
    assert set(result) == {'rows', 'same_display', 'environment_dependent', 'machine_dependent',
                           'different_display', 'missing', 'new_rows', 'invalid'}
    assert result['rows'] == result['same_display'] == 1937
    assert all(not value for key, value in result.items() if isinstance(value, list))
    locations = json.loads((public_build/'paper_table_locations.json').read_text(encoding='utf-8'))
    assert locations['printed_tables'] == len(list((public_build/'tables/paper').glob('*.tex'))) == 25
    assert h.check_table_deliverables(public_build) == []
    posted = h.compare_posted(public_build/'06_posted_number_check.md')
    assert posted['rows'] == posted['agree'] == 96
    assert posted['coverage_errors'] == posted['different'] == []


@pytest.mark.parametrize('fault', ['none', 'missing', 'invalid', 'changed', 'missing_ledger',
                                  'missing_posted', 'empty_posted', 'malformed_posted', 'missing_id',
                                  'duplicate_id', 'unknown_id', 'invalid_status', 'missing_table',
                                  'extra_table', 'empty_table', 'missing_map', 'wrong_map', 'stale_table'])
def test_finish_run_accepts_complete_build_and_rejects_public_data_faults(public_build, tmp_path, monkeypatch, fault):
    h = harness()
    output = tmp_path/'output/tables'
    shutil.copytree(public_build, output, copy_function=shutil.copyfile)
    data = rows(public_build/'05_number_provenance.csv')
    if fault == 'missing':
        data.pop(0)
    elif fault == 'invalid':
        data[0]['value'] = 'nan'
    elif fault == 'changed':
        data[0]['display'] = '0.000'
    ledger = output/'05_number_provenance.csv'
    write_rows(ledger, data)
    posted = output/'06_posted_number_check.md'
    table = output/'tables/paper/main.tex'
    location = output/'paper_table_locations.json'
    if fault == 'missing_ledger':
        ledger.unlink()
    elif fault == 'missing_posted':
        posted.unlink()
    elif fault in ('empty_posted', 'malformed_posted'):
        posted.write_text('' if fault == 'empty_posted' else 'All checks passed.', encoding='utf-8')
    elif fault in ('missing_id', 'duplicate_id', 'unknown_id', 'invalid_status'):
        lines = posted.read_text(encoding='utf-8').splitlines()
        idx = next(i for i, line in enumerate(lines) if h.POSTED_ROW.match(line))
        if fault == 'missing_id':
            lines.pop(idx)
        elif fault == 'duplicate_id':
            # Preserve 96 rows: a duplicate must not hide one absent quantity.
            lines[idx+1] = lines[idx]
        elif fault == 'unknown_id':
            lines[idx] = lines[idx].replace('`logo.fs1.delta`', '`unknown`')
        else:
            lines[idx] = lines[idx].replace('PASS', 'PASS_UNCHECKED')
        posted.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    elif fault == 'missing_table':
        table.unlink()
    elif fault == 'extra_table':
        (table.parent/'obsolete.tex').write_text('obsolete', encoding='utf-8')
    elif fault == 'empty_table':
        table.write_text('', encoding='utf-8')
    elif fault == 'missing_map':
        location.unlink()
    elif fault == 'wrong_map':
        obj = json.loads(location.read_text(encoding='utf-8'))
        obj['tables']['main']['current_name'] = 'Wrong table'
        location.write_text(json.dumps(obj), encoding='utf-8')
    elif fault == 'stale_table':
        import os
        os.utime(table, (347155200, 347155200))              # 1981: older than every input, valid on exFAT
    compare = h.compare_ledgers
    monkeypatch.setattr(h, 'ROOT', tmp_path)
    monkeypatch.setattr(h, 'CR', tmp_path)
    monkeypatch.setattr(h, 'compare_ledgers', lambda: compare(new=ledger))
    monkeypatch.setattr(h, 'mode_verify_only', lambda *args, **kwargs: ({}, 0))
    monkeypatch.setattr(h, 'timing_report', lambda *args, **kwargs: None)
    monkeypatch.setattr(h, 'RESULTS2', tmp_path/'results')
    run_dir = tmp_path/'run'
    run_dir.mkdir()
    record = {'run_id': 'public_test', 'run_start_epoch': time.time()-10,
              'steps': [{'id': 'C24', 'status': 'OK', 'elapsed_s': 0.1}]}
    code = h.finish_run(record, run_dir, None)
    assert code == (0 if fault == 'none' else 1)
    assert record['paper_numbers_ok'] is (fault == 'none')


def test_declared_products_and_posted_contract_cover_the_current_builder(public_build):
    import ast
    h = harness()
    steps = {s.id: s for s in h.steps_c()}
    assert any(str(p).endswith('.png') for p in steps['C23'].outputs)
    actual = {p.relative_to(public_build).as_posix() for p in public_build.rglob('*') if p.is_file()}
    assert set(h.table_deliverables()) == actual
    assert len(steps['C24'].outputs) == len(actual) == 52
    tree = ast.parse((CR/'scripts/authoring/generate_camera_ready_tables_v18.py').read_text(encoding='utf-8'))
    rows_ = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'POSTED' for t in node.targets))
    assert len(h.EXPECTED_POSTED_IDS) == len(set(h.EXPECTED_POSTED_IDS)) == 96
    assert set(h.EXPECTED_POSTED_IDS) == {row[0] for row in rows_}
