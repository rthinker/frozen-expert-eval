"""Standalone startup resolves the local code before loading data; nothing is fitted."""
import importlib.util
import sys
from pathlib import Path

import pytest

CR=Path(__file__).resolve().parents[1]
ROOT=CR.parent
RUNS=CR/'scripts/runs'
sys.path.insert(0,str(RUNS))
import _cr_paths


def test_default_setup_from_an_unrelated_working_directory(tmp_path,monkeypatch):
    monkeypatch.delenv('MITIGATION_CODEBASE',raising=False)
    monkeypatch.chdir(tmp_path)
    assert _cr_paths.setup()==ROOT


def test_an_invalid_environment_override_does_not_silently_fall_back(tmp_path,monkeypatch):
    monkeypatch.setenv('MITIGATION_CODEBASE',str(tmp_path/'missing'))
    with pytest.raises(SystemExit,match='not a code base root'):
        _cr_paths.setup()


def test_explicit_codebase_takes_precedence_over_environment(tmp_path,monkeypatch):
    monkeypatch.setenv('MITIGATION_CODEBASE',str(tmp_path/'missing'))
    assert _cr_paths.setup(ROOT)==ROOT


@pytest.mark.parametrize('name,extra',[
    ('11_disjoint_data_v4.py',[]),
    ('08_logo_extended_rungs_v3.py',[]),
    ('05_run_c_width_matched_v5.py',['--protocol','logo']),
])
def test_current_runners_reach_the_cache_loader_without_codebase(name,extra,tmp_path,monkeypatch):
    monkeypatch.delenv('MITIGATION_CODEBASE',raising=False)
    monkeypatch.chdir(tmp_path)
    spec=importlib.util.spec_from_file_location('standalone_'+name.replace('.','_'),RUNS/name)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import logo_common
    class ReachedLoader(Exception): pass
    def stop(*args,**kwargs): raise ReachedLoader
    def refuse_write(*args,**kwargs): raise AssertionError('dry run attempted a write')
    monkeypatch.setattr(logo_common,'load_generators',stop)
    monkeypatch.setattr(_cr_paths,'out_path',refuse_write)
    monkeypatch.setattr(_cr_paths,'make_logger',refuse_write)
    out=tmp_path/'outputs'
    monkeypatch.setattr(sys,'argv',[name,'--pool','arch6','--exclusion','none',
        '--out-dir',str(out),'--dry-run',*extra])
    with pytest.raises(ReachedLoader): module.main()
    assert not out.exists()
