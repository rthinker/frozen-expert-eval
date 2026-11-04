"""Paired sign tests discard ties while retaining all folds in descriptive summaries."""
import ast
import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binomtest, wilcoxon

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
SOURCES = ('authoring/full_ladder_stats_v13.py', 'authoring/full_ladder_stats_v4.py',
           'authoring/full_ladder_stats_standard_v3.py', 'runs/11_disjoint_data_v4.py',
           '../../scripts/11_disjoint_data.py')


def functions(source):
    # Execute the actual statistical functions without importing training dependencies or fitting models.
    tree = ast.parse((SCRIPTS/source).read_text(encoding='utf-8'))
    names = {'sign_test_p', 'sign_flip_p', 'fold_stats', 'logo_stats'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(np=np, binomtest=binomtest, wilcoxon=wilcoxon, itertools=itertools)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), source, 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('source', SOURCES)
@pytest.mark.parametrize('deltas, expected', [([-1]*4+[0]*4, 0.125), ([1]*4+[0]*4, 0.125),
                                             ([0]*8, 1.0), ([-1]*4+[1]*4, 1.0),
                                             ([-1]*8, 0.0078125)])
def test_exact_ties_are_excluded_and_all_ties_are_defined(source, deltas, expected):
    assert functions(source)['sign_test_p'](deltas) == expected


@pytest.mark.parametrize('source', SOURCES[:3])
@pytest.mark.parametrize('deltas', [[-1]*4+[0]*4, [0]*8])
def test_complete_fold_statistics_handle_ties(source, deltas):
    f = functions(source)
    if 'fold_stats' in f:
        result = f['fold_stats']({'test': deltas})['test']
        assert result['n'] == 8
    else:
        d = np.asarray(deltas, dtype=float)
        result = f['logo_stats'](d, d, np.zeros(8))['fs1']
    assert result['sign_p'] == (0.125 if any(deltas) else 1.0)
    assert result['wins'] == 0
    assert result['per_fold'] == deltas
    if not any(deltas):
        assert result['wilcoxon_p'] == result['signflip_p'] == 1.0
        assert result['ci95'] == [0.0, 0.0]


def test_runner_sign_test_does_not_depend_on_main_local_imports():
    source = SCRIPTS/'runs/11_disjoint_data_v4.py'
    node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'sign_test_p')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    assert namespace['sign_test_p']([0]*8) == 1.0
