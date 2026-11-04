"""Checks of the exact tests in the rebuilt Table A12 simulation (scripts/analyses/appF_power_mde_v2.py); no cache."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

CR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CR / "scripts" / "analyses"))
STATS = CR / "output" / "results" / "mitigation" / "mitigation_stats.json"


def _mde():
    spec = importlib.util.spec_from_file_location("appF_power_mde", CR / "scripts" / "analyses" / "appF_power_mde_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exact_bootstrap_resamples():
    m = _mde()
    assert m.COUNTS.shape == (6435, 8) and (m.COUNTS.sum(1) == 8).all()
    assert len({tuple(r) for r in m.COUNTS}) == 6435 and abs(m.PROBS.sum() - 1) < 1e-12


def test_rejection_regions_at_eight_folds():
    m = _mde()
    assert list(np.flatnonzero(m.SIGN_REJECT)) == [0, 8]           # 0 or 8 positive: p 0.008; 1 or 7: p 0.070
    assert list(np.flatnonzero(m.WILCOXON_REJECT)) == [0, 1, 2, 3]  # smaller rank sum 3: p 0.039; 4: p 0.055


def test_sizes_under_no_effect():
    m = _mde()
    z = np.random.default_rng(0).standard_normal((4000, 8))
    size = {t: float(m.detects(0.08 * z, t).mean()) for t in ("bootstrap_ci",) + m.OTHER_TESTS}
    assert size["sign_test"] < 0.02 and max(size["wilcoxon"], size["signflip"], size["t_test"]) < 0.07
    assert 0.07 < size["bootstrap_ci"] < 0.16                       # the percentile CI under-covers at n = 8


@pytest.mark.skipif(not STATS.exists(), reason="stored LOGO statistics not present")
def test_observed_arch6_logo_verdicts():
    """The paper's arch6 LOGO FS1 fold-deltas: CI above zero; sign test, Wilcoxon and sign-flip above 0.05."""
    m = _mde()
    d = np.array([json.loads(STATS.read_text(encoding="utf-8"))["logo_arch6"]["stats"]["fs1_full"]["per_fold"]])
    assert m.detects(d, "bootstrap_ci")[0]
    assert not any(m.detects(d, t)[0] for t in ("sign_test", "wilcoxon", "signflip"))
