"""Check that analysed LOGO runs have compatible partitions and complete prediction-space ladders.

Checks use membership fingerprints or trusted pairs for the supplied runs. Tests use supplied
results and synthetic runs.
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

CR = Path(__file__).resolve().parents[1]
ROOT = CR.parent
CD = CR / "output" / "results" / "mitigation" / "content_disjoint"
MECH = CR / "output" / "results" / "mitigation" / "mechanism"
SENS = CR / "output" / "results" / "sensitivity"
DEDUP = SENS / "dedup"
POOLS = {"arch6": 256, "clip4": 32, "strong4": 256, "weak6": 256}
GENS = ["whichisreal", "CollabDiff", "MidJourney", "heygen", "starganv2", "deepfacelab", "stargan", "styleclip"]
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
sys.path.insert(0, str(CR / "scripts" / "runs"))
import ladder_checks as lc  # noqa: E402


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _arch6():
    lp, rp = CD / "11_disjoint_arch6_d256.json", CD / "A3c_logo_extended_rungs_arch6.json"
    return lp, rp, load(lp), load(rp)


def test_the_stored_ladders_are_complete():
    for pool, d in POOLS.items():
        for base in (CD, DEDUP):
            logo, rungs = load(base / f"11_disjoint_{pool}_d{d}.json"), load(base / f"A3c_logo_extended_rungs_{pool}.json")
            assert lc.ladder_gaps(logo, rungs) == [], (base.name, pool)


def test_a_missing_base_method_and_a_wrong_recorded_best_are_gaps():
    _, _, logo, rungs = _arch6()
    bad = copy.deepcopy(logo)
    del bad["per_fold"][0]["ps_means"]["PS_4_xgboost"]
    assert lc.ladder_gaps(bad, rungs) == ["whichisreal: 1 of the 10 base methods missing (PS_4_xgboost)"]
    bad = copy.deepcopy(logo)
    fold = bad["per_fold"][3]
    second = sorted(fold["ps_means"], key=fold["ps_means"].get)[-2]
    fold["best_ps_name"], fold["best_ps_auc"] = second, fold["ps_means"][second]
    gaps = lc.ladder_gaps(bad, rungs)
    assert len(gaps) == 1 and gaps[0].startswith(f"heygen: the recorded best base method ({second}, ")
    assert lc.ladder_gaps(logo, None) == []                # the ten methods alone need no richer-input run


def test_capped_only_and_widened_only_runs_are_named():
    _, _, logo, rungs = _arch6()
    for grid, other in (("widened", "capped"), ("capped", "widened")):
        one = copy.deepcopy(rungs)
        for f in one["folds"].values():
            for k in [k for k in f["rungs"] if k.endswith(f"_xgb_{grid}")]:
                del f["rungs"][k]
        gaps = lc.ladder_gaps(logo, one)
        assert len(gaps) == 8, gaps
        assert all(g.endswith(f"a {other}-only richer-input run (no XGBoost head over the {grid} grid)") for g in gaps)
    partial = copy.deepcopy(rungs)
    del partial["folds"]["heygen"]["rungs"]["quadratic_logreg"]
    assert lc.ladder_gaps(logo, partial) == ["heygen: 1 of the 12 richer-input candidates missing (quadratic_logreg)"]
    assert lc.ladder_gaps(logo, partial, folds=["stargan", "styleclip"]) == []          # only the analysed folds


def test_the_standard_split_heads_need_their_ten_candidates():
    _, _, logo, _ = _arch6()
    recs = {g: load(SENS / "paper_heads" / "arch6" / f"arch6_{g}.json") for g in GENS}
    heads = {"folds": {g: {"rungs": recs[g]["auc"]} for g in GENS}}
    assert lc.ladder_gaps(logo, heads, candidates=lc.PAPER_HEADS_CANDIDATES) == []
    del heads["folds"]["stargan"]["rungs"]["K_cv_selected_XGB"]
    assert lc.ladder_gaps(logo, heads, candidates=lc.PAPER_HEADS_CANDIDATES) == [
        "stargan: 1 of the 10 richer-input candidates missing (K_cv_selected_XGB)"]


def test_the_stored_runs_pair_through_the_trusted_mapping_only(tmp_path):
    lp, rp, logo, rungs = _arch6()
    assert lc.pairing_problem(logo, rungs, lp, rp) is None
    lf = tmp_path / "lf.json"                              # the same content with other line endings
    lf.write_bytes(rp.read_bytes().replace(b"\r\n", b"\n"))
    assert lc.pairing_problem(logo, rungs, lp, lf) is None
    again = tmp_path / "again.json"                        # the same values written again: another digest
    again.write_text(json.dumps(rungs, indent=1), encoding="utf-8")
    why = lc.pairing_problem(logo, rungs, lp, again)
    assert why.endswith("are not listed together in the trusted provenance mapping (trusted_run_pairs.json); their "
                        "compatibility cannot be checked")
    assert lc.pairing_problem(logo, rungs).endswith("their compatibility cannot be checked")    # no paths: no proof
    for pool in ("arch6", "clip4"):                        # the mechanism runs, primary and deduplicated
        for base, mbase in ((CD, MECH), (DEDUP, DEDUP)):
            a, b = base / f"11_disjoint_{pool}_d{POOLS[pool]}.json", mbase / f"run_c_v2_logo_{pool}.json"
            assert lc.pairing_problem(load(a), load(b), a, b) is None, (pool, base.name)
    heads = SENS / "paper_heads" / "arch6" / "arch6_heygen.json"
    rec = load(heads)
    assert lc.pairing_problem(logo, {"pool": rec["pool"], "per_fold": [rec]}, lp, heads, folds=["heygen"]) is None


def test_mixtures_other_pools_and_missing_folds_are_refused_with_the_reason():
    lp, rp, logo, rungs = _arch6()
    dr = DEDUP / "A3c_logo_extended_rungs_arch6.json"
    names = ("the LOGO run", "the richer-input run")
    assert lc.pairing_problem(logo, load(dr), lp, dr, names) == (
        "fold whichisreal: n_fit 10995 in the richer-input run, 13535 in the LOGO run (another split or training set)")
    dl = DEDUP / "11_disjoint_arch6_d256.json"
    assert lc.pairing_problem(load(dl), rungs, dl, rp, names) == (
        "fold whichisreal: n_fit 13535 in the richer-input run, 10995 in the LOGO run (another split or training set)")
    other = CD / "A3c_logo_extended_rungs_clip4.json"
    assert lc.pairing_problem(logo, load(other), lp, other, names) == (
        "the richer-input run: pool 'clip4', the LOGO run: 'arch6'")
    short = copy.deepcopy(rungs)
    del short["folds"]["heygen"]
    assert lc.pairing_problem(logo, short, lp, rp, names) == (
        "the richer-input run has no fold for heygen, which the LOGO run has")


def _fingerprinted(run):
    out = copy.deepcopy(run)
    for g, f in lc.fold_records(out).items():
        f["fit_fingerprint"], f["select_fingerprint"] = f"fit-{g}", f"select-{g}"
    return out


def test_runs_with_fingerprints_pair_by_them_without_the_mapping():
    _, _, logo, rungs = _arch6()
    a, b = _fingerprinted(logo), _fingerprinted(rungs)
    assert lc.pairing_problem(a, b) is None
    c = copy.deepcopy(b)
    lc.fold_records(c)["stargan"]["select_fingerprint"] = "other rows"
    assert lc.pairing_problem(a, c, names=("x", "y")) == (
        "fold stargan: select_fingerprint differs between x and y (other training rows)")
    assert lc.pairing_problem(a, rungs).endswith("their compatibility cannot be checked")   # one side without


def test_the_membership_fingerprint_ignores_row_order_and_sees_any_change():
    rng = np.random.default_rng(0)
    feats = [rng.normal(size=(50, 4)).astype(np.float32), rng.normal(size=(50, 3)).astype(np.float32)]
    perm = rng.permutation(50)
    assert lc.membership_fingerprint([f[perm] for f in feats]) == lc.membership_fingerprint(feats)
    other = [f.copy() for f in feats]
    other[1][7, 2] += 1e-3
    assert lc.membership_fingerprint(other) != lc.membership_fingerprint(feats)
    assert lc.membership_fingerprint([f[:49] for f in feats]) != lc.membership_fingerprint(feats)


def test_the_trusted_mapping_matches_the_stored_runs_and_keeps_protocols_apart():
    r = subprocess.run([sys.executable, "-B", str(CR / "scripts" / "review" / "trusted_run_pairs_v1.py"), "--check"],
                       cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("PASS: 9 groups, 60 files"), r.stdout + r.stderr
    groups = load(lc.TRUSTED)["groups"]
    assert [g["group"].split(":")[0] for g in groups] == (
        [f"primary {p}" for p in POOLS] + [f"deduplicated {p}" for p in POOLS] + ["submitted arch6"])
    group_of = {}
    for g in groups:
        for path, digest in g["files"].items():
            assert group_of.setdefault(digest, g["group"]) == g["group"], path      # a digest in one group only
            assert ("sensitivity/dedup/" in path) == g["group"].startswith("deduplicated"), path
