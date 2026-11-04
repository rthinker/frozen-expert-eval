# C17 and C18: Fit combiners using the frozen experts' binary outputs.
"""K=3 fitting stage. Runs ONLY against a frozen, digest-verified manifest.

Implements the fixed-output three-class comparison; overview: camera_ready/scripts/analyses/README.md#three-class-protocol.
The fit refuses a manifest whose digest no longer matches.

Per fold:
    train  = rows with fold != f AND split == FIT
    select = rows with fold != f AND split == SELECT   (tuning ONLY)
    test   = rows with fold == f                        (scored ONCE)

Primary contrast: standardized full-width multinomial logistic
(`FS-LR(std)-K3`), MINUS the task-adapted prediction ceiling
(`PS-bin-to-K3 adapted ceiling`), where the PS rung and its
hyperparameters are chosen purely by SELECT macro-AUROC. These are not the
paper's FS1 or complete PS1-PS4 ladder. No retrospective maximum over TEST
on either side; if some other rung beats the chosen ceiling on TEST it is
reported as a note.

Secondary: PCA width-matched FS (fitted inside the training fold only).

Temperature calibration for the uncertainty/MI map uses the ORIGINAL binary
real/fake SELECT split - which contains reals - because the K=3 set is
fakes-only.

Usage:  python camera_ready/scripts/analyses/appI1_k3_fit.py --view A_df40 [--quick]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

REPROD = Path(__file__).resolve().parent
RUNS = REPROD  # the analyses folder holds the shared modules
SEED = 1024
N_PERM = 20        # group-level label permutations for the null

import _paths  # noqa: E402,F401
RESULTS_K = _paths.RESULTS_DIR / "three_class"
RESULTS_K.mkdir(parents=True, exist_ok=True)
import _common as common  # noqa: E402

_ra = importlib.util.spec_from_file_location(
    "extended_ladder", RUNS / "appB_richer_inputs.py")
RA = importlib.util.module_from_spec(_ra)
_ra.loader.exec_module(RA)
import _verify_cache_helpers as VC  # noqa: E402  shared cache-reading helpers

from src.utils.config import (CACHE_ROOT, FIT_TAG, SELECT_TAG,  # noqa: E402
                              get_pool_experts)

# Reduced XGBoost set drawn from the SHIPPED grid corners. Capacity is out
# of scope for this experiment; the rung and its
# config are both chosen on SELECT, so this is a selection set, not a
# capacity search.
XGB_SET = [{"n_estimators": n, "max_depth": d, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8}
           for n in (50, 100, 200) for d in (2, 3, 4)]


def load_pool(experts, tag, idx):
    """Concatenated features and stacked logits for the given row indices."""
    feats, logits = [], []
    for e in experts:
        f = np.load(CACHE_ROOT / tag / f"{e}_features.npy", mmap_mode="r")
        l = np.load(CACHE_ROOT / tag / f"{e}_logits.npy", mmap_mode="r")
        feats.append(np.asarray(f[idx], dtype=np.float32))
        logits.append(np.asarray(l[idx], dtype=np.float32))
    return np.concatenate(feats, axis=1), np.stack(logits, axis=1)


def macro_ovr(y, proba, classes):
    """Macro one-vs-rest AUROC, plus per-class."""
    per = {}
    for i, c in enumerate(classes):
        yb = (y == i).astype(int)
        per[c] = float(roc_auc_score(yb, proba[:, i])) if yb.sum() and \
            yb.sum() < len(yb) else float("nan")
    vals = [v for v in per.values() if not np.isnan(v)]
    return float(np.mean(vals)), per


def fit_logreg(Xtr, ytr, Xte_list, scale=True):
    sc = StandardScaler().fit(Xtr) if scale else None
    A = sc.transform(Xtr) if sc else Xtr
    m = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    m.fit(A, ytr)
    return [m.predict_proba(sc.transform(X) if sc else X) for X in Xte_list]


def fit_xgb(Xtr, ytr, Xte_list, params, n_cls):
    m = XGBClassifier(**params, objective="multi:softprob", num_class=n_cls,
                      random_state=SEED, eval_metric="mlogloss", verbosity=0)
    m.fit(Xtr, ytr)
    return [m.predict_proba(X) for X in Xte_list]


def main() -> None:
    # Stream progress even when stdout is redirected to a file.
    # Without this Python block-buffers and the log looks frozen
    # for the whole run, which is useless for a multi-hour job.
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="A_df40")
    ap.add_argument("--quick", action="store_true",
                    help="one fold, logits+FS-LR(std)-K3 only; smoke test")
    ap.add_argument("--n-perm", type=int, default=None,
                    help="group-shuffle permutations (default 20, or 3 "
                         "under --quick). The stored runs use 20.")
    args = ap.parse_args()

    man_path = RESULTS_K / f"K3_FROZEN_{args.view}.json"
    M = json.loads(man_path.read_text(encoding="utf-8"))

    # ---- digest gate: refuse to run on drifted splits -----------------
    rows = M["manifest"]
    digest = hashlib.sha256("\n".join(
        f"{r['split']}|{r['idx']}|{r['fold']}|{r['cls']}" for r in rows
    ).encode()).hexdigest()
    if digest != M["manifest_sha256"]:
        raise SystemExit(f"MANIFEST DIGEST MISMATCH\n  stored {M['manifest_sha256']}\n"
                         f"  computed {digest}\nSplits have drifted; refusing to fit.")
    print(f"manifest digest verified: {digest[:16]}...")

    experts = M["pool"]
    classes = M["classes"]
    cidx = {c: i for i, c in enumerate(classes)}
    n_folds = M["n_folds"]
    print(f"view {args.view}: {len(experts)} experts, {len(rows):,} rows, "
          f"{n_folds} folds, classes {classes}")

    by = defaultdict(list)
    for r in rows:
        by[r["split"]].append(r)

    # ---- calibration pool: ORIGINAL binary SELECT, reals included -----
    # Built ONCE here, but SUBSET PER FOLD below to exclude every row whose
    # content belongs to that fold's held-out groups. A global calibration
    # set would touch the test fold, violating "nothing is ever selected or
    # calibrated using the test fold".
    hexp = VC._hash_expert()
    held = set()
    for g in ["whichisreal", "CollabDiff", "MidJourney", "heygen",
              "starganv2", "deepfacelab", "stargan", "styleclip"]:
        d = CACHE_ROOT / g
        if not d.is_dir():
            raise SystemExit(f"missing held-out generator folder: {d}")
        held |= set(VC._content_hashes(d, hexp))
    sel_h = VC._content_hashes(CACHE_ROOT / SELECT_TAG, hexp)
    sel_lab = np.load(CACHE_ROOT / SELECT_TAG / "labels.npy")
    cal_pool = np.array([i for i, h in enumerate(sel_h) if h not in held])
    # content of each fold's held-out rows, so calibration can exclude it
    fit_h = VC._content_hashes(CACHE_ROOT / FIT_TAG, hexp)
    split_h = {FIT_TAG: fit_h, SELECT_TAG: sel_h}
    fold_content = defaultdict(set)
    for r in rows:
        fold_content[r["fold"]].add(split_h[r["split"]][r["idx"]])
    print(f"binary calibration pool (SELECT, reals kept, held-out generator "
          f"content removed): {len(cal_pool):,} rows, "
          f"{int((sel_lab[cal_pool]==0).sum()):,} real "
          f"- subset per fold below")

    oof = {"y": [], "fs": [], "ps": [], "pca": []}
    folds = range(1 if args.quick else n_folds)
    results = {"view": args.view, "digest": digest, "classes": classes,
               "folds": {}}

    for f in folds:
        t0 = time.time()
        tr = [r for r in rows if r["fold"] != f and r["split"] == FIT_TAG]
        se = [r for r in rows if r["fold"] != f and r["split"] == SELECT_TAG]
        te = [r for r in rows if r["fold"] == f]
        print(f"\n=== fold {f}: train {len(tr):,} | select {len(se):,} | "
              f"test {len(te):,} ===", flush=True)

        data = {}
        for name, grp in (("tr", tr), ("se", se), ("te", te)):
            per_split = defaultdict(list)
            for r in grp:
                per_split[r["split"]].append(r)
            F, L, Y = [], [], []
            for tag, rs in per_split.items():
                idx = np.array([r["idx"] for r in rs])
                order = np.argsort(idx)
                idx_sorted = idx[order]
                ff, ll = load_pool(experts, tag, idx_sorted)
                F.append(ff); L.append(ll)
                Y.append(np.array([cidx[rs[o]["cls"]] for o in order]))
            data[name] = (np.concatenate(F), np.concatenate(L),
                          np.concatenate(Y))

        (Ftr, Ltr, ytr), (Fse, Lse, yse), (Fte, Lte, yte) = \
            data["tr"], data["se"], data["te"]

        # ---- per-fold calibration set: drop this fold's held-out content
        cal_idx = np.array([i for i in cal_pool
                            if sel_h[i] not in fold_content[f]])
        cal_feats, cal_lg = load_pool(experts, SELECT_TAG, cal_idx)
        del cal_feats
        cal_y = sel_lab[cal_idx]
        print(f"  calibration: {len(cal_idx):,} rows "
              f"({int((cal_y==0).sum()):,} real), "
              f"{len(cal_pool)-len(cal_idx):,} dropped as held-out-fold content")

        # ---- PS feature maps; parameters fitted on TRAIN / binary SELECT
        unc, _ = RA.make_uncertainty_featmap(cal_lg, cal_y)   # binary SELECT, reals included
        rank = RA.make_rank_featmap(Ltr)
        maps = {"logits": RA.feat_flat, "disagreement": RA.feat_disagreement,
                "uncertainty+MI": unc, "quadratic": RA.feat_quadratic,
                "rank": rank}
        if args.quick:
            maps = {"logits": RA.feat_flat}

        # ---- adapted PS ceiling chosen on SELECT ONLY -----------------
        best = {"select": -1}
        ps_all = {}
        for mname, fn in maps.items():
            Xtr = np.asarray(fn(Ltr), dtype=np.float64)
            Xse = np.asarray(fn(Lse), dtype=np.float64)
            Xte = np.asarray(fn(Lte), dtype=np.float64)
            pse, pte = fit_logreg(Xtr, ytr, [Xse, Xte])
            s, _ = macro_ovr(yse, pse, classes)
            t, per = macro_ovr(yte, pte, classes)
            ps_all[f"{mname}|logreg"] = {"select": s, "test": t, "per": per}
            if s > best["select"]:
                best = {"select": s, "test": t, "per": per,
                        "rung": f"{mname}|logreg", "proba_te": pte}
            for pr in ([] if args.quick else XGB_SET):
                pse, pte = fit_xgb(Xtr, ytr, [Xse, Xte], pr, len(classes))
                s, _ = macro_ovr(yse, pse, classes)
                t, per = macro_ovr(yte, pte, classes)
                key = f"{mname}|xgb_d{pr['max_depth']}_n{pr['n_estimators']}"
                ps_all[key] = {"select": s, "test": t, "per": per}
                if s > best["select"]:
                    best = {"select": s, "test": t, "per": per,
                            "rung": key, "proba_te": pte}

        # ---- standardized feature LR and width-matched PCA control ----
        (fs_te,) = fit_logreg(Ftr, ytr, [Fte])
        fs_macro, fs_per = macro_ovr(yte, fs_te, classes)

        ps_width = 2 * len(experts)
        pca = PCA(n_components=ps_width, random_state=SEED).fit(Ftr)
        (pc_te,) = fit_logreg(pca.transform(Ftr), ytr,
                              [pca.transform(Fte)])
        pc_macro, pc_per = macro_ovr(yte, pc_te, classes)

        # ---- GROUP-LEVEL shuffled-label control ------------------------
        # The previous version permuted the three class NAMES globally,
        # which merely relabels and preserves every bit of structure - not
        # a null at all. Correct null: reassign class labels ACROSS content
        # groups, so the label no longer tracks which family an image came
        # from. Chance for macro OvR AUROC is 0.5, not 1/3.
        # The SAME permutation is applied to FS and PS, N_PERM times.
        tr_groups = sorted({r["group"] for r in tr})
        g_true = {}
        for r in tr:
            g_true[r["group"]] = cidx[r["cls"]]
        row_group_tr = []
        for name, grp in (("tr", tr),):
            per_split = defaultdict(list)
            for r in grp:
                per_split[r["split"]].append(r)
            for tag, rs in per_split.items():
                order = np.argsort(np.array([r["idx"] for r in rs]))
                row_group_tr.extend(rs[o]["group"] for o in order)

        Xps = np.asarray(RA.feat_flat(Ltr), dtype=np.float64)
        Xps_te = np.asarray(RA.feat_flat(Lte), dtype=np.float64)
        null_fs, null_ps = [], []
        rng = np.random.default_rng(SEED + f)
        n_perm = args.n_perm or (3 if args.quick else N_PERM)
        for _ in range(n_perm):
            labs = [g_true[g] for g in tr_groups]
            rng.shuffle(labs)
            gmap = dict(zip(tr_groups, labs))
            ysh = np.array([gmap[g] for g in row_group_tr])
            if len(np.unique(ysh)) < len(classes):
                continue
            (a,) = fit_logreg(Ftr, ysh, [Fte])
            (b,) = fit_logreg(Xps, ysh, [Xps_te])
            null_fs.append(macro_ovr(yte, a, classes)[0])
            null_ps.append(macro_ovr(yte, b, classes)[0])
        sh_macro = float(np.mean(null_fs)) if null_fs else float("nan")

        oracle = max(ps_all.values(), key=lambda d: d["test"])
        print(f"  PS-bin-to-K3 ceiling (SELECT): {best['rung']:<25}"
              f"test {best['test']:.4f}")
        print(f"  FS-LR(std)-K3    {fs_macro:.4f}   "
              f"delta {fs_macro - best['test']:+.4f}")
        print(f"  FS PCA->{ps_width:<3}       {pc_macro:.4f}   "
              f"delta {pc_macro - best['test']:+.4f}   [secondary]")
        print(f"  group-shuffled null over {len(null_fs)} permutations "
              f"(chance = 0.5):")
        print(f"      FS  mean {np.mean(null_fs):.4f}  "
              f"range [{min(null_fs):.4f}, {max(null_fs):.4f}]")
        print(f"      PS  mean {np.mean(null_ps):.4f}  "
              f"range [{min(null_ps):.4f}, {max(null_ps):.4f}]")
        if oracle["test"] > best["test"] + 1e-9:
            print(f"  NOTE: a different rung scores {oracle['test']:.4f} on "
                  f"TEST; not adopted (no retrospective max)")
        print(f"  ({time.time()-t0:.0f}s)")

        # ---- retain OOF predictions for the pooled secondary metric ----
        oof["y"].append(yte)
        oof["fs"].append(fs_te)
        oof["ps"].append(best["proba_te"])
        oof["pca"].append(pc_te)

        results["folds"][f] = {
            "n_train": len(tr), "n_select": len(se), "n_test": len(te),
            "n_calibration": int(len(cal_idx)),
            "ps_ceiling": {k: v for k, v in best.items() if k != "proba_te"},
            "ps_all": ps_all,
            "fs1": {"macro": fs_macro, "per": fs_per},
            "fs_pca": {"macro": pc_macro, "per": pc_per, "width": ps_width},
            "group_shuffled_null": {
                "n_perm": len(null_fs),
                "registered_n_perm": N_PERM,
                "fs_mean": float(np.mean(null_fs)),
                "fs_range": [float(min(null_fs)), float(max(null_fs))],
                "ps_mean": float(np.mean(null_ps)),
                "ps_range": [float(min(null_ps)), float(max(null_ps))],
                "chance": 0.5,
            },
            "test_max_note": oracle["test"],
            "method_semantics": {
                "fs1_legacy_key": "FS-LR(std)-K3; not paper FS1",
                "ps_ceiling": (
                    "PS-bin-to-K3 adapted ceiling; not complete PS1-PS4"
                ),
                "fs2_evaluated": False,
            },
        }

    # ---- aggregate: primary = mean of per-fold deltas ------------------
    ds = [results["folds"][f]["fs1"]["macro"] -
          results["folds"][f]["ps_ceiling"]["test"] for f in results["folds"]]
    results["primary"] = {
        "definition": "mean over folds of (FS-LR(std)-K3 macro OvR "
                      "AUROC minus SELECT-chosen PS-bin-to-K3 adapted "
                      "ceiling macro OvR AUROC)",
        "per_fold": ds, "mean": float(np.mean(ds)),
        "folds_positive": int(sum(1 for d in ds if d > 0)),
        "n_folds": len(ds),
        "inference": "DESCRIPTIVE ONLY - min two-sided sign-flip p is "
                     f"{2**(1-len(ds)):.4f} at n={len(ds)}",
    }
    print(f"\n=== PRIMARY: mean delta {np.mean(ds):+.4f} over {len(ds)} "
          f"folds, {sum(1 for d in ds if d > 0)} positive ===")
    print(f"per-fold: {[round(d, 4) for d in ds]}")

    # ---- secondary: pooled out-of-fold macro OvR AUROC -----
    # Folds are content-disjoint, so concatenating their held-out
    # predictions gives one out-of-fold score over every row. Reported
    # second to the mean-of-fold-deltas primary.
    if oof["y"] and all(len(v) == len(oof["y"]) for v in oof.values()):
        yv = np.concatenate(oof["y"])
        pooled = {}
        for key, label in (("fs", "FS-LR(std)-K3"),
                           ("ps", "PS-bin-to-K3 ceiling"),
                           ("pca", "FS_PCA")):
            try:
                mac, per = macro_ovr(yv, np.vstack(oof[key]), classes)
                pooled[label] = {"macro": float(mac), "per": per}
            except Exception as exc:                       # never lose the run
                pooled[label] = {"error": repr(exc)}
        if all("macro" in v for v in pooled.values()):
            pooled["delta_feature_lr_minus_PS"] = (
                pooled["FS-LR(std)-K3"]["macro"]
                - pooled["PS-bin-to-K3 ceiling"]["macro"]
            )
        pooled["note"] = ("secondary; the PS side pools each fold's "
                          "SELECT-chosen rung, so it mixes rungs across "
                          "folds - read the primary for the interface "
                          "contrast")
        pooled["n_rows"] = int(len(yv))
        results["pooled_oof"] = pooled
        d_ = pooled.get("delta_feature_lr_minus_PS")
        print(f"\npooled OOF over {len(yv):,} rows: "
              f"FS-LR(std)-K3 "
              f"{pooled['FS-LR(std)-K3'].get('macro', float('nan')):.4f}  "
              f"PS {pooled['PS-bin-to-K3 ceiling'].get('macro', float('nan')):.4f}"
              + (f"  delta {d_:+.4f}" if d_ is not None else ""))
        # Persist the raw OOF arrays so any further metric can be computed
        # offline. View A's first run accumulated these and wrote none of
        # them, which cost a 3.5 h re-run to recover a secondary.
        npz = RESULTS_K / (f"K3_OOF_{args.view}"
                                    + ("_QUICK" if args.quick else "")
                                    + ".npz")
        np.savez_compressed(npz, y=yv, fs=np.vstack(oof["fs"]),
                            ps=np.vstack(oof["ps"]), pca=np.vstack(oof["pca"]),
                            classes=np.array(classes, dtype=object))
        print(f"wrote {npz.name}")
    else:
        results["pooled_oof"] = {"error": "OOF arrays incomplete"}
        print("\nWARNING: pooled OOF not computed - arrays incomplete")

    dest = RESULTS_K / (f"K3_RESULT_{args.view}"
                                 + ("_QUICK" if args.quick else "") + ".json")
    dest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
