"""
CAF/FLEUR analysis: unpaired setting means + paired per-position drops for the
three perturbation axes, on Clotho and/or AudioCaps.

Reads the CAF sidecars (<name>_caf.json, built by the two-pass pipeline) and the
13-metric score files, pairs every perturbed row against its own source caption,
and writes:

  RESULTS_DIR/caf_summary.csv        dataset,setting,n,caf,clap,fleur
  RESULTS_DIR/caf_paired_drops.csv   dataset,metric,perturbation,pct_drop
      (CAF + its CLAP/FLEUR terms always; the 13 base metrics where the
       corresponding *_scores.json exist)

The synonym axis is reported under two conventions: "Synonym" averages over every
token position (including identity rows where no synonym was found), and
"Synonym-replaced" restricts to positions that actually received a replacement —
the convention used in the paper's tables.

The paper reports the FULL sets (CAF_SAMPLE=0 everywhere); sampled sidecars
(with an `indices` field) are also supported and paired correctly.

    python caf/caf_analysis.py
"""

import os
import sys
import csv
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, clotho_captions_csv  # noqa: E402

R = str(RESULTS_DIR)

BASE_METRICS = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor",
                "cider_d", "spice", "spider", "fense", "clap_sim_text",
                "sbert_sim", "clap_sim_audio"]
SIDE_TERMS = [("CAF", "caf_score"), ("CLAP-term", "clap_score"), ("FLEUR-term", "fleur_score")]
# synonym = WordNet; synonym_mlm = RoBERTa MLM (the paper's synonym axis).
# Both are reported when their sidecars exist; missing ones are skipped.
PERTS = [("synonym", "Synonym"), ("synonym_mlm", "Synonym-MLM"),
         ("swap_adjacent", "Swap"), ("masked", "Masking")]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def clotho_originals():
    cap = []
    with open(clotho_captions_csv("evaluation")) as f:
        r = csv.reader(f); next(r)
        for row in r:
            for i in range(2, len(row)):
                cap.append(row[i])
    return cap


def rowmap(counts):
    m = []
    for ri, c in enumerate(counts):
        m += [ri] * c
    return m


def drop_pct(base_vals, pert_vals, mask=None):
    b = np.asarray(base_vals, float); p = np.asarray(pert_vals, float)
    if mask is not None:
        b, p = b[mask], p[mask]
    keep = ~np.isnan(b) & ~np.isnan(p)
    b, p = b[keep], p[keep]
    return 100.0 * (b.mean() - p.mean()) / b.mean()


def main():
    summary_rows, drop_rows = [], []

    for ds in ["clotho", "audiocaps"]:
        if ds == "clotho":
            base_dir = R
            cands = clotho_originals()
            clotho_full_scores = True   # Clotho *_scores.json enumerate every position
        else:
            base_dir = os.path.join(R, "audiocaps")
            oc = load(os.path.join(base_dir, "original_candidates.json"))
            if oc is None:
                print(f"[skip] {ds}: no original_candidates.json"); continue
            cands = oc["candidates"]
            clotho_full_scores = False  # AudioCaps *_scores.json align with candidates

        orig_caf = load(os.path.join(base_dir, "original_caf.json"))
        orig_scores = load(os.path.join(base_dir, "original_scores.json"))
        if orig_caf is None:
            print(f"[skip] {ds}: no original_caf.json (run the CAF pipeline first)")
            continue

        ntok = [len(c.split()) for c in cands]
        map_full = rowmap(ntok)
        map_swap = rowmap([max(n - 1, 0) for n in ntok])

        summary_rows.append(
            [ds, "original", len(orig_caf["caf_score"]["scores"])] +
            [round(float(np.nanmean(orig_caf[k]["scores"])), 4) for _, k in SIDE_TERMS])

        for pname, plabel in PERTS:
            side = load(os.path.join(base_dir, f"{pname}_caf.json"))
            if side is None:
                print(f"[skip] {ds}/{pname}: sidecar missing"); continue
            n = len(side["caf_score"]["scores"])
            rmap = map_swap if pname == "swap_adjacent" else map_full
            idx = side.get("indices")
            if idx is None:
                idx = list(range(n))
            src = [rmap[i] for i in idx]

            summary_rows.append(
                [ds, pname, n] +
                [round(float(np.nanmean(side[k]["scores"])), 4) for _, k in SIDE_TERMS])

            # Synonym axis: also report the replaced-only convention (identity
            # rows — positions where no synonym was found — measure nothing and
            # dilute the drop; the paper's table uses replaced-only).
            variants = [(plabel, None)]
            if plabel.startswith("Synonym"):
                if ds == "clotho":
                    caps_full = load(os.path.join(base_dir, f"{pname}_sentences.json"))["pred_replaced"]
                    pert_caps = [caps_full[i] for i in idx]
                else:
                    pert_caps = load(os.path.join(base_dir, "synonym_candidates.json"))["candidates"]
                changed = np.array([pert_caps[j] != cands[src[j]] for j in range(len(src))])
                print(f"[{pname}] {ds}: replaced {changed.sum()}/{len(src)} "
                      f"({100 * changed.mean():.1f}%)")
                variants.append((f"{plabel}-replaced", changed))

            ps = load(os.path.join(base_dir, f"{pname}_scores.json"))
            for vlabel, vmask in variants:
                for mlabel, key in SIDE_TERMS:
                    base = [orig_caf[key]["scores"][s] for s in src]
                    drop_rows.append([ds, mlabel, vlabel,
                                      round(drop_pct(base, side[key]["scores"], vmask), 2)])

                if ps is None or orig_scores is None:
                    print(f"[note] {ds}/{pname}: 13-metric scores not available")
                    continue
                for mkey in BASE_METRICS:
                    pfull = ps[mkey]["scores"]
                    pert = [pfull[i] for i in idx] if clotho_full_scores else pfull
                    if len(pert) != len(src):
                        print(f"[warn] {ds}/{pname}/{mkey}: {len(pert)} vs {len(src)}; skip")
                        continue
                    base = [orig_scores[mkey]["scores"][s] for s in src]
                    drop_rows.append([ds, mkey, vlabel,
                                      round(drop_pct(base, pert, vmask), 2)])

    with open(os.path.join(R, "caf_summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "setting", "n", "caf", "clap", "fleur"])
        w.writerows(summary_rows)
    with open(os.path.join(R, "caf_paired_drops.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "metric", "perturbation", "pct_drop"])
        w.writerows(drop_rows)

    print("=== caf_summary.csv ===")
    for r in summary_rows: print(r)
    print("=== paired drops (CAF / CLAP-term / FLEUR-term) ===")
    for r in drop_rows:
        if r[1] in {m for m, _ in SIDE_TERMS}: print(r)


if __name__ == "__main__":
    main()
