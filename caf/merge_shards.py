"""
Merge FLEUR/CAF results from data-parallel shard caches back into the main caf_cache.

Each worker (caf/run_caf_af3.py --shard i --num_shards N) writes only its shard's
fleur_score/raw_fleur_score/caf_score into its own cache copy (caf_cache_w<i>.json) to
avoid concurrent-write races. Shards are disjoint, so each key is filled by at most one
worker. This copies those fields into results/caf_cache.json (which already holds CLAP
for every key).

    python caf/merge_shards.py --base results/caf_cache.json \
        --shards results/caf_cache_w0.json results/caf_cache_w1.json
"""

import os
import json
import argparse

R = os.environ.get("CAF_RESULTS_DIR",
                   os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")))
FIELDS = ("fleur_score", "raw_fleur_score", "caf_score", "alpha")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(R, "caf_cache.json"))
    ap.add_argument("--shards", nargs="+",
                    default=[os.path.join(R, "caf_cache_w0.json"),
                             os.path.join(R, "caf_cache_w1.json")])
    args = ap.parse_args()

    base = json.load(open(args.base))
    filled = 0
    for sp in args.shards:
        if not os.path.exists(sp):
            print(f"[skip] {sp} missing"); continue
        shard = json.load(open(sp))
        n = 0
        for k, v in shard.items():
            if v.get("fleur_score") is None:
                continue
            entry = base.setdefault(k, v)
            for f in FIELDS:
                if v.get(f) is not None:
                    entry[f] = v[f]
            n += 1
        filled += n
        print(f"  {os.path.basename(sp)}: merged {n} fleur entries")

    with open(args.base, "w") as f:
        json.dump(base, f)
    total_fleur = sum(1 for v in base.values() if v.get("fleur_score") is not None)
    print(f"DONE merged={filled}  base={len(base)}  with_fleur={total_fleur} -> {args.base}")


if __name__ == "__main__":
    main()
