"""
Pass-2 finalizer: refill every CAF sidecar (*_caf.json) in the results dir from the
built cache (results/caf_cache.json), using each sidecar's stored per-position keys.

Run (base env) after caf/run_caf_af3.py has produced the cache:
    python caf/finalize_sidecars.py [--results_dir DIR]
"""

import os
import glob
import argparse

from caf_lookup import refill_sidecar, RESULTS_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=RESULTS_DIR)
    ap.add_argument("--glob", default="*_caf.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, "**", args.glob),
                             recursive=True))
    if not paths:
        print(f"No sidecars matching {args.glob} under {args.results_dir}")
        return
    print(f"Refilling {len(paths)} sidecars from cache...")
    for p in paths:
        try:
            hit, total = refill_sidecar(p)
            frac = 100.0 * hit / total if total else 0.0
            print(f"  {os.path.relpath(p, args.results_dir):55s}  {hit}/{total} ({frac:.1f}%)")
        except Exception as e:
            print(f"  {os.path.relpath(p, args.results_dir):55s}  ERROR: {e}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
