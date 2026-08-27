#!/bin/bash
# End-to-end CAF/FLEUR pipeline: CLAP term -> FLEUR term (optionally sharded
# across GPUs) -> merge -> refill sidecars.
#
# Prereq: pairs registered to $RESULTS_DIR/caf_pairs.jsonl, e.g.
#     python caf/attach_clotho.py            # Clotho (full sets by default)
#     python audiocaps/audiocaps_perturb.py  # AudioCaps (full sets by default)
#
# Usage:
#     NUM_SHARDS=4 nohup bash run_caf_pipeline.sh > caf_pipeline.log 2>&1 &
#
# NUM_SHARDS FLEUR workers run on GPUs 0..NUM_SHARDS-1 (one Audio-Flamingo-3
# instance per ~24GB GPU). Fully resumable: already-scored pairs are skipped.
set -u
cd "$(dirname "$0")"
R=${RESULTS_DIR:-$PWD/results}
NUM_SHARDS=${NUM_SHARDS:-1}
mkdir -p logs

echo "[caf] start $(date)  results=$R  shards=$NUM_SHARDS"

# ── 1. CLAP term (base env, GPU 0) ──────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python caf/run_clap.py \
    --pairs "$R/caf_pairs.jsonl" --cache "$R/caf_cache.json" \
    > logs/caf_clap.log 2>&1 || { echo "[caf] CLAP FAILED (logs/caf_clap.log)"; exit 1; }
echo "[caf] CLAP done $(date)"

# ── 2. FLEUR term: sharded AF3 workers (caf_af3 env) ────────────────────────
# Each worker gets a private copy of the CLAP-filled cache and writes only it.
pids=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
    cp "$R/caf_cache.json" "$R/caf_cache_w$i.json"
    CUDA_VISIBLE_DEVICES=$i conda run -n caf_af3 python caf/run_caf_af3.py \
        --pairs "$R/caf_pairs.jsonl" --cache "$R/caf_cache_w$i.json" \
        --shard "$i" --num_shards "$NUM_SHARDS" --flush_every 500 \
        > "logs/caf_af3_w$i.log" 2>&1 &
    pids+=($!)
done
echo "[caf] FLEUR workers: ${pids[*]}"
wait "${pids[@]}"
echo "[caf] FLEUR done $(date)"

# ── 3. Merge shard caches + refill every sidecar ────────────────────────────
shard_files=()
for i in $(seq 0 $((NUM_SHARDS-1))); do shard_files+=("$R/caf_cache_w$i.json"); done
python caf/merge_shards.py --base "$R/caf_cache.json" --shards "${shard_files[@]}" \
    || { echo "[caf] MERGE FAILED"; exit 1; }
python caf/finalize_sidecars.py --results_dir "$R"
echo "[caf] ALL DONE $(date) — next: python caf/caf_analysis.py"
