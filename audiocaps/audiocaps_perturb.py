"""
Generate AudioCaps core perturbation candidate sets (leave-one-out originals +
masking + synonym + adjacent-swap), mirroring the Clotho experiments, and register
their (audio, caption) pairs for CAF scoring.

Outputs, under RESULTS_DIR/audiocaps/:
  - <name>_candidates.json : {"candidates","references","audio_paths","indices"?}
        used later by the 13-metric evaluation (evaluate_aac_batch).
  - <name>_caf.json        : CAF sidecar (via dump_sidecar) + registers pairs.

Full sets by default (the paper's setting); set CAF_SAMPLE=N to subsample
perturbation rows (originals are always full).
Runs in the base env (nltk + sentence-transformers available).
"""

import os
import csv
import sys
import json
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "caf"))
from caf_lookup import dump_sidecar, RESULTS_DIR  # noqa: E402
from config import AUDIOCAPS_AUDIO_DIR, AUDIOCAPS_REFS_CSV  # noqa: E402

AC_CSV = str(AUDIOCAPS_REFS_CSV)
AUDIO_DIR = AUDIOCAPS_AUDIO_DIR
OUT_DIR = os.path.join(RESULTS_DIR, "audiocaps")
SAMPLE = int(os.environ.get("CAF_SAMPLE", "0"))  # 0 = full (paper setting)
SEED = 42


def load_rows():
    rows = []
    with open(AC_CSV) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            fname = row[1]
            caps = [c for c in row[2:7] if c]
            if len(caps) == 5:
                rows.append((fname, caps))
    return rows


def build_originals(rows):
    """Leave-one-out: each caption as candidate, other 4 as references."""
    cands, refs, aps = [], [], []
    for fname, caps in rows:
        ap = os.path.join(AUDIO_DIR, fname)
        for i in range(5):
            cands.append(caps[i])
            refs.append(caps[:i] + caps[i + 1:])
            aps.append(ap)
    return cands, refs, aps


def _sample_idx(n, k, seed):
    if k <= 0 or k >= n:
        return list(range(n))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), k))


def build_masking(cands, refs, aps):
    # Non-word substitution (the paper's corrected masking) — same fixed non-word
    # as perturbations/masking.py, not a [MASK] token.
    out_c, out_r, out_a = [], [], []
    for cap, ref, ap in zip(cands, refs, aps):
        toks = cap.split()
        for j in range(len(toks)):
            t = toks[:]; t[j] = "xkqjvz"
            out_c.append(" ".join(t)); out_r.append(ref); out_a.append(ap)
    return out_c, out_r, out_a


def build_swap(cands, refs, aps):
    out_c, out_r, out_a = [], [], []
    for cap, ref, ap in zip(cands, refs, aps):
        toks = cap.split()
        for j in range(len(toks) - 1):
            t = toks[:]; t[j], t[j + 1] = t[j + 1], t[j]
            out_c.append(" ".join(t)); out_r.append(ref); out_a.append(ap)
    return out_c, out_r, out_a


def build_synonym(cands, refs, aps, sample_k):
    """WordNet best-synonym (SBERT-selected) replacement, on a sampled set of
    (caption, token) positions to keep it cheap. Mirrors synonym_replacement.py."""
    import nltk
    from nltk.corpus import wordnet
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    def wn_pos(tb):
        return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN,
                "R": wordnet.ADV}.get(tb[:1])

    # enumerate all positions, then sample
    positions = []  # (row_idx, token_idx)
    for ri, cap in enumerate(cands):
        for j in range(len(cap.split())):
            positions.append((ri, j))
    idx = _sample_idx(len(positions), sample_k, SEED)
    sampled = [positions[i] for i in idx]

    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    out_c, out_r, out_a = [], [], []
    for ri, j in sampled:
        toks = cands[ri].split()
        tagged = nltk.pos_tag(toks)
        word = toks[j]
        syns = set()
        for s in (wordnet.synsets(word, pos=wn_pos(tagged[j][1])) or wordnet.synsets(word)):
            for lem in s.lemmas():
                cand = lem.name().replace("_", " ").lower()
                if cand != word.lower():
                    syns.add(cand)
        syns = list(syns)
        if not syns:
            new = cands[ri]  # no synonym -> unchanged
        else:
            cand_sents = []
            for sy in syns:
                t = toks[:]; t[j] = sy; cand_sents.append(" ".join(t))
            emb = sbert.encode([cands[ri]] + cand_sents, convert_to_numpy=True)
            sims = cosine_similarity(emb[0:1], emb[1:])[0]
            new = cand_sents[int(np.argmax(sims))]
        out_c.append(new); out_r.append(refs[ri]); out_a.append(aps[ri])
    return out_c, out_r, out_a, idx


def save(name, cands, refs, aps, indices=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{name}_candidates.json"), "w") as f:
        json.dump({"candidates": cands, "references": refs, "audio_paths": aps,
                   "indices": indices}, f)
    hit, total = dump_sidecar(cands, aps, os.path.join(OUT_DIR, f"{name}_caf.json"),
                              indices=indices)
    print(f"  {name:16s} rows={total:6d} cache_hit={hit}/{total}")


def main():
    rows = load_rows()
    print(f"AudioCaps clips: {len(rows)}  (SAMPLE={SAMPLE})")
    oc, orf, oa = build_originals(rows)
    save("original", oc, orf, oa)

    mc, mr, ma = build_masking(oc, orf, oa)
    idx = _sample_idx(len(mc), SAMPLE, SEED)
    save("masked", [mc[i] for i in idx], [mr[i] for i in idx], [ma[i] for i in idx], indices=idx)

    sc, sr, sa = build_swap(oc, orf, oa)
    idx = _sample_idx(len(sc), SAMPLE, SEED)
    save("swap_adjacent", [sc[i] for i in idx], [sr[i] for i in idx], [sa[i] for i in idx], indices=idx)

    yc, yr, ya, yidx = build_synonym(oc, orf, oa, SAMPLE)
    save("synonym", yc, yr, ya, indices=yidx)

    print("Done. AudioCaps core perturbations generated + CAF pairs registered.")


if __name__ == "__main__":
    main()
