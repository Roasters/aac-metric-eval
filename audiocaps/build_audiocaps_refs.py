"""
Build an AudioCaps 5-caption evaluation reference file, drop-in compatible with the
Clotho experiment pipeline.

The on-disk AudioCaps eval set ($AUDIOCAPS_ROOT) stores only ONE
caption per wav and uses ids that are NOT the official audiocap_id. The official
test set (github.com/cdjkim/audiocaps, dataset/test.csv) has ~5 captions per YouTube
clip. We recover the link by matching each on-disk wav's single caption to the
official captions -> youtube_id -> the full set of 5 references, and pick one on-disk
wav per clip as the audio path.

Output: csv_files/audiocaps_captions_evaluation.csv with the SAME columns as
clotho_captions_evaluation.csv:  ['', 'file_name', 'caption_1'..'caption_5'].

Audio path scheme (used by experiments): AUDIO_DIR/<file_name>.
"""

import csv
import os
import re
import sys
import collections

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from config import (AUDIOCAPS_ONDISK_EVAL, AUDIOCAPS_AUDIO_DIR,  # noqa: E402
                    AUDIOCAPS_OFFICIAL_TEST, AUDIOCAPS_REFS_CSV)

OFFICIAL_TEST = str(AUDIOCAPS_OFFICIAL_TEST)
ONDISK_EVAL = str(AUDIOCAPS_ONDISK_EVAL)
AUDIO_DIR = AUDIOCAPS_AUDIO_DIR
OUT_CSV = str(AUDIOCAPS_REFS_CSV)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def load_official(path):
    """youtube_id -> list[caption] (dedup, preserve order)."""
    yt2caps = collections.OrderedDict()
    with open(path) as f:
        for r in csv.DictReader(f):
            yt2caps.setdefault(r["youtube_id"], [])
            if r["caption"] not in yt2caps[r["youtube_id"]]:
                yt2caps[r["youtube_id"]].append(r["caption"])
    return yt2caps


def load_ondisk(path):
    """on-disk wav id -> single caption."""
    out = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split(",", 1)
            if len(p) == 2:
                out.append((p[0], p[1]))
    return out


def main():
    yt2caps = load_official(OFFICIAL_TEST)
    cap2yts = collections.defaultdict(set)
    for yt, caps in yt2caps.items():
        for c in caps:
            cap2yts[norm(c)].add(yt)

    ondisk = load_ondisk(ONDISK_EVAL)

    # Choose one wav per youtube_id: unique caption match + existing audio file.
    yt2wav = {}
    ambiguous = unmatched = 0
    for wid, cap in ondisk:
        yts = cap2yts.get(norm(cap), set())
        if len(yts) == 0:
            unmatched += 1
            continue
        if len(yts) != 1:
            ambiguous += 1
            continue
        yt = next(iter(yts))
        wav = os.path.join(AUDIO_DIR, wid + ".wav")
        if os.path.exists(wav) and yt not in yt2wav:
            yt2wav[yt] = wid

    rows = []
    skipped_ncaps = 0
    for yt, caps in yt2caps.items():
        if yt not in yt2wav:
            continue
        if len(caps) < 5:
            skipped_ncaps += 1
            continue
        rows.append((yt2wav[yt] + ".wav", caps[:5]))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "file_name", "caption_1", "caption_2", "caption_3",
                    "caption_4", "caption_5"])
        for i, (fname, caps) in enumerate(rows):
            w.writerow([i, fname] + caps)

    print(f"official clips:        {len(yt2caps)}")
    print(f"ambiguous caption skips {ambiguous} | unmatched {unmatched} | <5 refs skipped {skipped_ncaps}")
    print(f"written clips:         {len(rows)}  -> {OUT_CSV}")
    print(f"audio dir:             {AUDIO_DIR}")


if __name__ == "__main__":
    main()
