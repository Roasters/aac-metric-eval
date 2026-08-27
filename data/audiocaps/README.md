# AudioCaps data

Audio is not redistributed. Set `AUDIOCAPS_ROOT` to the local AudioCaps root with
`eval_text.csv` and `16000/eval/*.wav`. Place the official multi-caption test CSV
from the AudioCaps repository at `data/audiocaps/official_test.csv`, then run:

```bash
python scripts/prepare_datasets.py --dataset audiocaps --prepare
```

This writes `captions_evaluation.csv` and a generated `manifest.json` locally.
