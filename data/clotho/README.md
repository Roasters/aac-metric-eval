# Clotho data

Audio is not redistributed. Download Clotho V2 under its official terms and set
`CLOTHO_ROOT` to a directory containing:

```text
clotho_csv_files/clotho_captions_evaluation.csv
evaluation/<audio-file>.wav
```

`scripts/prepare_datasets.py --dataset clotho` validates the layout and writes a
local `manifest.json`. The manifest is generated and is not committed by default.
