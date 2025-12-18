Extension 3: T5 + Encoder Adapter (t5-simplification-2)

Summary:

This extension trains a new model checkpoint in `t5-simplification-2/` by adding a small learned adapter on top of the T5 encoder hidden states. The notebook trains on `data/train.csv`, validates on `data/dev.csv`, and prints SARI scores (and keep/delete/add components). The resulting folder can be zipped and reused in other notebooks (especially `comparison.ipynb`).

Example Usage:

Upload `extension3.ipynb` to Google Colab and set the runtime to A100 GPU (or TPU). In the Colab file browser, upload:

- `data/` (the full folder from this repo)
- `t5-simplification.zip`
- `t5-simplification-2.zip`
- `score.py`

Unzip the two model archives so that the folders `t5-simplification/` and `t5-simplification-2/` exist in the Colab working directory. Then run the notebook all the way through to train/evaluate Extension 3 and to produce the reported results.