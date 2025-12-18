Extension 2: Adaptive Decoding Strategy

Summary:

This extension changes the decoding strategy based on how much of the input sentence is available (completion ratio). For early prefixes (25%, 50%) it uses greedy decoding for lower latency, and for later prefixes (75%, 100%) it uses beam search for higher quality. The notebook reports SARI (and keep/delete/add components) and average latency per completion ratio.

Example Usage:

Upload `extension2.ipynb` to Google Colab and set the runtime to A100 GPU (or TPU). In the Colab file browser, upload:

- `data/` (the full folder from this repo)
- `t5-simplification.zip`
- `t5-simplification-2.zip`
- `score.py`

Unzip the two model archives so that the folders `t5-simplification/` and `t5-simplification-2/` exist in the Colab working directory. Then run the notebook all the way through to produce the Extension 2 results (SARI + latency across completion ratios).