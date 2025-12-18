CIS 5300 Project: Streaming Text Simplification
Authors: Zora Mardjoko, Henry Sims, George Xue, Christopher Wun

Summary:
This project trains and evaluates two T5-based simplification models and several “extension” settings, then compares quality (SARI) and latency across incremental completion ratios (25%, 50%, 75%, 100%).

What to run to train and evaluate the two models:

1) Train the baseline model (writes `t5-simplification/`)

- Colab: upload and run `code/extension1.ipynb` all the way through. It trains the baseline model and includes cells to zip/download the resulting folder. It will output the SARI score results to the console and also save the model to `t5-simplification/` while comparing the results of the baseline and extension 1 (adaptive length penalty).
- Local: run `python code/strong_baseline.py` to train the same baseline model locally. It will output the SARI score results to the console.

2) Try the extension 2 model (adaptive decoding)

- Colab: upload and run `code/extension2.ipynb` all the way through. It tests the extension 2 model and includes cells to zip/download the resulting folder. You must first run `code/extension1.ipynb` to train the baseline model and get those weights. The notebook will output the SARI score results to the console and also save the model to `t5-simplification-2/` while comparing the results of the baseline and extension 2 (adaptive decoding) against extension 1 (adaptive length penalty).

3) Train the Extension 3 model (writes `t5-simplification-2/`)

- Colab: upload and run `code/extension3.ipynb` all the way through. It trains the adapter-augmented model and saves it to `t5-simplification-2/`. It will output the SARI score results to the console and also save the model to `t5-simplification-2/`.

4) Comparisons and plots (after Extension 3):

- `code/comparison.ipynb` is the “master comparison” notebook. It loads/unzips `t5-simplification.zip` and `t5-simplification-2.zip` if needed, and produces tables/plots comparing:
  - Strong Baseline (fixed)
  - Extension 1 (adaptive lenpen)
  - Extension 2 (adaptive decoding)
  - Extension 3 (arch, fixed)
  - Extension 3 + beam-techniques

- Colab: upload and run `code/comparison.ipynb` all the way through. It will output the SARI score results to the console and also save the model to `t5-simplification-2/` while comparing the results of the baseline and extension 3 (encoder adapter) against extension 1 (adaptive length penalty) and extension 2 (adaptive decoding). You can also just plug in the weights we got from the previous steps and run the notebook to get the plots. Note that everything must be run on A100 for the results to be consistent with ours. It will output the plots to the console.

- `code/comparison_plots_only.ipynb` is plotting-only. It reads the `PRECOMPUTED` dict near the top of the notebook, which were the results we got from the previous runs of the other notebooks.
  - If you rerun experiments and get new numbers in `comparison.ipynb` (or directly from `extension2.ipynb` / `extension3.ipynb`), copy those SARI/KEEP/DELETE/ADD/latency_ms values into the matching entries in `PRECOMPUTED`, then rerun the plotting notebook to get new plots.

- Colab: upload and run `code/comparison_plots_only.ipynb` all the way through. It will output the plots to the console. It does not require any weights to be run.

If you want to run with the weights we got, you can download the zip files from the following Google Drive links:
- `t5-simplification.zip`: https://drive.google.com/file/d/1Souaej0KMwFtTfW21_ROwjIrk6WUBI0U/view?usp=sharing
- `t5-simplification-2.zip`: https://drive.google.com/file/d/1mP9PB822jBU_5rggQzo6FBZ49bg8urqe/view?usp=sharing