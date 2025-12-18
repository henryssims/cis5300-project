CIS 5300 Project: Streaming Text Simplification
Authors: Zora Mardjoko, Henry Sims, George Xue, Christopher Wun

Summary:
This project trains and evaluates two T5-based simplification models and several “extension” settings, then compares quality (SARI) and latency across incremental completion ratios (25%, 50%, 75%, 100%).

Colab note:
When running any notebook in Colab, upload the `data/` folder from this repo (it contains `train.csv`, `dev.csv`, and `test.csv`).

What to run to train and evaluate the two models (note that each notebook also has its own README file with more detailed instructions):

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

- Colab: upload and run `code/comparison.ipynb` all the way through. It will output the SARI score results to the console and also save the model to `t5-simplification-2/` while comparing the results of the baseline and extension 3 (encoder adapter) against extension 1 (adaptive length penalty) and extension 2 (adaptive decoding). You can also just plug in the weights we got from the previous steps and run the notebook to get the plots. Note that this must be run on Mac M3 chip for the results to be consistent with ours. It will output the plots to the console.

If you want to run with the weights we got, you can download the zip files from the following Google Drive links:
- `t5-simplification.zip`: https://drive.google.com/file/d/1Souaej0KMwFtTfW21_ROwjIrk6WUBI0U/view?usp=sharing
- `t5-simplification-2.zip`: https://drive.google.com/file/d/1mP9PB822jBU_5rggQzo6FBZ49bg8urqe/view?usp=sharing