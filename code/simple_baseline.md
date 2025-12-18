Simple Baseline

The simple baseline for text simplification is just returning the source sentence unchanged. This establishes a lower bound for the task since any model that actually performs simplification should outperform it.

Example Input: While the physician was explaining the potential side effects of the medication, the patient appeared visibly anxious.

Example Output: While the physician was explaining the potential side effects of the medication, the patient appeared visibly anxious.

Example Usage:

python simple_baseline.py 

This will apply the simple baseline to the WikiLarge test set and do the SARI evaluation on it. You may need to first run pip install huggingface_hub

Output from evaluation:

==================================================
Simple Baseline SARI Score Results
==================================================
Number of samples: 191

  SARI:        19.91
    - Keep:    59.74
    - Delete:  0.00
    - Add:     0.00
==================================================
