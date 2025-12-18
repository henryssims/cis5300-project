Strong Baseline

The strong baseline uses T5-small fine-tuned on text simplification data (WikiLarge). We prefix inputs with "simplify:" and train the model to generate simplified versions.

Example Input: While the physician was explaining the potential side effects of the medication, the patient appeared visibly anxious.

Example Output: The doctor explained the possible side effects, and the patient looked worried.

Example Usage:

python strong_baseline.py 

This will train a T5-small model on the WikiLarge training dataset and then evalate the model on the test set. 

Output from evaluation:

==================================================
Strong Baseline SARI Score Results
==================================================
Number of samples: 191

  SARI:        32.14
    - Keep:    75.32
    - Delete:  18.19
    - Add:     24.86
==================================================