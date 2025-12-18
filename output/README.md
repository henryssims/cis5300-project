# Model outputs + gold labels
This folder contains model predictions on the **test split**, plus the gold labels.
## Files
- `output/sources.txt`: original sentences (one per line)
- `output/references.txt`: gold simplifications (one per line)
- `output/t5-simplification/<25|50|75|100>/predictions.txt`: baseline predictions
- `output/t5-simplification-2/<25|50|75|100>/predictions.txt`: Extension 3 predictions
- `output/aligned_<25|50|75|100>.tsv`: source/ref/preds table per completion ratio
## How to score (SARI)
Run from the repo root:
```bash
python score.py --outputs output/t5-simplification-2/100/predictions.txt --references output/references.txt --sources output/sources.txt
```
Example commands for all completion ratios:
```bash
python score.py --outputs output/t5-simplification-2/25/predictions.txt --references output/references.txt --sources output/sources.txt
```
```bash
python score.py --outputs output/t5-simplification-2/50/predictions.txt --references output/references.txt --sources output/sources.txt
```
```bash
python score.py --outputs output/t5-simplification-2/75/predictions.txt --references output/references.txt --sources output/sources.txt
```
```bash
python score.py --outputs output/t5-simplification/100/predictions.txt --references output/references.txt --sources output/sources.txt
```
## Example outputs (from this particular run)
### Extension 3 (`t5-simplification-2`)
#### 25%
```
==================================================
SARI Score Results
==================================================
Number of samples: 191

  SARI:        34.36
    - Keep:    26.81
    - Delete:  72.97
    - Add:     3.29
==================================================
```
#### 50%
```
==================================================
SARI Score Results
==================================================
Number of samples: 191

  SARI:        37.24
    - Keep:    38.92
    - Delete:  65.58
    - Add:     7.21
==================================================
```
#### 75%
```
==================================================
SARI Score Results
==================================================
Number of samples: 191

  SARI:        38.36
    - Keep:    45.41
    - Delete:  57.11
    - Add:     12.56
==================================================
```
#### 100%
```
==================================================
SARI Score Results
==================================================
Number of samples: 191

  SARI:        34.55
    - Keep:    48.85
    - Delete:  40.56
    - Add:     14.24
==================================================
```
