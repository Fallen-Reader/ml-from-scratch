# Random Forest

A single decision tree overfits. It memorizes the training data and fails on new data. Random Forest fixes this by building many trees, each slightly different, and averaging their predictions.


Two sources of randomness make each tree different:
```
1. Bootstrap sampling  →  each tree trains on a random sample of rows (with replacement)
2. Feature sampling    →  each split only considers a random subset of features
```

```
Dataset: 800 rows, 13 features

Tree 1  →  trained on 800 random rows (some repeated), splits consider 4 random features
Tree 2  →  trained on 800 different rows (different repeats), different 4 features
Tree 3  →  same idea, different randomness
...
Tree 100

Prediction  →  majority vote across all 100 trees
```

## Why this works
Each tree overfits differently - they make different mistakes. When you average them, the mistakes cancel out. This is called variance reduction.

```
Tree 1  wrong on sample 42, 87, 203
Tree 2  wrong on sample 12, 42, 651
Tree 3  wrong on sample 42, 99, 300

Sample 42 — 3 trees wrong → majority still gets it wrong
Sample 87 — only 1 tree wrong → majority vote corrects it
```

The more trees, the more individual mistakes get outvoted.