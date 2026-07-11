
# Decision Trees

## 1. Introduction

**Decision Trees**, a non-parametric supervised learning algorithm that partitions the feature space into a set of rectangular regions and fits a simple prediction rule (typically a constant) in each region.

Unlike the linear models we have studied (logistic regression, SVM, GDA), decision trees can capture **non-linear decision boundaries** without requiring an explicit feature mapping or kernel trick. They are also among the most **interpretable** models - the learned hypothesis can be visualized as a flowchart of if-else rules.

---

## 2. The Core Idea

A decision tree consists of:

- **Internal nodes**: Each tests a single feature $x_j$ against a threshold $t$ and routes the sample to the left child ($x_j \leq t$) or right child ($x_j > t$).
- **Leaf nodes**: Make a final prediction, typically the **majority class** of the training samples that reached that leaf.

The learning problem is: given training data, find the tree structure - which features to split on and what thresholds to use - that best predicts the labels.

![for eg](data:image/png;base64,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)


---

## 3. Impurity and the Splitting Criterion

### 3.1 Gini Impurity

For a set of labels $y$ with $C$ classes, the **Gini impurity** measures how often a randomly chosen element would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the subset:

$$G(y) = 1 - \sum_{c=1}^{C} p_c^2$$

where $p_c = \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}\{y^{(i)} = c\}$ is the empirical fraction of class $c$.

Properties:
- $G(y) = 0$ when all samples belong to the same class (**pure node**).
- $G(y)$ is maximized when classes are uniformly distributed ($G = 1 - \frac{1}{C}$).

```python
def gini_imp(y):
    m = len(y)
    if m == 0:
        return 0.0
    imp = 1
    for c in np.unique(y):
        p = np.sum(y == c) / m
        imp -= p**2
    return imp
```

### 3.2 Weighted Gini for a Split

When we split a parent node into left and right children, we compute the **weighted average** of the children's impurities:

$$G_{\text{split}} = \frac{m_{\text{left}}}{m} G(y_{\text{left}}) + \frac{m_{\text{right}}}{m} G(y_{\text{right}})$$

```python
def weigthed_gini_imp(y_left, y_right):
    m = len(y_left) + len(y_right)
    w_left = len(y_left) / m
    w_right = len(y_right) / m
    return w_left * gini_imp(y_left) + w_right * gini_imp(y_right)
```

**Goal:** Find the feature $j$ and threshold $t$ that **minimize** this weighted Gini impurity. A good split sends most samples of one class to the left and most of another class to the right, creating purer children.

---

## 4. Finding the Best Split

For each feature $j$ and each possible threshold $t$ (typically every unique value of that feature in the current node), we:

1. Partition: $x_j \leq t$ goes left, $x_j > t$ goes right.
2. Compute weighted Gini impurity of the split.
3. Track the $(j, t)$ pair with the lowest impurity.

```python
def best_split(x, y):
    m, n = x.shape
    best_feat = None
    best_gini = float('inf')
    best_thres = None

    for j in range(n):
        threshold = np.unique(x[:, j])
        for t in threshold:
            left_mask = x[:, j] <= t
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            g = weigthed_gini_imp(y_left, y_right)

            if g < best_gini:
                best_gini = g
                best_feat = j
                best_thres = t
    
    return best_feat, best_thres, best_gini
```

> **Note on the code:** The `return` statement should be placed outside the `for j` loop so that all features are evaluated. In the provided implementation, ensure the indentation places the `return` at the same level as `for j` rather than inside it.

---

## 5. Building the Tree Recursively

We build the tree via **greedy recursive partitioning**:

1. Find the best split for the current node.
2. If no valid split exists or a stopping criterion is met, make it a leaf.
3. Otherwise, create left and right children and recurse on each partition.

**Stopping criteria** (prevent infinite growth and overfitting):

| Criterion | Meaning |
|-----------|---------|
| **Pure node** | All labels in the node are identical. |
| **Max depth** | The tree has reached a pre-specified maximum depth. |
| **Min samples** | The node contains fewer than `min_sample` points. |
| **No improvement** | No split yields a valid partition. |

```python
def build_tree(x, y, depth=0, maxDepth=5, min_sample=2):
    node = Node()

    # Stopping criterion 1: pure node
    if len(np.unique(y)) == 1:
        node.is_leaf = True
        node.value = y[0]
        return node

    # Stopping criterion 2: max depth reached
    if depth >= maxDepth:
        node.is_leaf = True
        node.value = np.bincount(y.astype(int)).argmax()
        return node

    # Stopping criterion 3: too few samples
    if len(y) < min_sample:
        node.is_leaf = True
        node.value = np.bincount(y.astype(int)).argmax()
        return node

    # Find best split
    feat, thres, gini = best_split(x, y)

    # Stopping criterion 4: no valid split found
    if feat is None:
        node.is_leaf = True
        node.value = np.bincount(y.astype(int)).argmax()
        return node

    # Store split info and recurse
    node.features = feat
    node.threshold = thres

    l_mask = x[:, feat] <= thres
    r_mask = x[:, feat] > thres

    node.left = build_tree(x[l_mask], y[l_mask], depth + 1)
    node.right = build_tree(x[r_mask], y[r_mask], depth + 1)

    return node
```

**Why greedy?** At each node, we choose the locally optimal split without considering future splits. This is computationally efficient but may not yield the globally optimal tree.

---

## 6. Prediction

To predict the label of a new sample $x$, we traverse the tree from the root to a leaf:

1. At each internal node, compare $x_j$ to the threshold $t$.
2. Go left if $x_j \leq t$, else go right.
3. Return the leaf's stored value (majority class).

```python
def predict_one(node, x):
    if node.is_leaf:
        return node.value
    if x[node.features] <= node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

def predict(root, X):
    return np.array([predict_one(root, x) for x in X])
```

**Time complexity:** $O(\text{depth})$ per prediction, which is typically $O(\log m)$ for a balanced tree.

---

## 7. Tree Visualization

```python
def print_tree(node, feature_names=None, depth=0):
    indent = "  " * depth

    if node.is_leaf:
        print(f"{indent}Leaf -> predict class {node.value}")
        return

    fname = feature_names[node.features] if feature_names else f"feature_{node.features}"

    print(f"{indent}{fname} <= {node.threshold:.3f}?")
    print(f"{indent}  |-- YES (go left):")
    print_tree(node.left, feature_names, depth + 2)
    print(f"{indent}  \-- NO  (go right):")
    print_tree(node.right, feature_names, depth + 2)
```

This prints the tree as an indented if-else structure, making the learned rules fully transparent.


---

## 9. Summary

| Aspect | Description |
|--------|-------------|
| **Model type** | Non-parametric, non-linear, axis-aligned partitions |
| **Split criterion** | Gini impurity (or entropy / information gain) |
| **Search strategy** | Greedy, top-down recursive partitioning |
| **Stopping criteria** | Pure node, max depth, min samples, no improvement |
| **Prediction** | Majority vote in leaf |
| **Strengths** | Highly interpretable; no feature scaling needed; captures non-linearity |
| **Weaknesses** | Prone to overfitting; high variance; unstable to small data changes |

### Comparison with Other Models

| Model | Boundary Type | Interpretability | Handles Non-linearity |
|-------|--------------|------------------|---------------------|
| Logistic Regression | Linear | High | No |
| SVM (linear) | Linear | Medium | No |
| SVM (RBF) | Non-linear | Low | Yes |
| GDA | Linear (with shared $\Sigma$) | Medium | No |
| **Decision Tree** | **Axis-aligned** | **Very High** | **Yes** |

---

## 10. Extensions

- **Entropy / Information Gain**: An alternative to Gini impurity. For a node with distribution $p$, entropy is $H(p) = -\sum_c p_c \log p_c$. Information gain is the reduction in entropy after a split.
- **Pruning**: Grow the tree fully, then prune back branches that do not improve validation performance. This often yields better generalization than pre-defined stopping criteria.
- **Random Forests**: Train many trees on bootstrapped subsets of the data and average their predictions. This reduces variance dramatically.
- **Gradient Boosted Trees**: Train trees sequentially, where each new tree corrects the errors of the previous ensemble. XGBoost and LightGBM are state-of-the-art implementations.

---

## References

1. Andrew Ng, CS229 Lecture Notes: "Decision Trees."
2. Breiman, L., Friedman, J., Olshen, R., and Stone, C. *Classification and Regression Trees*. Wadsworth, 1984.
3. Quinlan, J. R. "Induction of Decision Trees." *Machine Learning*, 1986.

---

*This note follows the CS229 (Stanford Machine Learning) course conventions and notation.*
