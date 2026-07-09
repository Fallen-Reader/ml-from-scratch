
# Support Vector Machines (SVM)

## 1. Introduction

In this notes, we present **Support Vector Machines (SVM)**, one of the most powerful and widely used classification algorithms. Unlike generative models that model $p(x \mid y)$, SVM is a **discriminative algorithm** that directly finds the optimal decision boundary between classes.

The key idea: instead of modeling probability distributions, SVM asks - **what is the hyperplane that separates the classes with the largest margin?**

---

## 2. The Margin Intuition

### 2.1 Functional vs. Geometric Margin

Given a linear classifier defined by parameters $(w, b)$, the decision function is:

$$f(x) = w^T x + b$$

We predict $y = 1$ if $f(x) \geq 0$, and $y = -1$ otherwise.

**Functional margin** of a single example $(x^{(i)}, y^{(i)})$:

$$\hat{\gamma}^{(i)} = y^{(i)}(w^T x^{(i)} + b)$$

This is positive when the prediction is correct. But it scales with $||w||$ — doubling $w$ doubles the functional margin without actually changing the decision boundary.

**Geometric margin** (the true distance from a point to the decision boundary):

$$\gamma^{(i)} = \frac{y^{(i)}(w^T x^{(i)} + b)}{||w||}$$

This is **invariant to scaling** of $(w, b)$ and represents the actual perpendicular distance from $x^{(i)}$ to the hyperplane.

```python
def geometric_margin(x, y, w, b):
    return y * (x @ w + b) / np.linalg.norm(w)
```

**Key insight:** SVM maximizes the **minimum geometric margin** across all training examples. This leads to a unique, optimal separating hyperplane.

---

## 3. The Optimization Problem

### 3.1 Hard-Margin SVM (Linearly Separable Case)

If the data is linearly separable, we want to find $(w, b)$ that maximizes the minimum geometric margin:

$$\max_{w, b} \min_{i} \frac{y^{(i)}(w^T x^{(i)} + b)}{||w||}$$

Subject to: $y^{(i)}(w^T x^{(i)} + b) \geq 1$ for all $i$

By fixing the functional margin to be at least 1, this simplifies to:

$$\min_{w, b} \frac{1}{2}||w||^2$$

Subject to: $y^{(i)}(w^T x^{(i)} + b) \geq 1$ for all $i$

This is a **quadratic program** with linear constraints — convex and efficiently solvable.

### 3.2 Soft-Margin SVM (Non-Separable Case)

In practice, data is rarely perfectly separable. We introduce **slack variables** $\xi_i \geq 0$ that allow some points to violate the margin constraint:

$$y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i$$

The optimization becomes:

$$\min_{w, b, \xi} \frac{1}{2}||w||^2 + C \sum_{i=1}^{m} \xi_i$$

Subject to: $y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i$ ,
$\xi_i \geq 0$

Where $C > 0$ is the **regularization parameter** that trades off:
- **Large $C$**: Penalizes margin violations heavily → narrower margin, fewer misclassifications
- **Small $C$**: Allows more violations → wider margin, more tolerant of outliers

---

## 4. The Hinge Loss Formulation

The soft-margin SVM can be rewritten using the **hinge loss**:

$$\mathcal{L}(w, b) = \frac{1}{2}||w||^2 + C \sum_{i=1}^{m} \max\left(0, 1 - y^{(i)}(w^T x^{(i)} + b)\right)$$

The term $\max(0, 1 - z)$ is the hinge loss. It is:
- **Zero** when $z \geq 1$ (point is on or beyond the correct margin)
- **Linearly increasing** when $z < 1$ (point violates the margin)

```python
def hinge_loss(x, y, w, b, c=1.0):
    margins = y * (x @ w + b)
    loss = np.maximum(0, 1 - margins)       
    return (1/2) * np.dot(w, w) + c * np.sum(loss)
```

**Why this loss?**
- The $\frac{1}{2}||w||^2$ term maximizes the margin (minimizes $||w||$)
- The hinge loss term penalizes points that are too close to or on the wrong side of the boundary
- $C$ controls the trade-off between these two objectives

---

## 5. Gradient Computation

To optimize via gradient descent, we need the gradients of the hinge loss objective.

For a single example, the hinge loss is $\max(0, 1 - y^{(i)}(w^T x^{(i)} + b))$.

The gradient with respect to $w$:
- If $y^{(i)}(w^T x^{(i)} + b) \geq 1$: gradient is $0$ (no contribution)
- If $y^{(i)}(w^T x^{(i)} + b) < 1$: gradient is $-y^{(i)}x^{(i)}$

Including the regularization term:

$$\nabla_w \mathcal{L} = w - C \sum_{i: \text{margin}_i < 1} y^{(i)} x^{(i)}$$

$$\nabla_b \mathcal{L} = -C \sum_{i: \text{margin}_i < 1} y^{(i)}$$

```python
def gradient(x, y, w, b, c=1.0):
    margin = y * (x @ w + b)
    mask = (margin < 1).astype(float)  # 1 if violating margin, 0 otherwise
    dw = w - c * (y * mask) @ x
    db = -c * np.sum(y * mask)
    return dw, db
```

**The mask logic:** `mask[i] = 1` if example $i$ violates the margin (i.e., is a "support vector candidate"), and `0` otherwise. Only these points contribute to the gradient - a key property of SVMs.

---

## 6. Training via Gradient Descent

```python
def train_svm(x, y, c=1.0, rate=0.1, epochs=500):
    m, n = x.shape
    w = np.zeros(n)
    b = 0
    hist = []
    
    for e in range(epochs):
        dw, db = gradient(x, y, w, b, c)
        
        w -= rate * dw
        b -= rate * db
        
        loss = hinge_loss(x=x, y=y, w=w, b=b, c=c)
        hist.append(loss)
        
        # Early stopping if converged
        if len(hist) > 1 and abs(hist[-2] - hist[-1]) < 1e-6:
            print(f"Converged at epoch {e}")
            break
        
        if e % 50 == 0:
            print(f"epoch = {e}: loss = {loss:.4f}")
    
    return w, b, hist
```

**Note:** This is a simplified **subgradient descent** implementation. The true SVM optimization (via SMO, interior point methods, or quadratic programming) is more efficient, but gradient descent illustrates the core idea clearly.

---

## 7. The Decision Boundary and Margins

Once trained, the decision boundary is the hyperplane where $w^T x + b = 0$.

The **margin boundaries** are:
- Positive margin: $w^T x + b = +1$ (where positive-class support vectors lie)
- Negative margin: $w^T x + b = -1$ (where negative-class support vectors lie)

The **width of the margin** is:

$$\text{margin width} = \frac{2}{||w||}$$

```python
x1_range = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)

def get_x2(x1, w, b, offset=0):
    return -(w[0]*x1 + b + offset) / w[1]

# Decision boundary (offset=0)
x2_bound = get_x2(x1_range, w, b, offset=0)
# Positive margin (offset=+1)
x2_upper_margin = get_x2(x1_range, w, b, offset=1)
# Negative margin (offset=-1)
x2_lower_margin = get_x2(x1_range, w, b, offset=-1)
```

---

## 8. Support Vectors

**Support vectors** are the data points that lie on or within the margin boundaries (i.e., where $y^{(i)}(w^T x^{(i)} + b) \leq 1$). These are the only points that matter — if you removed all other points, the decision boundary would remain unchanged.

```python
margins = Y * (X @ w + b)
support_vectors = X[margins <= 1.05]  # slight tolerance for numerical precision
```

This sparsity property is one of SVM's biggest strengths: the model depends only on a subset of the training data.

---

## 9. The Kernel Trick

### 9.1 Motivation

What if the data is not linearly separable? Instead of using a linear boundary in the original feature space, we can:

1. Map features to a higher-dimensional space: $\phi: \mathbb{R}^n \to \mathbb{R}^d$ where $d \gg n$
2. Find a linear separator in that high-dimensional space
3. This corresponds to a non-linear boundary in the original space

### 9.2 The Kernel Function

Computing $\phi(x)$ explicitly can be expensive. Instead, we use a **kernel function** $K(x, z)$ that computes the inner product in the high-dimensional space without explicitly constructing $\phi$:

$$K(x, z) = \phi(x)^T \phi(z)$$

This is the **kernel trick** — we never need to know $\phi$, only $K$.

### 9.3 Common Kernels

**Linear Kernel** (no transformation, equivalent to original SVM):

$$K(x, z) = x^T z$$

```python
def kernel_linear(x1, x2):
    return np.dot(x1, x2)
```

**Polynomial Kernel**:

$$K(x, z) = (x^T z + c)^d$$

```python
def kernel_poly(x1, x2, degree=2, c=1):
    return (np.dot(x1, x2) + c) ** degree
```

**RBF (Radial Basis Function) / Gaussian Kernel**:

$$K(x, z) = \exp\left(-\gamma ||x - z||^2\right)$$

This kernel maps to an **infinite-dimensional** space and can model highly complex boundaries.

```python
def kernel_rbf(x1, x2, gamma=0.5):
    diff = x1 - x2
    return np.exp(-gamma * np.dot(diff, diff))
```

**The $\gamma$ parameter** controls the "reach" of each training example:
- **Small $\gamma$**: Far-reaching influence → smoother boundaries
- **Large $\gamma$**: Local influence → can overfit to individual points

```python
gammas = [0.01, 0.1, 0.5, 2.0, 10.0]
for gamma in gammas:
    K = compute_kernel_matrix(X, kernel_rbf, gamma=gamma)
    avg_similarity = K[Y==1][:, Y==-1].mean() 
    print(f"Gamma={gamma:>5}  avg cross-class similarity = {avg_similarity:.4f}")
```

### 9.4 Kernel Matrix

The **kernel matrix** (or Gram matrix) $K \in \mathbb{R}^{m \times m}$ stores all pairwise kernel evaluations:

$$K_{ij} = K(x^{(i)}, x^{(j)})$$

```python
def compute_kernel_matrix(X, kernel_fn, **kwargs):
    m = len(X)
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = kernel_fn(X[i], X[j], **kwargs)
    return K
```

For the dual SVM formulation, the optimization depends only on this matrix — we never touch the original features directly.


---

## 11. Summary

| Concept | Description |
|---------|-------------|
| **Discriminative** | Models $p(y \mid x)$ directly via decision boundary |
| **Margin** | Distance from decision boundary to nearest data point |
| **Hinge Loss** | Penalizes points inside or on wrong side of margin |
| **$C$ parameter** | Regularization: trade-off between margin width and violations |
| **Support Vectors** | Only points near the boundary matter; model is sparse |
| **Kernel Trick** | Implicitly map to high-dimensional space via $K(x, z)$ |
| **RBF Kernel** | Infinite-dimensional mapping; handles complex boundaries |

**SVM vs. Generative Models:**
- SVM (discriminative): Finds the boundary. Doesn't care about $p(x)$.
- GDA/Naive Bayes (generative): Models how data is generated for each class. Uses Bayes' rule.
- SVM often wins when $m$ is large; generative models can win with small $m$ and correct assumptions.

---

## References

1. Andrew Ng, CS229 Lecture Notes: "Support Vector Machines."
2. Cortes, C. and Vapnik, V. "Support-vector networks." *Machine Learning*, 1995.

---

*This note follows the CS229 (Stanford Machine Learning) course conventions and notation.*
