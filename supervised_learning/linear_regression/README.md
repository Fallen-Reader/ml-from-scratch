## 01 — Linear Regression

> *Fitting a line through data points — but understanding exactly why the math works.*

### What I Learned
---
The goal is to find `θ` (theta) such that our hypothesis `h_θ(x)` predicts y as accurately as possible — by minimizing the cost function `J(θ)`.

```
h_θ(x) = θ₀ + θ₁x              → the prediction (a line)
J(θ)   = (1/2m) Σ(h_θ(xᵢ)−yᵢ)² → the error to minimize
```

The `1/2` in `J(θ)` isn't arbitrary — it cancels the exponent when you differentiate, giving a cleaner gradient.

### Three ways to find θ
---
**Normal Equation** — closed form, exact answer in one shot:
```
θ = (XᵀX)⁻¹ Xᵀy
```
No iterations, no learning rate. But doesn't scale to large datasets (matrix inversion is expensive).

**Batch Gradient Descent** — uses all m examples every update:
```
θⱼ := θⱼ − α × (1/m) Σᵢ (h_θ(xᵢ) − yᵢ) xⱼ⁽ⁱ⁾
```
Smooth convergence. Needs a learning rate `α`.

**Stochastic Gradient Descent (SGD)** — updates θ after every single example:
```
θⱼ := θⱼ − α × (h_θ(xᵢ) − yᵢ) xⱼ⁽ⁱ⁾
```
Noisier path to minimum, but converges faster early on. Useful when m is large.

---
### Things that actually tripped me up
---
**Feature normalization is not optional.**
X (years: 1–10) and y (salary: 37k–122k) are on completely different scales. Without normalizing both, the gradients explode and theta flies off to `e+28`. After normalizing:
- X and y both have mean=0, std=1
- J(θ) starts below 1.0 instead of in the billions
- The same `α=0.1` that would have diverged now converges cleanly

Always compute `μ` and `σ` from the **training set only**, then apply to test set.

**Updating the gradient instead of theta.**
Easy bug to miss:

```python
# updates the gradient, theta never changes
grad_0 -= alpha * grad_0

# Correct — theta is what should move
theta[0] -= alpha * grad_0
```

**Reading theta directly is misleading.**
`θ = [2.44e-16, 0.976]` looks strange. But `2.44e-16` is just floating point for zero — the intercept being near 0 makes perfect sense when y is normalized to mean=0. Judge theta by `J(θ)` going down and by whether predictions make real-world sense, not by raw numbers.

### How to check if your model is working

```
During training → J(θ) should decrease every epoch
After training  → check one prediction manually
Final check     → R² on test set (0.85+ is solid for one feature)
```

### Dataset

Salary dataset — 30 rows, `YearsExperience` → `Salary`. Simple, clean, perfect for a first implementation.


---

## Stack

```
Python 3
numpy     — all the math (matrix ops, gradients)
pandas    — loading and exploring data
```

No sklearn. If sklearn is used, it'll be explicitly for comparison only.

---

## How to Follow Along

Each folder is self-contained. Clone the repo, pick a topic, run the file.

```bash
git clone https://github.com/Fallen-Reader/ml-from-scratch
cd supervised_learning/linear_regression
python Bacth_GradientDescent.py
```

---

## Resources I'm Using

- [CS229 Lecture Notes](https://cs229.stanford.edu/notes/) — Andrew Ng, Stanford
- Dataset: UCI / Kaggle (linked in each folder)

---

*Updated as I go. If something's wrong or unclear, open an issue — I'm learning too.*