# Generative Learning Algorithms

## 1. Introduction

In this note, we discuss **generative learning algorithms** and contrast them with **discriminative learning algorithms**, which have been our focus in much of this course so far.

Recall that in classification, our goal is to learn a hypothesis $h: \mathcal{X} 	o \mathcal{Y}$ that maps input features $x$ to output labels $y$. The key difference between generative and discriminative approaches lies in *what* they choose to model.

---

## 2. Discriminative vs. Generative Learning

### 2.1 Discriminative Learning

**Discriminative algorithms** directly model the conditional probability $p(y \mid x)$ — the probability of the label $y$ given the input $x$.

Given a training set, these algorithms try to find a decision boundary that separates different classes. They learn to **discriminate** between classes.

**Examples we've seen:**
- Logistic Regression


For logistic regression, we explicitly model:

$$p(y = 1 \mid x; 	\theta) = \frac{1}{1 + e^{-	\theta^T x}}$$

**Key insight:** Discriminative algorithms learn $p(y \mid x)$ directly. They ask: *"Given the input $x$, what is the probability of each class?"*

### 2.2 Generative Learning

**Generative algorithms** take a different approach. Instead of modeling $p(y \mid x)$ directly, they model:

1. $p(x \mid y)$ — the probability of the features $x$ given the class label $y$
2. $p(y)$ — the prior probability of each class

Then, using **Bayes' rule**, they derive $p(y \mid x)$:

$$p(y \mid x) = \frac{p(x \mid y) p(y)}{p(x)}$$

Since $p(x)$ does not depend on $y$, we can write the decision rule as:

$$arg\max_y p(y \mid x) = arg\max_y p(x \mid y) \, p(y)$$

**Key insight:** Generative algorithms model how the data is *generated* for each class. They ask: *"If I know the class $y$, what features $x$ would I expect to see?"*

---

## 3. An Analogy

Consider the problem of distinguishing between images of dogs and cats.

| Approach | Philosophy |
|----------|-----------|
| **Discriminative** | Learn what makes dogs different from cats. Focus on the boundary between classes. |
| **Generative** | Learn what a dog looks like, and separately learn what a cat looks like. Then, given a new image, ask: "Does this look more like a dog or a cat?" |

---

## 4. Gaussian Discriminant Analysis (GDA)

### 4.1 The Model

A canonical example of a generative algorithm is **Gaussian Discriminant Analysis**. Here, we assume that $p(x \mid y)$ follows a multivariate Gaussian distribution.

For binary classification with $y \in \{0, 1\}$:

$$y \sim 	ext{Bernoulli}(\phi)$$

$$x \mid y = 0 \sim \mathcal{N}(\mu_0, \Sigma)$$

$$x \mid y = 1 \sim \mathcal{N}(\mu_1, \Sigma)$$

The parameters are:
- $\phi$: prior probability of classes
```python
phi_y1 = len(x1)/len(x_train)
phi_y0 = len(x0)/len(x_train)
```

- $\mu_0, \mu_1$: mean vectors for each class
```python
mu_0 = x0[:,gaussian_col].mean(axis=0)
mu_1 = x1[:,gaussian_col].mean(axis=0)
```

- $\Sigma$: shared covariance matrix

```python
std_0 = x0[:,gaussian_col].std(axis=0)
std_1 = x1[:,gaussian_col].std(axis=0)
```

### 4.2 Maximum Likelihood Estimation

Given a training set $\{(x^{(i)}, y^{(i)})\}_{i=1}^m$, the log-likelihood is:

$$\ell(\phi, \mu_0, \mu_1, \Sigma) = \sum_{i=1}^m \log p(x^{(i)} \mid y^{(i)}; \mu_0, \mu_1, \Sigma) + \log p(y^{(i)}; \phi)$$


The maximum likelihood estimates have closed-form solutions:

$$\phi = \frac{1}{m} \sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}$$

$$\mu_0 = \frac{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 0\} x^{(i)}}{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 0\}}$$


$$\mu_1 = \frac{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\} x^{(i)}}{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}}$$

$$\Sigma = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T$$

***scary,right? chill it's just notation actually it's pretty easy , lemme show you implementation***

## The Generative Model Being Fit

modeling **p(x∣y)**  as a product of independent distributions across feature types:

$$
p(x \mid y) = \prod_{j \in \text{gaussian}} p(x_j \mid y; \mu_j, \sigma_j) \times \prod_{j \in \text{bernoulli}} p(x_j \mid y; \phi_j)
$$

And the full joint for a single example:

### p(x,y)=p(y)⋅p(x∣y)

## 1. guassian_pdf(x, mu, sigma) — The Gaussian Likelihood
```python
coff = 1/(sigma*np.sqrt(2*np.pi))
expo = -((x-mu)**2/(2*sigma**2))
return coff*np.exp(expo)
```
This is exactly the **Gaussian PDF**:

$$p(x;\mu,\sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**MLE intuition**: Under this model, the most likely μ  is the sample mean, and the most likely σ2  is the sample variance. The PDF gives the "likelihood score" for each data point under those parameters.

## 2. log_likelihood(...) — The Complete Log-Likelihood
```python
gauss_ll = np.sum(np.log(guassian_pdf(gauss_x, mu, sigma) + 1e-9))
```
This is:

$$\sum_{i=1}^{m} \log p(x_i \mid y; \mu, \sigma) = \sum_{i=1}^{m} \left[ -\log(\sigma \sqrt{2\pi}) - \frac{(x_i - \mu)^2}{2\sigma^2} \right]$$

**Why log?** Products become sums, making optimization tractable. Maximizing the log-likelihood is equivalent to maximizing the likelihood (log is monotonic).

Why + 1e-9? Numerical stability. Prevents log(0)=−∞  when a Gaussian tail assigns near-zero probability.

```python
bern_ll = np.sum(bern_x * np.log(phi_bern + 1e-9) + (1 - bern_x) * np.log((1 - phi_bern) + 1e-9))
```
This is the **Bernoulli log-likelihood**:

$$\sum_{i=1}^{m} \left[ x_i \log \phi + (1 - x_i) \log (1 - \phi) \right]$$
Where 

$$\phi = P(x_j = 1 \mid y)$$

```python
prior = np.log(phi_y)
```
This is logp(y) - **the class prior**.

## 3. The Full Expression

```python
return prior + bern_ll + gauss_ll
```
This computes:


$$\log p(x, y) = \log p(y) + \sum_{j \in \text{gauss}} \log p(x_j \mid y) + \sum_{j \in \text{bern}} \log p(x_j \mid y)$$

This is exactly the log of the joint likelihood for the generative model. To perform MLE, you would maximize this function with respect to the parameters (μ,σ,ϕbern​,ϕy​) .

### If you were to actually solve for the MLE parameters by maximizing this log-likelihood:

| Parameter                              | MLE Closed Form                                | What Your Code Computes         |
| -------------------------------------- | ---------------------------------------------- | ------------------------------- |
| $\mu$ (Gaussian mean)                  | $\frac{1}{m}\sum_{i=1}^m x_i$                  | Used as input to `guassian_pdf` |
| $\sigma$ (Gaussian std)                | $\sqrt{\frac{1}{m}\sum_{i=1}^m (x_i - \mu)^2}$ | Used as input to `guassian_pdf` |
| $\phi_{\text{bern}}$ (Bernoulli param) | $\frac{1}{m}\sum_{i=1}^m x_i$                  | Used as input to `bern_ll`      |
| $\phi_y$ (class prior)                 | $\frac{\text{count}(y)}{m}$                    | Used as `phi_y`                 |


***This function evaluates the log-likelihood at given parameter values — this is the objective that MLE maximizes. If you pre-computed those parameters from the data and just plug them in, you're computing the likelihood under the MLE solution.***


### 4.3 Decision Boundary

Because both classes share the same covariance matrix $\Sigma$, the decision boundary is **linear**. This gives GDA a close relationship to logistic regression.

---

## 5. Naive Bayes(Andrew Ng, CS229 Lecture Notes) Example

### 5.1 The Model

**Naive Bayes** is another generative algorithm, particularly useful when the feature dimension $n$ is large (e.g., text classification).

The "naive" assumption is that the features are **conditionally independent** given the class:

$$p(x \mid y) = \prod_{j=1}^n p(x_j \mid y)$$

Despite this strong (and often incorrect) assumption, Naive Bayes works surprisingly well in practice.

### 5.2 Text Classification Example

For spam classification with a vocabulary of size $n$, let $x_j \in \{0, 1\}$ indicate whether word $j$ appears in the email. We model:

$$p(x_j = 1 \mid y = 1) = \phi_{j \mid y=1}$$

$$p(x_j = 1 \mid y = 0) = \phi_{j \mid y=0}$$

The maximum likelihood estimates are:

$$\phi_{j \mid y=1} = \frac{\sum_{i=1}^m \mathbf{1}\{x_j^{(i)} = 1, y^{(i)} = 1\}}{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}}$$

### 5.3 Laplace Smoothing

To handle words that never appear in the training set for a given class, we use **Laplace smoothing**:

$$\phi_{j \mid y=1} = \frac{1 + \sum_{i=1}^m \mathbf{1}\{x_j^{(i)} = 1, y^{(i)} = 1\}}{2 + \sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}}$$

---

## 6. Generative vs. Discriminative: Trade-offs

| Aspect | Discriminative | Generative |
|--------|---------------|------------|
| **What is modeled** | $p(y \mid x)$ | $p(x \mid y)$ and $p(y)$ |
| **Goal** | Find decision boundary | Model data distribution for each class |
| **Data efficiency** | Often needs more data | Can work with less data (stronger assumptions) |
| **Assumptions** | Weaker | Stronger (e.g., Gaussian, conditional independence) |
| **Accuracy** | Often higher asymptotically | May be higher with small $m$ |
| **Can generate data?** | No | Yes — sample from $p(x \mid y)$ |
| **Handles missing features?** | Harder | Easier via marginalization |

### 6.1 Theoretical Result

There is an interesting theoretical result relating the two: if $p(x \mid y)$ is truly Gaussian (with shared $\Sigma$), then GDA is **asymptotically efficient** — it achieves the best possible performance as $m 	o \infty$. However, logistic regression is more **robust** to incorrect modeling assumptions.

Ng & Jordan (2001) showed that:
- If $p(x \mid y)$ is Gaussian, GDA converges faster (needs less data)
- But logistic regression converges to the correct classifier under a broader set of assumptions

---
## 7. Comparision

### What the model actually learns
```
Logistic Regression learns:
    theta — a set of weights found by gradient descent
Naive Bayes learns:
    mu_0, sigma_0  — what healthy patients look like
    mu_1, sigma_1  — what sick patients look like
    phi_0, phi_1   — binary feature rates per class
    phi_y0, phi_y1 — how common each class is
```
### The full picture of what's happening
```
TRAINING (happens on x_train):
    Step 1 → split into X0, X1 by class
    Step 2 → compute phi_y0, phi_y1      (priors)
    Step 3 → compute mu, sigma per class  (Gaussian params)
    Step 4 → compute phi per class        (Bernoulli params)
    Done.

PREDICTION (happens on x_test):
    For each new patient:
        compute P(x | y=0) using the learned mu_0, sigma_0, phi_0
        compute P(x | y=1) using the learned mu_1, sigma_1, phi_1
        predict whichever class gives higher probability
```
### Comparing the two models
```
                    Logistic Regression    Naive Bayes
Accuracy                84%                 85%
Recall                 0.843               0.961
TP                      86                  98
TN                      82                  72
FP                      16                  26
FN                      16                   4
```

- **Naive Bayes** caught 12 more sick patients (98 vs 86 TP) and missed only 4 compared to logistic regression's 16. For a disease dataset that's a significant difference — 12 people who would have been sent home healthy now get flagged for further testing.

- **The tradeoff** — Naive Bayes is more aggressive. FP went from 16 to 26, meaning 10 more healthy people get incorrectly flagged. In medicine that means unnecessary tests and anxiety, but it's the better tradeoff compared to missing sick patients.

### Why Naive Bayes outperformed logistic regression here

A few reasons:

- The Gaussian assumption fit your continuous features well — glucose, cholesterol, blood pressure being Gaussian means the model's likelihood estimates were accurate.

- Naive Bayes naturally outputs calibrated probabilities when the distributional assumptions hold, which means its decision boundary ends up in a better place for this data.
- Logistic regression with more epochs and tuning could likely close the gap  but out of the box on this dataset, Naive Bayes fits better.
## 7. Summary

- **Discriminative algorithms** model $p(y \mid x)$ directly. They focus on the boundary between classes.
- **Generative algorithms** model $p(x \mid y)$ and $p(y)$, then use Bayes' rule to obtain $p(y \mid x)$. They model how data is generated for each class.
- **GDA** assumes Gaussian class-conditional distributions and yields linear decision boundaries.
- **Naive Bayes** assumes feature independence given the class and is widely used in text classification.
- The choice between generative and discriminative approaches depends on the amount of data available and the validity of the modeling assumptions.

---

## References

1. Andrew Y. Ng and Michael I. Jordan. "On Discriminative vs. Generative Classifiers: A comparison of logistic regression and naive Bayes." *NIPS*, 2001.
2. Andrew Ng, CS229 Lecture Notes: "Generative Learning Algorithms."

---

*This note follows the CS229 (Stanford Machine Learning) course conventions and notation.*
