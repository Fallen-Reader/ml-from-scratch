# Supervised Learning

Before jumping into any algorithm, there's a set of ideas that everything else builds on.

---

## The Core Idea

Supervised learning is teaching a model by showing it examples where you already know the answer.

You give it input → output pairs, and it figures out the pattern. Once it has the pattern, it can predict the output for inputs it has never seen before.

```
Input (what you know)  →  [Model]  →  Output (what you want to predict)
Years of experience    →  [  ?  ]  →  Salary
Email content          →  [  ?  ]  →  Spam or not spam
House size + location  →  [  ?  ]  →  Price
```

The word *supervised* just means the training data is **labeled** — every input already has the correct answer attached. The model learns by comparing its guesses to those known answers and correcting itself.

Unsupervised learning is the opposite — no labels, the model finds patterns on its own. That comes later.

---

## Two Types of Problems

Supervised learning solves exactly two kinds of problems. Knowing which one you're dealing with determines everything — which algorithm to use, which cost function, which evaluation metric.

### Regression

The output is a **continuous number** — it can be any value on a scale.

```
Years experience = 5  →  Salary = $74,000
Years experience = 8  →  Salary = $96,000
```

The answer isn't a category. It's a point on a number line. Could be 74,000, could be 74,001, could be 73,847.52.

Examples: predicting salary, house prices, temperature, stock value, a patient's blood pressure.

 **Linear Regression** is a regression algorithm.

---

### Classification

The output is a **category** — one of a fixed set of labels.

```
Email content  →  "spam" or "not spam"
Tumor size     →  "malignant" or "benign"
Image          →  "cat" or "dog" or "bird"
```

There's no in-between. The answer has to be one of the defined classes.

**Binary classification** — exactly two classes (yes/no, 0/1, spam/not spam).

**Multiclass classification** — more than two classes (cat/dog/bird, digits 0–9).

The algorithm doesn't output a hard label directly — it usually outputs a **probability**. Then you apply a threshold:
```
P(spam) = 0.91  →  threshold 0.5  →  classified as spam
P(spam) = 0.34  →  threshold 0.5  →  classified as not spam
```

**Logistic Regression** is a classification algorithm (confusing name, but it predicts probabilities not numbers).

---

## Vocabulary/Notation You Need

Every algorithm in supervised learning uses these same terms. Learn them once, they apply everywhere.

**Features — X**
The inputs. What you know about each example. Also called independent variables, predictors, or attributes.
```
x₁ = years of experience
x₂ = age
x₃ = education level
```
When you have multiple features, they're usually written as a vector `x = [x₁, x₂, x₃]`.

---

**Label / Target — y**
The output you're trying to predict. Also called the dependent variable or response.
```
y = salary
```

---

**Training Set**
The labeled examples the model learns from. This is what you pass into the algorithm.

---

**Test Set**
Examples the model has never seen. Used purely to measure how well the model generalizes to new data.

The critical rule: **never touch the test set during training.** If the model sees it while learning, your evaluation is meaningless — it's like giving someone the exam answers while they're studying, then acting surprised that they scored 100%.

---

**m — Number of training examples**
CS229 convention. If your dataset has 30 rows and you use 24 for training, then `m = 24`.

---

**n — Number of features**
If each example has 3 input columns, `n = 3`.

---

**Hypothesis — h_θ(x)**
The model's current best guess for a given input `x`. It's called a hypothesis because before training it's just a guess — theta hasn't been tuned yet.
```
h_θ(x) = θ₀ + θ₁x     ← linear regression hypothesis
```

---

**Parameters — θ (theta)**
The knobs the model tunes during training. `θ₀` is the intercept, `θ₁` is the slope, and so on. Before training they're usually initialized to 0. After training they hold the learned values.

---

**Cost Function — J(θ)**
A number that measures how wrong the model's predictions are across the whole training set. The lower J(θ), the better the model.
```
J(θ) = (1/2m) Σ (h_θ(xᵢ) − yᵢ)²
```
Training is just the process of finding the θ that minimizes J(θ).

---

## The Pattern Every Supervised Algorithm Follows

This is the insight that makes everything click. Every algorithm you'll implement — linear regression, logistic regression, SVM, neural networks — follows the exact same skeleton:

```
1. Start with random θ (usually zeros)
2. Compute h_θ(x)     → make a prediction
3. Compute J(θ)       → measure how wrong it is
4. Update θ           → adjust to reduce the error
5. Repeat until J(θ) stops decreasing
```

What changes between algorithms is:
- The shape of h_θ(x) (a line, a sigmoid curve, a tree...)
- The cost function J(θ) (squared error, cross-entropy...)
- How θ is updated (gradient descent, closed form...)

The skeleton stays the same. Once you see it in linear regression, you'll recognize it in every algorithm that follows.

---

## One Thing to Watch Out For

**Overfitting** — the model memorizes the training data instead of learning the pattern. It performs perfectly on training examples but fails on new data. Think of a student who memorizes past exam questions word for word — useless when the actual exam has different questions.

**Underfitting** — the model is too simple to capture the actual pattern. A straight line through data that clearly curves.

The test set exists specifically to detect both. If training accuracy is high but test accuracy is low, the model is overfitting.

This becomes important once you move past toy datasets. Something to keep in the back of your mind.

---

## What's in This Folder

```
supervised-learning/
├── README.md                  ← you are here
└── linear-regression/
    ├── README.md              ← explanation in plain language
    └── Batch_gradientDescent.py
```

Each algorithm folder has its own README explaining the specific concepts, what tripped me up, and what the code actually does.