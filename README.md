# Into ML

A hands-on journal of learning machine learning from scratch.

Every algorithm here is implemented using only **numpy** and **pandas** — no sklearn, no black boxes. The goal is to understand what's actually happening before ever touching a library that hides it.

Math notation follows **CS229 (Andrew Ng, Stanford)** — `h_θ(x)`, `J(θ)`, gradient descent and all. If you've watched those lectures, the code will feel familiar.

---

## Structure

```
Into ML/
├── supervised-learning/
│   ├── README.md               ← what supervised learning is
│   └── linear-regression/
│       ├── README.md           ← explanation in my own words
│       └── Batch_gradientDescent.py
└── ...more topics coming
```

Each folder has its own README — written in plain language, not textbook language. The code files are the hands-on part; the READMEs are me explaining it back to make sure I actually understood it.

---

## Roadmap

- [x] Supervised Learning
  - [x] Linear Regression
  - [ ] Logistic Regression
  - [ ] Support Vector Machines
  - [ ] Decision Trees
- [ ] Unsupervised Learning
  - [ ] K-Means Clustering
  - [ ] PCA
- [ ] Neural Networks

---

## Rules I'm Following

- Implement the math before using any library
- Move on only when I can explain it in my own words
- Document every bug and mistake

---

## Stack

```
Python 3 · numpy · pandas
```

---

*This is a learning repo, not a polished project. If something is wrong or could be clearer, open an issue.*