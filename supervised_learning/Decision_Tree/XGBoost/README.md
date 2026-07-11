# XGBoost
Random Forest builds trees in parallel independently, then votes. Boosting builds trees sequentially, each tree fixes the mistakes of the previous one.
XGBoost is gradient boosted trees. 

The key insight:
```
Iteration 1  →  build tree on original data
               prediction = F₁(x)

Iteration 2  →  compute residuals (actual - predicted)
               build tree on the RESIDUALS, not the original labels
               F₂(x) = F₁(x) + learning_rate × tree₂(x)

Iteration 3  →  compute new residuals
               build tree on those residuals
               F₃(x) = F₂(x) + learning_rate × tree₃(x)

Final prediction  =  F₁(x) + lr×F₂(x) + lr×F₃(x) + ...
```

Each tree predicts how wrong the current ensemble is, then adds a small correction. The learning rate controls how big each correction is.


## Why residuals?
If your current prediction is 0.6 but the true answer is 1.0, the residual is 0.4 , the model needs to increase its prediction by 0.4. The next tree learns to predict that 0.4 correction. This is literally just a gradient descent in function space , you're minimizing the loss by adding trees instead of updating weights.

