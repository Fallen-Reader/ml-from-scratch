# Perceptron Learning Algorithm

### Quick summary

The perceptron is a simple linear binary classifier that maps an input vector to a class (+1 or −1) by computing a weighted sum and applying a sign activation. The Perceptron Learning Algorithm (PLA) iteratively updates weights to find a hyperplane that separates the two classes when the data is linearly separable.


### Intuition

A perceptron draws a straight decision boundary (a line in 2D, hyperplane in higher dimensions). For each training example, if the perceptron classifies it correctly, do nothing; if it misclassifies, shift the boundary slightly toward the correct side by adjusting the weights and bias. Over repeated updates this nudging moves the boundary until points are correctly separated (if separation is possible).

![for eg](https://karthikvedula.com/assets/images/2024/07/perceptronVisLine.png)

### Model and prediction

- Features: x = (x1, x2, ..., xn).
- Parameter: θ = (θ1,θ2,....,θn).
- z = θ₀ + θ₁x (same as before)
    ```python
    z = X_m_train @ theta
    ```
- Prediction: z > 0 -> 1 || z < 0 -> -1

### Learning rule (update)

Given a labeled training example (x, y) with y ∈ {+1, −1} and learning rate η > 0:

    If the perceptron predicts correctly , no update.

    If it misclassifies, update:

        θ ← θ + η * d * x

        where,
        x -> missclassified value
        d -> {-1,1} according to x

        This moves the weight vector in the direction that increases alignment with correctly labeled examples.
## Extension

- when plot is not looking linear, sometime it's possible that , this non linear plot can be converted into linear 
by **Ploar coordinates**
![non linear](https://www.edureka.co/blog/wp-content/uploads/2017/07/Linear-528x264.jpg)





## Convergence and limitations

- **The Perceptron Convergence Theorem**:

 if the training data are `linearly separable`, PLA will converge in a finite number of updates to some weight vector that correctly classifies the data (the number of updates depends on the margin and data scale).

If data are `not linearly separable` (e.g., XOR), PLA will not converge and will keep updating indefinitely unless you stop after a fixed number of epochs. This limitation motivated multilayer networks and gradient-based training (backpropagation)

## Practical notes

- Choice of learning rate η affects update magnitude; many implementations use η = 1 for simplicity because scaling x or η is equivalent in direction-based updates.

- PLA finds some separating hyperplane but does not optimize margin; algorithms like Support Vector Machines explicitly maximize margin for better generalization.
