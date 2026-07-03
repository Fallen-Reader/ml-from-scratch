## 02 — Logistic Regression

> *A supervised machine learning algorithm used to predict the probability of a binary outcome (0 or 1, Yes/No, True/False) based on independent input features.*

### What I Learned
---

Instead of fitting a straight line through the data points like linear regression, logistic regression applies the mathematical **sigmoid** function to map continuous values into a bounded probability range between 0 and 1

**Linear Combination (z)**:

 The model starts similarly to linear regression by calculating a weighted sum of the inputs

z = θ₀ + θ₁x (same as before)
```python
z = X_m_train @ theta
```

**The Sigmoid Function (σ)**:

used to estimate the probability that y belongs to a particular category given inputs X=(x1,x2,...,xk):

$$\frac{1}{1 + e^{-z}}$$

```pyhton
h = sigmoid(X_m_train @ theta)

def sigmoid(z):
    return 1/(1+np.exp(-z))
```
**Log-Likelyhood**:

When fitting our model, the goal is to find the parameters that optimize a function that defines how well the model is performing. Put simply, the goal is to make predictions as close to 1 when the outcome is 1 and as close to 0 when the outcome is 0. In machine learning, the function to be optimized is called the loss function or cost function. We use the loss function to determine how well our model fits the data.

***(derivation)***:
```
we assume,

P(Y=1|X,θ) = h_θ(x)
P(Y=0|X,θ) = 1 - h_θ(x)

so ->

P(Y|X,θ) = [h_θ(x)]^(y=1/0)[1 - h_θ(x)]^(y=0/1)

```
By taking **Log of MLE(Maximum Likelyhood Estimation)** , we get

$$\text{Log-Loss} =\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(h_θ(x)_i) + (1 - y_i) \log(1 - h_θ(x)_i) \right]$$

```python
def log_likelyhood(theta,X_m_train,y_train):
    m = len(y_train)
    h = sigmoid(X_m_train @ theta)
    h = np.clip(h, 1e-9, 1 - 1e-9)
    return -(1/m)*np.sum(y_train*np.log(h)+(1-y_train)*np.log(1-h))
```
*the Log-Loss depends on the true value for y and the predicted probability.*

**Why negative in code** : because log(x) will return a negative value so to make it positive we're taking negative

**Estimating Coefficients**:

How do we find the coefficients (​θ₀,θ₁....) that minimize the loss function? There are two main approaches for logistic regression: 
- gradient descent 
- maximum likelihood estimation.(gradient ascent)

**Gradient Descent**:

the goal is to minimize the Log-Loss cost function over all samples. 

This method involves selecting initial parameter values, and then updating them incrementally by moving them in the direction that decreases the loss. At each iteration, the parameter value is updated by the gradient, scaled by the step size (otherwise known as the learning rate). The gradient is the vector encompassing the direction and rate of the fastest increase of a function, 

which can be calculated using partial derivatives. The parameters are updated in the opposite direction of the gradient by the step size in an attempt to find the parameter values that minimize the Log-Loss.


Because the gradient calculates where the function is increasing, going in the opposite direction leads us to the minimum of our function. In this manner, we can repeatedly update our model's coefficients such that we eventually reach the minimum of our error function and obtain a sigmoid curve that fits our data well.

```python
def Gradient(x_train_n,y_train,rate=0.001,epochs=500):
    m = len(x_train_n)
    n = len(x_train_n[1])

    X_m_train = np.column_stack([np.ones(m),x_train_n])
    theta = np.zeros(X_m_train.shape[1])
    Mle =[]
    z = X_m_train @ theta

    print(z.min(),z.max(),z.mean())

    for epoch in range(epochs):
        h = sigmoid(X_m_train @ theta) 
        error = h - y_train
        grad = (1/m)* X_m_train.T @ error
        theta -= (rate)*grad
        
        Mle.append(log_likelyhood(theta,X_m_train,y_train))

        if epoch % 50 == 0:
            print(f"Epoch {epoch:>4}  J(θ) = {Mle[-1]:.6f}")
        
        if len(Mle) > 1 and abs(Mle[-2] - Mle[-1]) < 1e-6:
            print(f"Converged at epoch {epoch}")
            break

    return theta,Mle
```
---

## How I Implicated This

1. Imported the Dataset- diease.csv 

2. Since we can't pass Yes/No I have to map Columns which are YES/NO into 1/0.

```python

df['gender'] = df['gender'].map({'Male':1,'Female':0})
df['smoking'] = df['smoking'].map({'Yes':1,'No':0})
df['alcohol_consumption'] = df['alcohol_consumption'].map({'Yes':1,'No':0})
df['disease'] = df['disease'].map({'Yes':1,'No':0})
```

3. Column physical_activity has three Feautres - Low,Medium,High.
So we'll do **Hot-encoding**.
```
 physical_activity  →   Low,Medium,High
    Low                 1    0    0    
    Medium              0    1    0    
    High                0    0    1    
```
```pyhton
df = pd.get_dummies(df,columns=['physical_activity'],drop_first=True,dtype=int)
```

4. Remove irrelevent Columns (Eg.-patient_id)

5. Now Same as before in linear learning:
    1. split set int 80/20
    2. Nomrmalize Traing and Test Dataset
    3. In logistic regression we don't need to normalize Columns with 1/0.
    (eg.-smoking,alcohol_consumption, etc..)

```python
con_col= [0,2,3,4,5,6,7]

mu = x_train[:,con_col].mean(axis=0)
std = x_train[:,con_col].std(axis=0)

x_train_n = x_train.copy().astype(float)
x_test_n = x_test.copy().astype(float)

x_train_n[:,con_col] = (x_train[:,con_col]-mu)/std
x_test_n[:,con_col] = (x_test[:,con_col]-mu)/std
```

**Doubt that may come in your mind**:

```
in error it's y_i - h_(x_i)
then gradient is summation of error with Xj_i ?
don't this we need two loops i from 0 to m and j 0 to n?
```

Written out longhand, the gradient update looks like this:
```python
for i in range(m):          ← loop over every training example
    error = h(xᵢ) − yᵢ
    for j in range(n):      ← loop over every feature
        grad[j] += error * xᵢⱼ
```

That's O(m×n) — two nested loops.

NumPy collapses both into one matrix operation:

```python
error    = h(theta, X_b) - y       # shape (m,)  ← replaces i loop
gradient = X_b.T @ error / m       # shape (n+1,) ← replaces j loop
```

```text
X_b.T          error            result
(n+1, m)   @   (m,)    =       (n+1,)

row 0 = [1, 1, 1, ...]  · errors  →  Σ errorᵢ × 1     = ∂J/∂θ₀
row 1 = [x₁₁,x₁₂,...]  · errors  →  Σ errorᵢ × x₁ᵢ  = ∂J/∂θ₁
row 2 = [x₂₁,x₂₂,...]  · errors  →  Σ errorᵢ × x₂ᵢ  = ∂J/∂θ₂
```

X_b.T @ error is doing exactly the double loop — each row of X_b.T (which is each feature column) gets dot-producted with the error vector.

Same result, zero loops, runs in a fraction of the time.

6. pass the arguments(feature,output) and compute "theta"


## Test the model 

```python
m_test = len(x_test_n)

X_m_test = np.column_stack([np.ones(m_test),x_test_n])

probabilites = sigmoid(X_m_test @ theta)

prediction = (probabilites>=0.5).astype(int)

```

- prediction is thershold that Convert probabilities to class labels (1 or 0)

**Accuracy**:
```python
accuracy = np.mean(prediction == y_test)
print(f"accuracy : {accuracy:.2f}")
```
- tells how accurate this model is

**Confusion Matrix**:

- A confusion matrix is a table used to evaluate the performance of a classification machine learning model.

**NOTE** - The confusion matrix matters more than accuracy here — for a disease dataset, FN (missed diagnoses) is the dangerous number. A model with 85% accuracy but high FN is dangerous in practice.

```text
Actual 0          Actual 1
Predicted 0    True Negative (TN)   False Negative (FN)
Predicted 1    False Positive (FP)  True Positive (TP)
```

- True Positive — model said disease, patient actually has it. Correct.

- True Negative — model said no disease, patient actually doesn't. Correct.

- False Positive — model said disease, patient actually doesn't. Wrong — unnecessary panic, extra tests, but patient is fine.

- False Negative — model said no disease, patient actually does have it. Wrong — patient goes home thinking they're healthy.

### A model that predicts "no disease" for everyone would have zero FN — but it would miss every single sick patient. That's why accuracy alone is misleading.

```python
TP = np.sum((prediction==1)&(y_test==1))
TN = np.sum((prediction==0)&(y_test==0))
FP = np.sum((prediction==1)&(y_test==0))
FN = np.sum((prediction==0)&(y_test==1))
print(f"TP : {TP} TN: {TN} FP:{FP} FN: {FN}")
```

>>> TP : 86 TN: 82 FP:16 FN: 16

- The metric that captures this - **Recall**

- Recall = TP / (TP + FN)

    This specifically measures: of all patients who actually have the disease, how many did the model catch?

- precision = TP / (TP + FP)

    of disease predictions, how many are actually correct

## A good disease model aims for high recall even if precision drops a little
