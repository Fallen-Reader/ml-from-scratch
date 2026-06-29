import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

np.random.seed(42)


path = os.getenv("D_PATH")

data = pd.read_csv(f"{path}/Salary_dataset.csv")

x = data['YearsExperience'].values #feature
y = data['Salary'].values #output

x_mean,x_std = np.mean(x),np.std(x)
y_mean,y_std = np.mean(y),np.std(y)

x_norm = (x-x_mean)/x_std
y_norm = (y-y_mean)/y_std

def base_error(theta,X_norm):
    return theta @ X_norm.T

def cost_func(X_norm,Y_norm,theta):
    m = len(Y_norm)
    error = base_error(theta,X_norm) - Y_norm
    return (1/(2*m)*np.sum(error**2))
    
def Stochastic(feature,output,rate=0.1,epochs=50):
    feature = np.reshape(feature,(-1,1))
    
    theta = np.zeros(feature.shape[1]+1)
    m= len(output)
    feature = np.column_stack([np.ones(m),feature])
    print(theta.shape,feature.shape)
    

    for epoch in range(epochs):
        for i in range(m):
            xi = feature[i]
            yi = output[i]

            error_i = base_error(theta=theta,X_norm=xi) - yi
            grad = error_i * xi

            theta -= rate*grad

        if epoch % 5 == 0:
            print(f"Epoch {epoch:>4}  J(θ) = {cost_func(feature,output,theta):.6f}")

    return theta
theta = Stochastic(feature=x_norm,output=y_norm)

def predict_dollars(years,theta):
    years_norm = (years - x_mean) / x_std 
    X_b = np.array([1,years_norm])
    y_norm = base_error(theta, X_b)
    return y_norm * y_std+y_mean

year = 5
y_pred = predict_dollars(year,theta)

print(round(y_pred,2))
