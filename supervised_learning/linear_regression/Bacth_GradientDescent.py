import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

np.random.seed(42)


path = os.getenv("D_PATH")

data = pd.read_csv(f"{path}/Salary_dataset.csv")

X = data['YearsExperience'].values #feature
Y = data['Salary'].values #output


idx = np.random.permutation(len(X))
split = int(0.8*len(X))

X_train,X_test = X[idx[:split]],X[idx[split:]]
Y_train,Y_test = Y[idx[:split]],Y[idx[split:]]

m= len(X_train)

mean_x,std_x = np.mean(X_train,axis=0),np.std(X_train,axis=0)
mean_y,std_y = np.mean(Y_train,axis=0),np.std(Y_train,axis=0)

norm_x_train = (X_train-mean_x)/std_x
norm_x_test = (X_test-mean_x)/std_x
norm_y_train = (Y_train-mean_y)/std_y
norm_y_test = (Y_test-mean_y)/std_y

print(f"X -> mean={mean_x:.2f}, std={std_x:.2f}")
print(f"Y -> mean={mean_y:.2f}, std={std_y:.2f}")
print(f"Normalize X range ->[{norm_x_train.min():.2f},{norm_x_train.max():.2f}]")
print(f"Normalize Y range ->[{norm_y_train.min():.2f},{norm_y_train.max():.2f}]")

#hypothesis

def h(theta,X_b):
    return X_b @ theta

#cost function

def J(theta,X_b,y):
    m = len(y)
    error = h(theta,X_b) - y
    return (1/(2*m))*np.sum(error**2)

#batch gradient desent

def bacth_gd(X_norm,y,learning_rate=0.1,epochs=50):
    m = len(y)
    X_norm = X_norm.reshape(-1,1)
    theta = np.zeros(X_norm.shape[1]+1)
    X_b = np.column_stack([np.ones(m),X_norm])
    cost_hist = []
    for epoch in range(epochs):
        error = h(theta,X_b) - y

        gradient =(1/m)* X_b.T @ error

        theta -= learning_rate*gradient

        cost_hist.append(J(theta,X_b,y))
        if epoch % 5 == 0:
            print(f"Epoch {epoch:>4}  J(θ) = {J(theta, X_b, y):.6f}")

    return theta,cost_hist

print("\n" + "="*55)
print("Training")
print("="*55)

theta_bgd,cost_bdg = bacth_gd(norm_x_train,norm_y_train)

print(f"theta -> {theta_bgd}\n")
  
def predict_dollars(years,theta):
    years_norm = (years - mean_x) / std_x 
    X_b = np.array([1,years_norm])
    y_norm = h(theta, X_b)
    return y_norm * std_y + mean_y

year = 7
y_pred_bgd = predict_dollars(year,theta_bgd)

x_test_b = np.column_stack([np.ones(len(norm_x_test)),norm_x_test])
y_pred_norm = h(theta_bgd,x_test_b)

y_pred = y_pred_norm*std_y+mean_y

ss_res = np.sum((Y_test - y_pred) ** 2)   
ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
r2 = 1 - ss_res / ss_tot


print(f"Predicted salary for {year} years: ${predict_dollars(year, theta_bgd):,.0f}")