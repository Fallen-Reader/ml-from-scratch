import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

np.random.seed(42)


path = os.getenv("PATH")

data = pd.read_csv(f"{path}/Salary_dataset.csv")
"""
fig,ax = plt.subplots()
ax.scatter(data['YearsExperience'],data['Salary'])
ax.set_xlabel("Experince")
ax.set_ylabel("Salary")
fig.savefig("dataset_plot.png",dpi=300)
"""

X = data['YearsExperience'].values #feature
Y = data['Salary'].values #output


idx = np.random.permutation(len(X))
split = int(0.8*len(X))

X_train,X_test = X[idx[:split]],X[idx[split:]]
Y_train,Y_test = Y[idx[:split]],Y[idx[split:]]

m= len(X_train)

mean_x,std_x = np.mean(X_train),np.std(X_train)
mean_y,std_y = np.mean(Y_train),np.std(Y_train)

norm_x_train = (X_train-mean_x)/std_x
norm_x_test = (X_test-mean_x)/std_x
norm_y_train = (Y_train-mean_y)/std_y
norm_y_test = (Y_test-mean_y)/std_y

print(f"X -> mean={mean_x:.2f}, std={std_x:.2f}")
print(f"Y -> mean={mean_y:.2f}, std={std_y:.2f}")
print(f"Normalize X range ->[{norm_x_train.min():.2f},{norm_x_train.max():.2f}]")
print(f"Normalize Y range ->[{norm_y_train.min():.2f},{norm_y_train.max():.2f}]")

#hypothesis

def h(theta,X):
    return theta[0]+theta[1]*X

#cost function

def J(theta,X,y):
    m = len(y)
    error = h(theta,X) - y
    return (1/(2*m))*np.sum(error**2)

#batch gradient desent

def bacth_gd(X,y,learning_rate=0.1,epochs=50):
    m = len(y)
    theta = np.zeros(2)
    cost_hist = []
    for epoch in range(epochs):
        error = h(theta,X) - y
        grad_0 = 1/m*np.sum(error)
        grad_1 = 1/m*np.sum(error*X)

        theta[0] -= learning_rate*grad_0
        theta[1] -= learning_rate*grad_1

        cost_hist.append(J(theta,X,y))
        if epoch % 50 == 0:
            print(f"Epoch {epoch:>4}  J(θ) = {J(theta, X, y):.6f}")

    return theta,cost_hist

print("\n" + "="*55)
print("Training")
print("="*55)

theta_bgd,cost_bdg = bacth_gd(norm_x_train,norm_y_train)

print(f"theta -> {theta_bgd}\n")
  
def predict_dollars(years,theta):
    years_norm = (years - mean_x) / std_x 
    y_norm = h(theta, years_norm)
    return y_norm * std_y + mean_y

year = 5
y_pred_bgd = predict_dollars(year,theta_bgd)

 

print(f"pridected salary for 5 year experience person: {y_pred_bgd:.0f}")