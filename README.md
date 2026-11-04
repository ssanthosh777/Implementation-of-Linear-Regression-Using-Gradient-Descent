# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess the dataset by importing the required libraries, reading the CSV file, applying one-hot encoding to the categorical State column, and standardizing the input features and target values.

2.Prepare the data for training by separating the independent features (X) and target variable (y), then adding a bias (intercept) column to the feature matrix.

3.Train the Linear Regression model using Batch Gradient Descent by initializing the parameters, repeatedly updating them to minimize the Mean Squared Error (MSE) cost function, and plotting the cost convergence graph.

4.Predict the profit for a new startup by preprocessing the new input in the same way as the training data, applying the learned parameters to make a prediction, and converting the predicted value back to the original scale.
## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: SANTHOSH S
RegisterNumber:  212224100052
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("exp_3_50_Startups.csv")

# Display first few rows
print("First 5 rows of the dataset:")
display(data.head())

# One-hot encode the 'State' column
data_encoded = pd.get_dummies(data, columns=['State'], drop_first=True)

# Display the encoded dataset
print("Encoded dataset:")
display(data_encoded.head())

# Independent features (all columns except Profit)
X_raw = data_encoded.drop('Profit', axis=1).values

# Target variable (Profit)
y_raw = data_encoded['Profit'].values.reshape(-1, 1)

print("Shape of features:", X_raw.shape)
print("Shape of target:", y_raw.shape)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X = X_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw)

# Add a column of ones to X for bias term
m = X.shape[0]
X = np.hstack((np.ones((m, 1)), X))
print("Shape of X after adding bias term:", X.shape)

def compute_cost(X, y, theta):
    """
    Compute Mean Squared Error cost.
    """
    m = len(y)
    preds = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((preds - y) ** 2)
    return cost

def gradient_descent(X, y, learning_rate=0.01, num_iters=2000, tol=1e-8, verbose=False):
    """
    Batch Gradient Descent for Linear Regression.
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []

    prev_cost = compute_cost(X, y, theta)

    for i in range(num_iters):
        preds = X.dot(theta)
        errors = preds - y
        grad = (1 / m) * (X.T.dot(errors))
        theta -= learning_rate * grad

        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        if abs(prev_cost - cost) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break
        prev_cost = cost

        if verbose and (i % 500 == 0 or i < 5):
            print(f"Iteration {i:4d}, Cost: {cost:.6f}")

    return theta, J_history
```

## Output:
<img width="647" height="260" alt="Screenshot 2026-11-04 141128" src="https://github.com/user-attachments/assets/c812ab33-e78b-4bd9-a04a-008e48c83883" />
<img width="853" height="257" alt="Screenshot 2026-11-04 141135" src="https://github.com/user-attachments/assets/cfdaa4d4-be95-472f-b688-cb2a50cf8b9f" />
<img width="281" height="50" alt="Screenshot 2026-11-04 141140" src="https://github.com/user-attachments/assets/4af44707-de01-462a-a184-d2c2235580ba" />
<img width="427" height="35" alt="Screenshot 2026-11-04 141146" src="https://github.com/user-attachments/assets/d55ea32f-d70a-4d7d-8d13-da76c2af0c42" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
