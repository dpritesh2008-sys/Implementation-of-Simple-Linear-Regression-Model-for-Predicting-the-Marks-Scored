# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Load the dataset into a DataFrame and explore its contents to understand the data structure.
2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.
3.Create a linear regression model and fit it using the training data.
4.Predict the results for the testing set and plot the training and testing sets with fitted lines.
5.Calculate error metrics (MSE, MAE, RMSE) to evaluate the model’s performance.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Ritesh DP
RegisterNumber: 212225040339
*/
```
```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
df = pd.read_csv(r"C:\Users\admin\Downloads\exp_2_dataset_student_scores.csv")
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")
print("Developed by Deetchana S")

X = df.iloc[:, :-1].values   
Y = df.iloc[:, -1].values    
print("X (features):", X.flatten())
print("Y (targets):", Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)

plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

new_hours = np.array([[2.5], [8.0]])   
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))
```
## Output:
<img width="273" height="393" alt="image" src="https://github.com/user-attachments/assets/a2622f12-171f-4c68-8e9f-899fd671c836" />
<img width="757" height="77" alt="image" src="https://github.com/user-attachments/assets/dec27220-bff6-4878-89d5-3e9cd6bf5074" />
<img width="805" height="507" alt="image" src="https://github.com/user-attachments/assets/01b3735f-062b-460e-b54f-7e6a3fbbf248" />
<img width="993" height="106" alt="image" src="https://github.com/user-attachments/assets/ebfb1a5f-5433-41b2-b33b-7af819865ed0" />
<img width="721" height="490" alt="image" src="https://github.com/user-attachments/assets/439bace4-c29c-4670-8c7d-d53693b4e273" />
<img width="541" height="87" alt="image" src="https://github.com/user-attachments/assets/84723e40-1d00-4fa6-84e1-042145ee7c7e" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
