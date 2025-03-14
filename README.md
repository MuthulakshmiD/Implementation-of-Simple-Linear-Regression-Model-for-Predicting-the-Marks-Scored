# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Store data in a structured format (e.g., CSV, DataFrame).

2. Use a Simple Linear Regression model to fit the training data.

3. Use the trained model to predict values for the test set.

4. Evaluate performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Program:

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MUTHULAKSHMI D
RegisterNumber:  212223040122
*/
```
# exp-2 ML
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv("/content/student_scores.csv")
print(df.head())
print(df.tail())
print(df.info())

X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
print(X)
print(Y)

print(X.shape)
print(Y.shape)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test) 
print(Y_pred)
print(Y_test)

mse=mean_squared_error(Y_pred,Y_test)
print("MSE =",mse)
mae=mean_absolute_error(Y_pred,Y_test)
print("MAE =",mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)

plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,Y_pred,color="green")
plt.title("Test set(H vs s)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/46d53222-9283-4196-8bfa-ab8b22d69b0e)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
