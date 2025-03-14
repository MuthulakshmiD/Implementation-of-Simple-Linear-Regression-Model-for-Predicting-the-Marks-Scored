# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

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
![Screenshot 2025-03-14 150713](https://github.com/user-attachments/assets/6699ea83-e5b8-40e5-b05d-654349cf7b7b)

![Screenshot 2025-03-14 150734](https://github.com/user-attachments/assets/385fcc62-6b3d-4954-b25f-b8fdb10128e2)

![Screenshot 2025-03-14 150745](https://github.com/user-attachments/assets/26248815-c61d-4f7b-93e9-1b65ce9fe0a5)

![Screenshot 2025-03-14 150752](https://github.com/user-attachments/assets/4fd92284-ce57-4fe1-96c2-948c3c56417a)

![Screenshot 2025-03-14 150801](https://github.com/user-attachments/assets/a80ba3d3-a1a7-4abe-8a81-d6a8949348f5)

![Screenshot 2025-03-14 150807](https://github.com/user-attachments/assets/95af5024-f78a-4239-ab73-4374031a02fa)

![Screenshot 2025-03-14 150814](https://github.com/user-attachments/assets/3394878e-3581-43e0-a2ed-a2843b7f6164)

![Screenshot 2025-03-14 150823](https://github.com/user-attachments/assets/936cdc37-09bf-4268-b751-89d62e6cec9a)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
