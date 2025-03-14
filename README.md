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

![Screenshot 2025-03-14 150713](https://github.com/user-attachments/assets/909b771f-a084-491d-b104-1089c94c9e96)
![Screenshot 2025-03-14 150823](https://github.com/user-attachments/assets/aa0158e0-5038-45ba-80b2-dfc25b4f7c5d)
![Screenshot 2025-03-14 150814](https://github.com/user-attachments/assets/d8d55de3-20db-4ef7-ae86-ca5ef3291392)
![Screenshot 2025-03-14 150807](https://github.com/user-attachments/assets/e31a3eae-f45a-40d1-b982-2ada332bdbbc)
![Screenshot 2025-03-14 150801](https://github.com/user-attachments/assets/1ab601d7-8feb-4a30-97c4-533f41a08c97)
![Screenshot 2025-03-14 150752](https://github.com/user-attachments/assets/26a8cdf0-88c0-482c-a73e-23cbff13c925)
![Screenshot 2025-03-14 150745](https://github.com/user-attachments/assets/0c578d81-1407-4f36-8d9d-4d3abfe9f9cd)
![Screenshot 2025-03-14 150734](https://github.com/user-attachments/assets/137ae7b5-9ed4-4fdd-8b69-dee7095c9872)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
