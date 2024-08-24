# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Import the standard libraries.
3. Set variablees for assigning dataset vslues.
4. Import linear regression from sklearn.
5. Assign the points for representing in the graph.
6. Predict the regression for marks by using the representation of the graph.
7. End the program.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KRITHIGA U
RegisterNumber:  212223240076
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
print(df.head())
print(df.tail())
X=df.iloc[:,:-1].values
print("\nArray values of x")
print(X)
Y=df.iloc[:,1].values
print("\nArray values of Y")
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print("\nValues of y prediction")
print(Y_pred)
print("\nArray values of Y test")
print(Y_test)
print("\nTraining set graph")
plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print("\nTest set graph")
plt.scatter(X_test,Y_test,color='brown')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


print("\nValues of MSE,MAE,RMSE")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)


```

## Output:

![image](https://github.com/user-attachments/assets/52c80767-770c-4661-86ee-6c50e9282e13)

![image](https://github.com/user-attachments/assets/898f6039-e9a5-4f78-9a1e-7914364d6d28)

![image](https://github.com/user-attachments/assets/523c9fb2-d99c-4286-bbf7-692a3b7cb94b)

![image](https://github.com/user-attachments/assets/c125c29b-6ad8-41c2-aaab-774b2e669982)

![image](https://github.com/user-attachments/assets/ad971905-f1a0-4be4-929a-82554892b356)

![Screenshot 2024-08-23 203632](https://github.com/user-attachments/assets/63b1d922-ad7e-4369-8d19-ebc863257a5f)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
