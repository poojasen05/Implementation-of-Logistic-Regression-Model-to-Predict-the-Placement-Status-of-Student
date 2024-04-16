# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

 1.Import the standard libraries. 
 
 2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively. 
 
 3.Import LabelEncoder and encode the dataset. 
 
 4.Import LogisticRegression from sklearn and apply the model on the dataset.
 
 5.Predict the values of array. 
 
 6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. 
 
 7.Apply new unknown values.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: pooja.s
RegisterNumber: 212223040146 
*/
import pandas as pd
data=pd. read_csv ('/content/Placement_Data.csv')
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1.isnull().sum()
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x=data1.iloc[: ,: -1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression

model= LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy = accuracy_score(y_test,y_pred)
confusion = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nconfusion matrix:\n",confusion)
print("\nclassification report:\n",cr)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
```

## Output:
![Screenshot 2024-03-12 105415](https://github.com/poojasen05/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150784373/6b83504a-4d1b-4fd4-8d19-2ba60e9e100b)

![Screenshot 2024-03-12 104341](https://github.com/poojasen05/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150784373/f1cd7014-42a7-4384-a03b-017545c534ec)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
