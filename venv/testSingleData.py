import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

#load the dataset to a pandas Datafram
data = pd.read_csv('C:/Users/dell/Desktop/7th semester/Final Year Project/single_csv_file.csv')
print(data.head)
#dataset information
print(data.info())
print(data.describe())
#checking the number of missing value in each column
print(data.isnull().sum())
#distribution of legit transactions and fraudulent transactions
print(data['Class'].value_counts())
legit=data[data.Class==0]
fraud=data[data.Class==1]
print(legit.shape)
print(fraud.shape)
# print(legit.Amount.describe())
# print(fraud.Amount.describe())
#compare the values for both transaction
#print(data.groupby('Class').mean())

#udersampling
#Build a sample dataset containing similar distribution of normal and fraudulent transaction
#legit_sample=legit.sample(n=492)
#print(legit_sample)
#concatenation two dataframes
# new_data=pd.concat([legit_sample,fraud],axis=0)
# print(new_data)

#splitting the data into features and targets
new_input = [[0	-1.359807134,-0.072781173,2.536346738,1.378155224,-0.33832077,0.462387778,0.239598554,0.098697901,0.36378697,0.090794172,-0.551599533,-0.617800856,-0.991389847,-0.311169354,1.468176972,-0.470400525,0.207971242,0.02579058,0.40399296,0.251412098,-0.018306778,0.277837576,-0.11047391,0.066928075,0.128539358,-0.189114844,0.133558377,-0.021053053,149.62,0
]]

model=LogisticRegression()
new_output = model.predict(new_input)
#training the logistic regression model with training data
model.fit(X_train,Y_train)
X_test_prediction=model.predict(X_test)
print(X_test_prediction,Y_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy score using Logistic Regression Algorithm: ",test_data_accuracy)

dtc=DecisionTreeClassifier()
dtc.fit(X_train,Y_train)
dtc_X_test_prediction=dtc.predict(X_test)
print(dtc_X_test_prediction,Y_test)
dtc_test_data_accuracy=accuracy_score(dtc_X_test_prediction,Y_test)
print("Accuracy score using Decision Tree classifier Algorithm:",dtc_test_data_accuracy)

LABELS = ["legit_sample", "fraud"]
count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.plot(kind = 'pie', rot=0)
plt.title("Transaction class distribution")
plt.legend(loc='lower right')
plt.show()
