#Importing required libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

#load the dataset to a pandas Datafram
data = pd.read_csv('C:/Users/dell/Desktop/7th semester/Final Year Project/creditcard.csv')
dt=data.head(10)
print(dt)
#dataset Description
desc=data.describe()
print("Description about data \n",desc)
#checking the number of missing value in each column
print(data.isnull().sum())
#distribution of legit transactions and fraudulent transactions
print(data['Class'].value_counts())
legit=data[data.Class==0]
fraud=data[data.Class==1]
print(legit.shape)
print(fraud.shape)



#udersampling
# #Build a sample dataset containing similar distribution of normal and fraudulent transaction
# legit_sample=legit.sample(n=492)
# print("legit data after udersampling",legit_sample)
# #concatenation two dataframes
# new_data=pd.concat([legit_sample,fraud],axis=0)
# print("new dataset",new_data)


data['Class'].replace([0,1],['legit','fraud'],inplace=True)
print("data is",data.head(20))
#splitting the data into features and targets
df=data.drop(columns='Class',axis=1)
df=df.drop(columns='Time',axis=1)


target=data['Class']


#converting them to numpy arrays
X=np.array(df)
Y=np.array(target)
# #
# # #oversampling
# # from imblearn.over_sampling import SMOTE
# # oversample = SMOTE()
# #
# # X, Y= oversample.fit_resample(X, Y)
# # print(X.shape)
# # print(Y.shape)
#split the data into training and testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,stratify=Y,random_state=2)
print(X_train)
#print(X.shape,X_train.shape,X_test.shape)
#model training
model=LogisticRegression()
#training the logistic regression model with training data
model.fit(X_train,Y_train)
X_test_prediction=model.predict(X_test)
# print("prediction :\n",X_test_prediction)
# print("y_test:",Y_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
np_array=np.column_stack((X_test_prediction, Y_test))
prediction=pd.DataFrame(np_array,columns=['test_prediction','actual_output'])
print(prediction.head(20))
#
#
#
# # dtc=DecisionTreeClassifier()
# # #training the Decision tree classifier model with training data
# # dtc.fit(X_train,Y_train)
# # dtc_X_test_prediction=dtc.predict(X_test)
# # # print("X_test_prediction \n",dtc_X_test_prediction)
# # # print("Y_test \n",Y_test)
# # np_array=np.column_stack((dtc_X_test_prediction, Y_test))
# # prediction=pd.DataFrame(np_array,columns=['test_prediction','actual_output'])
# # print(prediction.head(20))
#
#
# # dtc_test_data_accuracy=accuracy_score(dtc_X_test_prediction,Y_test)
# aaa=test_data_accuracy
# # bbb=dtc_test_data_accuracy
print("Accuracy score using Logistic Regression Algorithm: ",test_data_accuracy)
# # print("Accuracy score using Decision Tree classifier Algorithm:",dtc_test_data_accuracy)
#
#
# # LABELS = ["legit_sample", "fraud"]
# # count_classes = pd.value_counts(data['Class'], sort = True)
# # count_classes.plot(kind = 'pie', rot=0)
# # plt.title("Transaction class distribution")
# # plt.legend(loc='lower right')
# #
# # plt.show()
# filename='finalized_model.joblib'
# joblib.dump(model,filename)

#
#




