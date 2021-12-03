import pandas as pd
import numpy as np

#importing the data set
df=pd.read_csv("C:/Users/dell/Desktop/7th semester/Final Year Project/creditcard.csv")
print(df)
#creating target series
target=df['Class']


#dropping the target variable from the data set
dr=df.drop(columns='Class',axis=1,inplace=True)
print(dr)


print("_____")


#converting them to numpy arrays
X=np.array(df)
y=np.array(target)
X.shape
y.shape

#distribution of the target variable
len(y[y==1])
len(y[y==0])

#splitting the data set into train and test (75:25)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#applyting SMOTE to oversample the minority class
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
print(X.shape,y.shape)
print(len(y[y==1]),len(y[y==0]))
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
#Logistic Regression
logreg=LogisticRegression()
logreg.fit(X,y)
y_logreg=logreg.predict(X_test)
y_logreg_prob=logreg.predict_proba(X_test)[:,1]

#Performance metrics evaluation
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test,y_logreg))
print("Accuracy:\n",metrics.accuracy_score(y_test,y_logreg))
print("Precision:\n",metrics.precision_score(y_test,y_logreg))
#print("Recall:\n",metrics.recall_score(y_test,y_logreg))
#print("AUC:\n",metrics.roc_auc_score(y_test,y_logreg_prob))
auc=metrics.roc_auc_score(y_test,y_logreg_prob)

#plotting the ROC curve
fpr,tpr,thresholds=metrics.roc_curve(y_test,y_logreg_prob)
plt.plot(fpr,tpr,'b', label='AUC = %0.2f'% auc)
plt.plot([0,1],[0,1],'r-.')
plt.xlim([-0.2,1.2])
plt.ylim([-0.2,1.2])
plt.title('Receiver Operating Characteristic\nLogistic Regression')
plt.legend(loc='lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
#
# #load the dataset to a pandas Datafram
# data = pd.read_csv('C:/Users/dell/Desktop/7th semester/Final Year Project/creditcard.csv')
# print(data.head)
# #dataset information
# print(data.info())
# print(data.describe())
# #checking the number of missing value in each column
# print(data.isnull().sum())
# #distribution of legit transactions and fraudulent transactions
# print(data['Class'].value_counts())
# legit=data[data.Class==0]
# fraud=data[data.Class==1]
# print(legit.shape)
# print(fraud.shape)
# # print(legit.Amount.describe())
# # print(fraud.Amount.describe())
# #compare the values for both transaction
# #print(data.groupby('Class').mean())
# LABELS = ["legit", "fraud"]
# count_classes = pd.value_counts(data['Class'], sort = True)
# count_classes.plot(kind = 'pie', rot=0)
# plt.title("Transaction class distribution")
# plt.legend(loc='lower right')
# plt.show()

# #udersampling
# #Build a sample dataset containing similar distribution of normal and fraudulent transaction
# legit_sample=legit.sample(n=492)
# #print(legit_sample)

#
# #concatenation two dataframes
# new_data=pd.concat([legit_sample,fraud],axis=0)
# print(new_data)
# print(new_data['Class'].value_counts())
# print(new_data.groupby('Class').mean())
# #splitting the data into features and targets
# X=new_data.drop(columns='Class',axis=1)
# Y=new_data['Class']
# print("Xis",X)
# print(Y)
#
# #split the data into training and testing data
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
# print(X.shape,X_train.shape,X_test.shape)
# #model training
# model=LogisticRegression()
# #training the logistic regression model with training data
# model.fit(X_train,Y_train)
# X_train_prediction=model.predict(X_train)
# training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
# print("Accuracy on training data",training_data_accuracy)
# #Accuracy on test data
# X_test_prediction=model.predict(X_test)
# test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
# print("Accuracy on test data",test_data_accuracy)
# randon forest
# rf=RandomForestClassifier()
# rf.fit(X_train,Y_train)
# X_train_prediction_rf=rf.predict(X_test)
# test_accuracy=accuracy_score(X_train_prediction_rf,Y_test)
# print("Accuracy on score using random forest",test_accuracy)








