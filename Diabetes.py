import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#load the data into pandas data frame
data = pd.read_csv('D:/ML/diabetes.csv')

#printing the first 5 rows of the dataset
#print(data.head())

#no of rows and columns in the dataset
print('np of rows & columns: ', data.shape)

#getting the statistical measure of the data
print(data.describe())
print(data['Outcome'].value_counts())
print(data.groupby('Outcome').mean())

#separating the data and labels
X = data.drop(columns= 'Outcome', axis=1)
Y = data['Outcome']
print(X)
print(Y)

#data standardization
scaler= StandardScaler()
scaler.fit(X)
standardized_data= scaler.transform(X)
print('standardized data: ', standardized_data)

X = standardized_data
Y = data['Outcome']
print(X)
print(Y)

#train test split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)
print('train test split: ', X.shape, X_train.shape, X_test.shape)

#traning the model
classifier= svm.SVC(kernel='linear')

#training the svm classifier
classifier.fit(X_train, Y_train)

#model evaluation
#accuracy score
X_train_prediction= classifier.predict(X_train)
training_data_accuracy= accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data: ', training_data_accuracy)

#accuracy score on the test data
X_test_prediction= classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data: ', test_data_accuracy)

#making a predictive system
input_data= (5,166,72,19,175,25.8,0.587,51)

#changing the input_data to numpy array
input_data_as_numpy_array= np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
std_data= scaler.transform(input_data_reshaped)
print(std_data)

prediction= classifier.predict(std_data)
print(prediction)

if(prediction[0] == 0):
    print('the person is not diabetic')
else:
    print('the person is diabetic')
