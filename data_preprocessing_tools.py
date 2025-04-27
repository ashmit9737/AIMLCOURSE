# Data Preprocessing Tools

# Importing the libraries
import numpy as np  # Imports NumPy library for numerical operations, especially arrays and mathematical functions
import matplotlib.pyplot as plt  # Imports Matplotlib's plotting module for data visualization (not used in this code)
import pandas as pd  # Imports Pandas library for data manipulation and analysis, particularly for handling datasets

# Importing the dataset
dataset = pd.read_csv('Data.csv')  # Reads a CSV file named 'Data.csv' into a Pandas DataFrame
X = dataset.iloc[:, :-1].values  # Extracts all rows and all columns except the last one as the feature matrix (independent variables)
y = dataset.iloc[:, -1].values  # Extracts all rows of the last column as the target vector (dependent variable)
print(X)  # Prints the feature matrix to inspect its contents
print(y)  # Prints the target vector to inspect its contents

# Taking care of missing data
from sklearn.impute import SimpleImputer  # Imports SimpleImputer from scikit-learn to handle missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # Creates an imputer object that replaces NaN (missing values) with the mean of the column
imputer.fit(X[:, 1:3])  # Fits the imputer on columns 1 and 2 (index 1:3) to compute the mean for those columns
X[:, 1:3] = imputer.transform(X[:, 1:3])  # Replaces missing values in columns 1 and 2 with the computed means
print(X)  # Prints the updated feature matrix to verify missing values are handled

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer  # Imports ColumnTransformer to apply transformations to specific columns
from sklearn.preprocessing import OneHotEncoder  # Imports OneHotEncoder to convert categorical variables into binary (dummy) variables
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')  # Defines a transformer that applies OneHotEncoder to column 0 and passes other columns unchanged
X = np.array(ct.fit_transform(X))  # Applies the transformation, converting the categorical column into dummy variables and updating X
print(X)  # Prints the transformed feature matrix with encoded categorical variables
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder  # Imports LabelEncoder to convert categorical labels into numerical values
le = LabelEncoder()  # Creates a LabelEncoder object
y = le.fit_transform(y)  # Encodes the target variable (y) into numerical labels (e.g., 'Yes'/'No' to 1/0)
print(y)  # Prints the encoded target vector

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split  # Imports train_test_split to split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  # Splits data: 80% training, 20% testing, with a fixed random seed for reproducibility
print(X_train)  # Prints the training feature matrix
print(X_test)  # Prints the testing feature matrix
print(y_train)  # Prints the training target vector
print(y_test)  # Prints the testing target vector

# Feature Scaling
from sklearn.preprocessing import StandardScaler  # Imports StandardScaler to standardize features (mean=0, variance=1)
sc = StandardScaler()  # Creates a StandardScaler object
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])  # Fits the scaler on training data (columns 3 onward) and transforms them to standardized values
X_test[:, 3:] = sc.transform(X_test[:, 3:])  # Transforms the test data using the same scaler (without refitting) to avoid data leakage
print(X_train)  # Prints the scaled training feature matrix
print(X_test)  # Prints the scaled testing feature matrix
