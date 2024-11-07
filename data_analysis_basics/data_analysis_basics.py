import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# pd.read_csv() stores the certain data as a dataframe inside a variable
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #Training dataset
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') #Testing dataset
print(dftrain)

# .pop() removes the column from the data and stores it in variable
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print(y_train)
print(dftrain.head())

# This is different ways of accessing the specific columns or rows from dataframe
print(f"This is the first row of data: \n {dftrain.loc[0]}")
print(f"This is the first row of data: \n {dftrain.iloc[0]}")
print(f"This is the age column of data: \n {dftrain["age"]}")
print(f"This is the age column of data: \n {dftrain.iloc[:, 1]}")

#  This prints both the first row of feature data (dftrain.loc[0])
#  and the target value (y_train.loc[0]) for the first row.
print(f" It is: \n {dftrain.loc[0], y_train.loc[0]}")

#.describe() gives us a information about a mean, median, std, IQR of the dataframe for each column
print(dftrain.describe())
print(dftrain.shape)

dftrain.age.hist(bins = 20)
plt.show()