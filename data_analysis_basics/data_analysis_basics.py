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

# Displays the graph that shows histogram of age with bins = 20
# Bins tells in what intervals data should be distributed
# In below example age is between 0 and 80 and bins are set to 20,
# therefore, the intervals will be in 4 (80 / 20 = 4).
# There two ways of doing it
# 1:
dftrain.age.hist(bins = 20)
plt.show()
# 2:
dftrain['age'].hist(bins = 20)
plt.show()

# It plots vertical bar graph that counts how many is
# there (females, males) in the column sex
dftrain.sex.value_counts().plot(kind = 'bar')
plt.show()

# It plots vertical bar graph that counts how many is
# there (females, males) in the column sex
dftrain['class'].value_counts().plot(kind = 'barh')
plt.show()

# pd.concat() concatenates (combines) two DataFrames along a specified axis.
# axis=1 specifies that the concatenation should occur horizontally,
# adding y_train as a new column in dftrain.

#.groupby('sex') groups the concatenated DataFrame by the sex column,
# creating two groups: one for Male and one for Female.

# .survived.mean() calculates for each gender group
# (Male and Female) the mean of the survived column.

# set_xlabel('% survive') adds a label to the x-axis to
# indicate that the bar values represent the percentage of survival.
pd.concat([dftrain, y_train], axis = 1).groupby('sex').survived.mean().plot(kind = 'barh').set_xlabel('% survive')
plt.show()

