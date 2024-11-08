import pandas as pd
import numpy as np
import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #Training dataset
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') #Testing dataset

y_train = dftrain.pop('survived')
y_test = dfeval.pop('survived')

print(dftrain.iloc[0])

CATEGORICAL_COLUMNS = ['sex', 'class', 'deck', 'deck', 'embark_town', 'alone']
NUMERICAL_COLUMNS = ['age', 'n_siblings_spouses', 'parch', 'fare']

