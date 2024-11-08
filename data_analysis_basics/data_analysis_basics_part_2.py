import pandas as pd
import numpy as np
import tensorflow as tf
from keras.src.layers import StringLookup

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #Training dataset
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') #Testing dataset

y_train = dftrain.pop('survived')
y_test = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'class', 'deck', 'deck', 'embark_town', 'alone']
NUMERICAL_COLUMNS = ['age', 'n_siblings_spouses', 'parch', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    print(vocabulary)
    lookup_layer = StringLookup(vocabulary=list(vocabulary), output_mode='int', name=f"{feature_name}")
    print(lookup_layer)
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    print(feature_columns)

