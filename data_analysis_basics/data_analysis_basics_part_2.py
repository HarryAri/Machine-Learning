import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.src.layers import StringLookup, CategoryEncoding

# Load datasets
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # Training dataset
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')  # Testing dataset

# Separate target variable
y_train = dftrain.pop('survived')
y_test = dfeval.pop('survived')

# Define categorical and numerical columns
CATEGORICAL_COLUMNS = ['sex', 'class', 'deck', 'embark_town', 'alone']
NUMERICAL_COLUMNS = ['age', 'n_siblings_spouses', 'parch', 'fare']

# Scale numerical columns
scaler = StandardScaler()
dftrain[NUMERICAL_COLUMNS] = scaler.fit_transform(dftrain[NUMERICAL_COLUMNS])
dfeval[NUMERICAL_COLUMNS] = scaler.transform(dfeval[NUMERICAL_COLUMNS])

# Define inputs and preprocess categorical features
inputs = {}
preprocessed_features = []

# Loop through each categorical column
for feature_name in CATEGORICAL_COLUMNS:
    # Create an input layer for each feature
    inputs[feature_name] = tf.keras.Input(shape=(1,), name=feature_name, dtype=tf.string)

    # Use StringLookup to convert categorical strings to integer indices
    vocabulary = dftrain[feature_name].unique()
    lookup = StringLookup(vocabulary=list(vocabulary), output_mode="int")

    # Apply lookup and encoding layers in sequence
    feature = lookup(inputs[feature_name])  # Convert to integer index
    feature = CategoryEncoding(num_tokens=len(vocabulary) + 1, output_mode="one_hot")(feature)  # One-hot encoding

    # Append the preprocessed feature to the list
    preprocessed_features.append(feature)

# Handle numerical features by creating input layers for each
for feature_name in NUMERICAL_COLUMNS:
    inputs[feature_name] = tf.keras.Input(shape=(1,), name=feature_name, dtype=tf.float32)
    feature = inputs[feature_name]
    preprocessed_features.append(feature)

# Concatenate all preprocessed features
concatenated_features = tf.keras.layers.Concatenate()(preprocessed_features)

# Define the rest of the model
x = tf.keras.layers.Dense(64, activation='relu')(concatenated_features)  # First hidden layer
x = tf.keras.layers.Dropout(0.2)(x)  # Dropout layer for regularization
x = tf.keras.layers.Dense(32, activation='relu')(x)  # Second hidden layer
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

# Build the model
model = tf.keras.Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Summarize the model architecture
model.summary()

# Define early stopping callback to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Prepare inputs for training
train_inputs = {name: dftrain[name] for name in CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS}
eval_inputs = {name: dfeval[name] for name in CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS}

# Train the model
history = model.fit(train_inputs, y_train, epochs=50, batch_size=32,
                    validation_data=(eval_inputs, y_test), callbacks=[early_stopping])

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(eval_inputs, y_test)
print(f'Test accuracy: {test_acc}')