import tensorflow as tf

# Creating rank 0 tensor (scalar) that is containing datatype string
string = tf.Variable("Hello there", tf.string)

# Creating rank 0 tensor (scalar) that is containing datatype int
number = tf.Variable(123, tf.int64)

# Creating rank 0 tensor (scalar) that is containing datatype float
floating = tf.Variable(3.123, tf.float64)

print(f"Rank 0 tensor (scalar) containing string: {string}")
print(f"Rank 0 tensor (scalar) containing string: {number}")
print(f"Rank 0 tensor (scalar) containing float: {floating}")