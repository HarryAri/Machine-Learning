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

# Creating rank 1 tensor (vector = 1D) that is containing datatype int
v = tf.Variable([1, 2, 3, 4], tf.int64)
# Creating rank 2 tensor (matrix = 2D) that is containing datatype int
m = tf.Variable([[1, 2, 3], [1, 2, 3], [1, 2, 3]], tf.int64)

print(f"Rank {tf.rank(v)} tensor (vector) with shape: {v.shape} and looks like that: {v.numpy()}")
print(f"Rank {tf.rank(m)} tensor (matrix) with shape: {m.shape} and looks like that: \n {m.numpy()}" )