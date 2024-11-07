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
print(f"Rank {tf.rank(m)} tensor (matrix) with shape: {m.shape} and looks like that: \n {m.numpy()}")

# Creating a tensor with shape [2, 2, 3] filled with ones
ones = tf.ones([2, 2, 3], tf.int64)

print(f"Rank {tf.rank(ones)} tensor (3D) with shape: {ones.shape} and looks like that: \n {ones.numpy()}")

#Reshaping tensor "ones" into tensor with shape [3, 2, 2]
ones_r = tf.reshape(ones, [3, 2, 2])

print(f"Rank {tf.rank(ones_r)} tensor with shape: {ones_r.shape} and looks like that: \n {ones_r.numpy()}")

#There is also another way were you just apply (-1) and it will reshape on its own
ones_r_1 = tf.reshape(ones, [3, -1])
ones_r_2 = tf.reshape(ones, [3, 2, -1])

print(f"Rank {tf.rank(ones_r_1)} tensor with shape: {ones_r_1.shape} and looks like that: \n {ones_r_1.numpy()}")
print(f"Rank {tf.rank(ones_r_2)} tensor with shape: {ones_r_2.shape} and looks like that: \n {ones_r_2.numpy()}")