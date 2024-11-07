import tensorflow as tf

string = tf.Variable("Hello there", tf.string)
number = tf.Variable(123, tf.int64)
floating = tf.Variable(3.123, tf.float64)

print(string)
print(number)
print(floating)