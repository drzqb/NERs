import tensorflow as tf

a = tf.random.normal([2, 3, 4])
b = tf.random.normal([2, 3, 4])
print(a)
print(b)
d = tf.stack([a, b], axis=1)

print(d)
