import tensorflow as tf

lossobj=tf.keras.losses.BinaryCrossentropy()

y_true = [[0], [1], [0], [0]]
y_pred = [[-18.6], [0.51], [2.94], [-12.8]]
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
loss=bce(y_true, y_pred)
print(loss)