# import tensorflow as tf
# from tensorflow import keras
# import numpy as np

#
# class CustomModel(keras.Model):
#     def train_step(self, data):
#         # Unpack the data. Its structure depends on your model and
#         # on what you pass to `fit()`.
#         x, y = data
#
#         with tf.GradientTape() as tape:
#             y_pred = self(x, training=True)  # Forward pass
#             # Compute the loss value
#             # (the loss function is configured in `compile()`)
#             loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
#
#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         # Update metrics (includes the metric that tracks the loss)
#         self.compiled_metrics.update_state(y, y_pred)
#         # Return a dict mapping metric names to current value
#         return {m.name: m.result() for m in self.metrics}
#
#
# # Construct and compile an instance of CustomModel
# inputsx = keras.Input(shape=(32,))
# inputsy = keras.Input(shape=(1,))
# outputs = keras.layers.Dense(1)(inputsx)
# loss = tf.reduce_mean(tf.square(outputs - inputsy))
# model = CustomModel([inputsx, inputsy], outputs)
#
# model.compile(optimizer="adam", loss=loss,metrics=["mae"])
#
# # Just use `fit` as usual
# x = np.random.random((1000, 32))
# y = np.random.random((1000, 1))
# model.fit([x, y], epochs=3)

import tensorflow as tf
import numpy as np


class MyCustomMetricCallback(tf.keras.callbacks.Callback):

    def __init__(self, validation=None):
        super(MyCustomMetricCallback, self).__init__()
        self.validation = validation

    def on_epoch_end(self, epoch, logs=None):
        mse = tf.keras.losses.mean_squared_error

        if self.validation:
            logs['val_my_metric'] = float('inf')
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_pred = self.model.predict(X_valid)
            val_score = tf.reduce_mean(mse(y_pred, y_valid))  # val_score是(100,)的tensor，bar输出时会自动求平均
            logs['val_my_metric'] = val_score.numpy()

            # print(logs)  # my_metric_val是一个tensor

def build_model():
    inp1 = tf.keras.layers.Input((5,))
    inp2 = tf.keras.layers.Input((5,))
    out = tf.keras.layers.Concatenate()([inp1, inp2])
    out = tf.keras.layers.Dense(1)(out)

    model = tf.keras.models.Model([inp1, inp2], out)
    model.compile(loss='mse', optimizer='adam')

    return model


X_train1 = np.random.uniform(0, 1, (100, 5))
X_train2 = np.random.uniform(0, 1, (100, 5))
y_train = np.random.uniform(0, 1, (100, 1))

X_val1 = np.random.uniform(0, 1, (100, 5))
X_val2 = np.random.uniform(0, 1, (100, 5))
y_val = np.random.uniform(0, 1, (100, 1))

model = build_model()

history=model.fit([X_train1, X_train2], y_train, epochs=10,
                  validation_data=([X_val1, X_val2], y_val),
          callbacks=[MyCustomMetricCallback(validation=([X_val1, X_val2], y_val))])

print(history.history)