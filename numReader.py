import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np



mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


x_trainflat = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
x_testflat = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)


new_model = tf.keras.models.load_model('numreader.model')
predictions = new_model.predict(x_testflat)

print(predictions)
print(np.argmax(predictions[1]))

plt.imshow(x_test[1])
plt.show()



