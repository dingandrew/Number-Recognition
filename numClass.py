import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
print(tf.VERSION)

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train[0])
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
print(y_train[0])

print(x_test[0])
plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()
print(y_test[0])

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

print(x_train[0])
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()

x_trainflat = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
x_testflat = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

model = tf.keras.models.Sequential()


model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
	                    metrics=['accuracy'])

model.fit(x_trainflat, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_testflat, y_test)
print(val_loss)
print(val_acc)

#saves the model as .hd5h file format
model.save('numreader.model')



