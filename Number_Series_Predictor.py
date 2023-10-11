import tensorflow as tf
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
features = np.array([0, 2, 4, 6, 8, 10],dtype = float)# Enter the features of the series that you need to predict 
labels = np.array([2, 4, 6, 8, 10, 12], dtype = float)# Enter the labels of the series parllelly for training the model 
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(loss = "mean_squared_error", optimizer = tf.keras.optimizers.Adam(0.1))
his = model.fit(features, labels, epochs = 1000, verbose = False)
user_input = eval(input("Enter the value"))
predicted_solution = model.predict([user_input])
print(predicted_solution)
