import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

celcius = np.array([-40, -10, 0, 8, 15, 22, 38],
                   dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100],
                      dtype=float)

layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Training the model...")
history = model.fit(celcius, fahrenheit, epochs=1000, verbose=False)
print("Finished training the model.")

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

plt.show()


print("Making predictions using the model...")
result = model.predict([100.0])
print("The result is" + str(result) + " fahrenheit")
