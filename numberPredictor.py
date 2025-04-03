import tensorflow as tf
import tensorflow as keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


train_images = train_images / 255.0 
test_images = test_images / 255.0


model = models.Sequential([
 layers.Flatten(input_shape=(28, 28)),  
      layers.Dense(128, activation='relu'),  
    layers.Dense(10, activation='softmax')  
])


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

predictions = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"True: {test_labels[i]}")
    plt.title(f"Prediction: {predictions[i].argmax()}")
    plt.show()