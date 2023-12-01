import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Convert labels to categorical one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# Replicate single channel to create three channels
train_images_rgb = np.concatenate([train_images, train_images, train_images], axis=-1)
test_images_rgb = np.concatenate([test_images, test_images, test_images], axis=-1)

# Resize images to the expected shape
resized_train_images = tf.image.resize(train_images_rgb, (32, 32))
resized_test_images = tf.image.resize(test_images_rgb, (32, 32))

# Create a more advanced CNN model using transfer learning (VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

model = models.Sequential()

# Add the VGG16 base model
model.add(base_model)

# Add custom dense layers
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Adjust data augmentation input shape
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Train the model with data augmentation
history = model.fit(datagen.flow(resized_train_images, train_labels, batch_size=32),
                    steps_per_epoch=len(resized_train_images) / 32, epochs=10,
                    validation_data=(resized_test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(resized_test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
