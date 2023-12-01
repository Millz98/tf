import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNet
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

# Create a more advanced CNN model using MobileNet
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Resize images to match the MobileNet input shape
resized_train_images_rgb = tf.image.resize(train_images_rgb, (32, 32))
resized_test_images_rgb = tf.image.resize(test_images_rgb, (32, 32))

model = models.Sequential()

# Add the MobileNet base model
model.add(base_model)

# Add custom dense layers with dropout and batch normalization
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Compile the model with the legacy Adam optimizer
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

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
history = model.fit(datagen.flow(resized_train_images_rgb, train_labels, batch_size=32),
                    steps_per_epoch=len(train_images_rgb) / 32, epochs=20,
                    validation_data=(resized_test_images_rgb, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(resized_test_images_rgb, test_labels)
print(f'Test accuracy: {test_acc}')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
