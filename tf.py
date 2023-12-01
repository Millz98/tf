import tensorflow as tf
from keras import layers, models
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Convert labels to categorical one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# Create a simple CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
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
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    steps_per_epoch=len(train_images) / 32, epochs=10,
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Save model
model.save('my_model.keras')


# Load model
loaded_model = tf.keras.models.load_model('my_model.keras')

# Visualize predictions
for i in range(10):
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'Actual: {np.argmax(test_labels[i])}, Predicted: {np.argmax(model.predict(test_images[i][np.newaxis, :, :, np.newaxis]))}')
    plt.show()



# Load the image (replace 'Documents/Projects/tf/image2.png' with the actual path)
image_path = '/Users/davemills/Documents/Projects/tf/image4.png'
new_image = Image.open(image_path).convert('L')  # Convert to grayscale

# Preprocess the image (assuming your model expects 28x28 images)
new_image = new_image.resize((28, 28))
new_image = np.array(new_image) / 255.0  # Normalize pixel values to the range [0, 1]

# Add batch and channel dimensions to the image
new_image = np.expand_dims(new_image, axis=(0, -1))

# Print the current working directory
print("Current Directory:", os.getcwd())

# Print the files in the current directory
print("Files in Directory:", os.listdir())

# Make a prediction using the loaded model
predictions = loaded_model.predict(new_image)
predicted_label = np.argmax(predictions)

print(f'Predicted label: {predicted_label}')     

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()