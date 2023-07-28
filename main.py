import tensorflow as tf
from tensorflow.keras import layers
from keras.utils import to_categorical

# Load dataset using TensorFlow's ImageDataGenerator
image_size = (64, 64)
batch_size = 32

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    'sign_language_dataset',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',  # Set the color_mode to 'grayscale' for grayscale images
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'sign_language_dataset',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',  # Set the color_mode to 'grayscale' for grayscale images
    subset='validation'
)

# Convert labels to one-hot encoding
num_classes = 27
train_labels = to_categorical(train_generator.classes, num_classes=num_classes)
validation_labels = to_categorical(validation_generator.classes, num_classes=num_classes)

# Build the CNN model
model = tf.keras.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('sign_language_model.tflite', 'wb') as f:
    f.write(tflite_model)
# ... (previous code)

# Get class names from the data generator
class_names = list(train_generator.class_indices.keys())

# Save class names to 'labels.txt'
with open('labels.txt', 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

# ... (rest of the code)

