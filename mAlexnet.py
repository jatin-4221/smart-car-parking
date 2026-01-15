import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set the path to your dataset (Update with the correct path)
dataset_path = './full_image_1000x750/'  # Google Drive path or local path

# Define ImageDataGenerator for loading and augmenting the dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Function to get subdirectories recursively (date/camera structure)

def get_image_data_dirs(base_dir):
    directories = []
    for weather in ['overcast', 'rainy', 'sunny']:
        weather_path = os.path.join(base_dir, weather)
        for date in os.listdir(weather_path):
            date_path = os.path.join(weather_path, date)
            if os.path.isdir(date_path):  # Check if it's a folder
                for camera in os.listdir(date_path):
                    camera_path = os.path.join(date_path, camera)
                    if os.path.isdir(camera_path):
                        directories.append(camera_path)
    return directories

# Get all directories that contain the images
image_dirs = get_image_data_dirs(dataset_path)

# Print out the directories to check the structure
print(image_dirs)

# Define the Minimal AlexNet model
def create_minialexnet(input_shape=(224, 224, 3), num_classes=5):
    model = models.Sequential()

    # Conv1
    model.add(layers.Conv2D(16, (11, 11), strides=4, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    # Conv2
    model.add(layers.Conv2D(20, (5, 5), strides=1, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    # Conv3
    model.add(layers.Conv2D(30, (3, 3), strides=1, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    # Fully Connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(48, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model
num_classes = len(image_dirs)  # Number of classes based on camera directories
model = create_minialexnet(input_shape=(224, 224, 3), num_classes=num_classes)

# Print model summary
model.summary()

# Use ImageDataGenerator to load and process images from the directories
train_generator = datagen.flow_from_directory(
    dataset_path,  # Use the main directory as root
    target_size=(224, 224),  # Resize images for your model (e.g., 224x224)
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    subset='training'  # For training data (validation will be used later)
)

# Validation data
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# If you want to fine-tune the model, unfreeze the base layers
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
model.fit(train_generator, validation_data=validation_generator, epochs=5)