import tensorflow.compat.v1 as tf
import cv2
import numpy as np

# Disable TensorFlow 2.x behavior
tf.disable_v2_behavior()

# Define the model path and tags for loading the SavedModel
model_path = './tf_model/saved_model.pb'  # Replace with your model's path

# Load the model with the correct parameters
# 'serve' is typically the tag used for serving models
tags = ['serve']
model = tf.saved_model.load(model_path, tags)

# Define the inference function
def predict_car(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Resize the image to match the input size expected by the model
    image_resized = cv2.resize(image, (256, 256))

    # Convert the image to RGB (OpenCV loads in BGR by default)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Normalize the image to the range [0, 1] if needed
    image_normalized = image_rgb / 255.0

    # Add batch dimension (models usually expect a batch of images)
    image_batch = np.expand_dims(image_normalized, axis=0)

    # Convert the batch to a tensor
    input_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    # Run the image through the model to get predictions
    infer = model.signatures["serving_default"]
    output = infer(input_tensor)

    # Get the predicted class (assuming binary classification: car vs. no car)
    output_tensor = output['dense_1/BiasAdd']  # Adjust according to the actual model's signature
    prediction = tf.argmax(output_tensor, axis=-1)

    # Return the predicted class
    return prediction.numpy()

# Test the function with an image
image_path = './path/to/your/image.jpg'  # Replace with your image's path
prediction = predict_car(image_path)

# Print the prediction
if prediction == 1:
    print("Car detected!")
else:
    print("No car detected.")
