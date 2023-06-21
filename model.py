import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np

# Load pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Define image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match input size of the model
    image = image.convert('RGB')
    image = preprocess_input(np.array(image))
    return image

# Define image classification function
def classify_image(image):
    processed_image = preprocess_image(image)
    inputs = tf.expand_dims(processed_image, axis=0)
    preds = model.predict(inputs)
    decoded_preds = decode_predictions(preds, top=3)[0]  # Get top 3 predictions
    return decoded_preds

# Load and classify an example image
# example_image = Image.open('test.png')
# predictions = classify_image(example_image)
