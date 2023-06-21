import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from model import classify_image, preprocess_image

# Set the app title
st.markdown('<h1>Truce<span style="color: red;">Net</span> App</h1>', unsafe_allow_html=True)
# st.title('TruceNet App')

# Add a column for the README
st.sidebar.title('README')
with st.sidebar.container():
    st.markdown('''Deepfake Detection Model

This repository contains a deepfake detection model that can be used to determine whether an image is a deepfake or not. The model is built using a deep learning architecture and is trained on a large dataset of both real and deepfake images.

## Model Architecture

The major goal of this study is to quickly discern between deep fakes and genuine images. Several studies have
been conducted on the complex subject of ”deepfake.” To
detect deepfake photos, some researchers used feature-based
algorithms, while others employed CNN-based methods. To
recognise deep fake images, several of them applied machine
learning classifiers. This study is distinctive in that it uses
the CNN model, which has a 94.20% accuracy rate, to
discriminate between deep-fake images and real ones. The fact
that we used more CNN architectures in our work set us apart
from other academics. We also performed a thorough study,
and the results were better than we had anticipated.

## Dataset

The model is trained on a diverse dataset that consists of real images and deepfake images. The real images are sourced from various publicly available image datasets, while the deepfake images are generated using state-of-the-art deepfake generation techniques. The dataset is carefully curated to ensure a balanced distribution of real and deepfake images, allowing the model to learn discriminative features.

## Model Performance

The deepfake detection model achieves high accuracy in distinguishing between real and deepfake images. During the training phase, the model is evaluated on a separate validation set to monitor its performance. The model's performance metrics, such as accuracy, precision, recall, and F1 score, are measured to assess its effectiveness in detecting deepfakes.

## Usage

To use the deepfake detection model, follow the steps below:

1. Install the required dependencies by running `pip install -r requirements.txt`.

2. Run the steamlit app using `streamlit run app.py`.

3. Upload an image to the app to perform inference.

## Limitations

While the deepfake detection model achieves high accuracy, it may still have certain limitations. Some potential limitations include:

- The model's performance may vary depending on the quality of the deepfake images and the sophistication of the deepfake generation techniques.
- The model may struggle with detecting deepfake images that are generated using advanced adversarial methods specifically designed to evade detection.
- The model's performance may be affected by variations in lighting conditions, image resolution, and other factors that can impact the quality of the image.

## Contributing

Contributions to the deepfake detection model are welcome. If you have any suggestions, improvements, or bug fixes, feel free to submit a pull request. Please ensure that your contributions align with the repository's guidelines and standards.

## License

This deepfake detection model is released under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as per the terms of the license.
''')
    st.markdown('The app is built with Streamlit, TensorFlow, and PIL (Python Imaging Library).')

# Create a file uploader
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

# Perform inference when the file is uploaded
if uploaded_file is not None:
    # Open and preprocess the image
    image = Image.open(uploaded_file)
    st.subheader('Uploaded Image')
    st.image(image, use_column_width=True)
    predictions = classify_image(image)

    # Display the predictions
    st.subheader('Predictions:')
    for pred in predictions:
        class_label = pred[1]
        confidence = pred[2]
        st.write(f'{class_label}: {confidence:.2%}')
