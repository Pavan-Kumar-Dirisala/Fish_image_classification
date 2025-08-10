# Fish Image Classification with Deep Learning

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue)](https://www.python.org/)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange)](https://jupyter.org/)


This project uses deep learning models like ResNet18 and MobileNetV2 to classify images of fish species. The models are deployed using **Streamlit** and **Jupyter Notebooks** for exploration and experimentation.

## Jupyter Notebooks

[![View on NBViewer](https://img.shields.io/badge/View%20Notebook-NBViewer-brightgreen)](https://nbviewer.org/github/yourusername/reponame/blob/main/YourNotebook.ipynb)

This project includes Jupyter notebooks for preprocessing, model training, and evaluation. The notebooks provide step-by-step guidance on how the models are developed.

## Streamlit Application

[![View on Streamlit](https://img.shields.io/badge/View%20on-Streamlit-blueviolet)](https://yourstreamlitapplink)

This project is deployed using **Streamlit**, providing a web-based interface where users can upload fish images and get real-time classification predictions using the pre-trained models.


This project demonstrates a fish image classification system using deep learning models such as ResNet18, MobileNetV2, and custom CNN. It uses Streamlit for the user interface and Gradio Client to query a pre-trained model hosted on Hugging Face Spaces.

## Overview

This project is designed to classify fish species based on images. The models used for this classification include ResNet18, MobileNetV2, and a custom CNN. The system takes an image of a fish as input and returns the predicted species along with the confidence score.

The app is hosted using **Streamlit** and **Gradio** for a seamless interface. The models are deployed using **Hugging Face Spaces** to provide easy access to the pre-trained models.

## Features

- **Fish Species Classification**: Uses pre-trained models to classify images of fish into one of 11 categories.
- **Real-time Image Prediction**: Upload an image of a fish, and get the predicted species with the confidence score.
- **Multiple Model Support**: The app uses various models like ResNet18, MobileNetV2, and a custom CNN for prediction.
- **Simple Interface**: Built with Streamlit for an easy-to-use and intuitive web interface.

## Models Used

### 1. **ResNet18**
   - Pre-trained model fine-tuned on a custom fish dataset.
   - Efficient for image classification with robust feature extraction capabilities.

### 2. **MobileNetV2**
   - Lightweight model suitable for mobile and edge devices, optimized for fish image classification.

### 3. **Custom CNN**
   - A custom-built Convolutional Neural Network specifically designed for fish image classification.

These models were trained on a custom dataset and are accessible via a Hugging Face endpoint.

## Project Setup

### 1. Clone the Repository
Clone the repository to your local machine:

```bash
git clone https://github.com/Pavan-Kumar-Dirisala/Fish_image_classification.git
cd Fish_image_classification
````

### 2. Install Dependencies

Create a virtual environment and install the required dependencies:

```bash
# Create a virtual environment (Optional but recommended)
python3 -m venv venv
source venv/bin/activate  # For Linux or macOS
venv\Scripts\activate     # For Windows

# Install the dependencies
pip install -r requirements.txt
```

### 3. Running the App Locally

To run the app locally, use the following command:

```bash
streamlit run app.py
```

This will open the Streamlit app in your browser, and you can upload images of fish to classify them.

## Deployment

The app is deployed on **Streamlit Cloud** and can be accessed using the following link:

[**Fish Image Classification App**](https://fishimageclassification.streamlit.app)

## How It Works

1. **Upload Image**: The user uploads a fish image via the file uploader in the Streamlit app.
2. **Model Inference**: The app uses pre-trained models (ResNet18, MobileNetV2, and Custom CNN) to process the image and predict the fish species.
3. **Prediction and Confidence**: The app displays the predicted species along with the confidence score for each model.
4. **Display Results**: The result is shown with a visualization of the confidence levels for each model, helping users interpret the prediction quality.

## Example Usage

1. Go to the **[Fish Image Classification App](https://fishimageclassification.streamlit.app)**.
2. Upload an image of a fish (supported formats: JPG, JPEG, PNG).
3. View the prediction results, including the predicted species and confidence scores.

## Dependencies

* **Streamlit**: For building the interactive web application.
* **Gradio Client**: For querying the Hugging Face API.
* **Torch**: For loading the pre-trained models (ResNet18, MobileNetV2, Custom CNN).
* **Pandas, Numpy**: For data manipulation and processing.
* **Plotly**: For interactive visualizations.
* **Scikit-learn**: For model evaluation and metrics.
* **Matplotlib, Seaborn**: For visualizations and charts.
* **TensorFlow**: For training deep learning models if needed.

