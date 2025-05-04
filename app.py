import streamlit as st
import pickle
import numpy as np
from PIL import Image
import joblib 
import time

# Load the trained model
# with open("./model.pkl", "rb") as model_file:
#     model = pickle.load(model_file)
model = joblib.load("./model.pkl")

# Streamlit app title
st.set_page_config(page_title="Iris Species Classifier", page_icon="🌼", layout="wide")
st.title("🌼 Iris Species Classifier 🌼")
st.write("Enter the flower measurements to classify the species.")

# User input for flower measurements
sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, step=0.1)

# Improved Sidebar Styling
st.sidebar.title("ℹ️ Model Information")
st.sidebar.markdown(
    """
    This is a simple **Iris flower classification model** using machine learning.

    The model is trained on the famous **Iris dataset**, which contains measurements of:
    - Sepal length
    - Sepal width
    - Petal length
    - Petal width

    **Accuracy:** :green[97%]

    The dataset includes **3 classes** of iris flowers:
    - Setosa
    - Versicolor
    - Virginica

    Learn more about the Iris dataset [here](https://en.wikipedia.org/wiki/Iris_flower_data_set).
    """,
    unsafe_allow_html=True,
)

# Prediction button
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(features)

    with st.spinner("Predicting..."):
        time.sleep(2)

    if prediction[0] == 0:
        st.success("Predicted Iris Species: Setosa")
    elif prediction[0] == 1:
        st.success("Predicted Iris Species: Versicolor")
    elif prediction[0] == 2:
        st.success("Predicted Iris Species: Virginica")
    else:
        st.error("Prediction Error: Unknown Species")

# Define the desired height
desired_height = 200  # Adjust this value as needed

# Function to resize image while maintaining aspect ratio
def resize_image(img, height):
    width = int(img.width * (height / img.height))
    resized_img = img.resize((width, height))
    return resized_img

# Load and resize images
image_setosa = Image.open("C:/Users/JOHN/Downloads/DataScienceR/session 15/Iris_Predicter/Iris_setosa.jpg")
image_versicolor = Image.open("C:/Users/JOHN/Downloads/DataScienceR/session 15/Iris_Predicter/Iris_versicolor_3.jpg")
image_virginica = Image.open("C:/Users/JOHN/Downloads/DataScienceR/session 15/Iris_Predicter/Iris_virginica.jpg")

resized_setosa = resize_image(image_setosa, desired_height)
resized_versicolor = resize_image(image_versicolor, desired_height)
resized_virginica = resize_image(image_virginica, desired_height)

# Create three columns for images
col1, col2, col3 = st.columns(3)

# Display resized images in columns
with col1:
    st.image(resized_setosa, caption="Setosa", use_container_width=True)

with col2:
    st.image(resized_versicolor, caption="Versicolor", use_container_width=True)

with col3:
    st.image(resized_virginica, caption="Virginica", use_container_width=True)

# Copyright notice
st.markdown(
    """
    ---
    <p style="text-align: center;">
        &copy; 2025, Iris Species Classifier. All rights reserved.
        <br>
        This application utilizes a machine learning model trained on the Iris dataset for classification purposes.
    </p>
    """,
    unsafe_allow_html=True,
)       