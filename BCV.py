import streamlit as st
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to apply image processing techniques
def apply_image_processing(image, technique):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if technique == 'Canny':
        processed_image = cv2.Canny(gray_image, 50, 150)
    elif technique == 'Log':
        processed_image = cv2.Laplacian(gray_image, cv2.CV_64F)
        processed_image = np.uint8(np.absolute(processed_image))
    elif technique == 'Dog':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed_image = cv2.filter2D(gray_image, -1, kernel)
    else:
        processed_image = gray_image

    return processed_image

# Streamlit app
def main():
    st.title("Image Analyzer App")
    
    # Sidebar with options
    st.sidebar.title("Options")
    uploaded_image = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    techniques = st.sidebar.selectbox("Select Image Processing Technique", ['Original', 'Canny', 'Log', 'Dog'])

    if uploaded_image is not None:
        image = np.array(Image.open(uploaded_image))

        # Main content
        st.image(image, caption="Original Image", use_column_width=True)

        if techniques != 'Original':
            processed_image = apply_image_processing(image, techniques)
            st.image(processed_image, caption=f"{techniques} Processed Image", use_column_width=True)

# Run the Streamlit app
if __name__ == '__main__':
    main()

