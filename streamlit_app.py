import streamlit as st
import numpy as np
from PIL import Image
import cv2  # Import OpenCV here, it's now required

# Sidebar controls
st.sidebar.header("Image Processing Parameters")
image_source = st.sidebar.radio(
    "Select Image Source:",
    ("Fluorescence (IF)", "Upload Image"),
    help="Choose between pre-loaded images or upload your own."
)

# Filter Size (initial value)
filter_size = 3

# Load images based on selection
if image_source == "Fluorescence (IF)":
    try:
        image = Image.open('assets/IFCells.jpg')
        image = np.array(image)
        print(f"Image IF loaded with size: {image.shape}")
    except FileNotFoundError as e:
        st.error(f"Image not found: {e}")
        st.stop()
else:  # Upload Image
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        print(f"Uploaded image loaded with size: {image.shape}")
    else:
        st.warning("Please upload an image.")
        st.stop()

# Define a simple 3x3 averaging filter
def apply_average_filter(img):
    kernel = np.ones((3, 3), np.float32) / 9
    filtered_img = cv2.filter2D(img, -1, kernel)
    return filtered_img

# Apply Sobel Filter
def apply_sobel_filter(img, ksize):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    return sobelx, sobely

# Apply filters
sobel_ksize = 3
sobelx, sobely = apply_sobel_filter(image, sobel_ksize)

# Display results
st.image(image, caption="Original Image")
st.image(np.uint8(np.absolute(sobelx)), caption="Sobel X")
st.image(np.uint8(np.absolute(sobely)), caption="Sobel Y")