import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Sidebar controls
st.sidebar.header("Edge Detection Parameters")
image_source = st.sidebar.radio(
    "Select Image Source:",
    ("Fluorescence (IF)", "Brightfield (BF)", "Upload Image"),
    help="Choose between pre-loaded images or upload your own."
)
filter_size = st.sidebar.slider("Filter Size", 3, 9, 3, step=2, help="Adjust the size of the filter kernel.")
sobel_ksize = st.sidebar.slider("Sobel Kernel Size", 3, 11, 3, step=2, help="Adjust the size of the Sobel kernel.")

# Disclaimer
st.sidebar.markdown("⚠️ **Disclaimer:** Please do not upload any sensitive or confidential data. This application is for demonstration purposes only.")

# Load images based on selection
if image_source == "Fluorescence (IF)":
    try:
        image = Image.open('assets/IFCells.jpg')
        image = np.array(image)
        print(f"Image IF loaded with size: {image.shape}")
    except FileNotFoundError as e:
        st.error(f"Image not found: {e}")
        st.stop()

elif image_source == "Brightfield (BF)":
    try:
        image = Image.open('assets/BloodSmear.png')
        image = np.array(image)
        print(f"Image BF loaded with size: {image.shape}")
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

# Define custom horizontal and vertical edge filters dynamically
def create_filters(size):
    horiz_filter = np.zeros((size, size), dtype=np.float32)
    horiz_filter[size // 2, :] = [1, 1, 1]
    horiz_filter[size // 2 - 1, :] = [0, 0, 0]
    horiz_filter[size // 2 + 1, :] = [-1, -1, -1]

    vert_filter = np.zeros((size, size), dtype=np.float32)
    vert_filter[:, size // 2] = [1, 1, 1]
    vert_filter[:, size // 2 - 1] = [0, 0, 0]
    vert_filter[:, size // 2 + 1] = [-1, -1, -1]

    return horiz_filter, vert_filter

horiz_filter, vert_filter = create_filters(filter_size)

# Apply filters and display for Fluorescence Image
if image_source == "Fluorescence (IF)":
    st.subheader('Fluorescence Image')

    IF_horiz = cv2.filter2D(image, -1, horiz_filter)
    IF_vert = cv2.filter2D(image, -1, vert_filter)
    IF_sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    IF_sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_ksize)

    # Convert back to uint8 for display
    IF_sobelx = np.uint8(np.absolute(IF_sobelx))
    IF_sobely = np.uint8(np.absolute(IF_sobely))

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(image, channels='RGB', caption='Original')
    with col2:
        st.image(IF_horiz, channels='RGB', caption='Horizontal')
    with col3:
        st.image(IF_vert, channels='RGB', caption='Vertical')
    with col4:
        st.image(IF_sobelx, channels='RGB', caption='Sobel X')
    with col5:
        st.image(IF_sobely, channels='RGB', caption='Sobel Y')

# Apply filters and display for Brightfield Image
if image_source == "Brightfield (BF)":
    st.subheader('Brightfield Image')

    # Load Brightfield Image
    image = Image.open('assets/BloodSmear.png')
    image = np.array(image)

    # Apply filters to Brightfield Image
    BF_horiz = cv2.filter2D(image, -1, horiz_filter)
    BF_vert = cv2.filter2D(image, -1, vert_filter)
    BF_sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    BF_sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_ksize)

    # Convert back to uint8 for display
    BF_sobelx = np.uint8(np.absolute(BF_sobelx))
    BF_sobely = np.uint8(np.absolute(BF_sobely))

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(image, channels='RGB', caption='Original')
    with col2:
        st.image(BF_horiz, channels='RGB', caption='Horizontal')
    with col3:
        st.image(BF_vert, channels='RGB', caption='Vertical')
    with col4:
        st.image(BF_sobelx, channels='RGB', caption='Sobel X')
    with col5:
        st.image(BF_sobely, channels='RGB', caption='Sobel Y')