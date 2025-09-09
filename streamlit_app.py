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
# Ensure filter_size is always odd
filter_size = filter_size if filter_size % 2 != 0 else filter_size + 1
sobel_ksize = st.sidebar.slider("Sobel Kernel Size", 3, 11, 3, step=2, help="Adjust the size of the Sobel kernel.")

# Disclaimer
st.sidebar.markdown("⚠️ **Disclaimer:** Please do not upload any sensitive or confidential data. This application is for demonstration purposes only.")

# Define custom horizontal and vertical edge filters dynamically
def create_filters(size):
    print(f"Creating filters with size: {size}")  # Debugging line
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

# Load images based on selection
if image_source == "Fluorescence (IF)":
    try:
        image = Image.open('assets/IFCells.jpg')
        image = np.array(image)
        print(f"Image IF loaded with size: {image.shape}")
        image_float32 = image.astype(np.float32)  # Convert to float32

        IF_horiz = cv2.filter2D(image_float32, -1, horiz_filter)
        IF_vert = cv2.filter2D(image_float32, -1, vert_filter)
        IF_sobelx = cv2.Sobel(image_float32, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        IF_sobely = cv2.Sobel(image_float32, cv2.CV_64F, 0, 1, ksize=sobel_ksize)

        # Convert back to uint8 for display, scaling to 0-255
        IF_horiz = np.uint8(IF_horiz)
        IF_vert = np.uint8(IF_vert)
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

    except FileNotFoundError as e:
        st.error(f"Image not found: {e}")
        st.stop()
elif image_source == "Brightfield (BF)":
    st.subheader('Brightfield Image')
    try:
        image = Image.open('assets/BloodSmear.png')
        image = np.array(image)
        print(f"Brightfield image loaded with shape: {image.shape}")
        image_float32 = image.astype(np.float32)  # Convert to float32

        BF_horiz = cv2.filter2D(image_float32, -1, horiz_filter)
        BF_vert = cv2.filter2D(image_float32, -1, vert_filter)
        BF_sobelx = cv2.Sobel(image_float32, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        BF_sobely = cv2.Sobel(image_float32, cv2.CV_64F, 0, 1, ksize=sobel_ksize)

        # Normalize to 0-255 and convert to uint8
        BF_horiz = np.uint8(np.clip(BF_horiz, 0, 255))
        BF_vert = np.uint8(np.clip(BF_vert, 0, 255))
        BF_sobelx = np.uint8(np.clip(np.absolute(BF_sobelx), 0, 255))
        BF_sobely = np.uint8(np.clip(np.absolute(BF_sobely), 0, 255))

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

    except FileNotFoundError as e:
        st.error(f"Error loading Brightfield image: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        print(f"Uploaded image loaded with size: {image.shape}")
    else:
        st.warning("Please upload an image.")
        st.stop()