import streamlit as st
import numpy as np
from pdf2image import convert_from_bytes
from task_processor.blob_diagram_detect import diagram_detector  # Replace 'your_module' with the actual module name

# Upload a file
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["pdf", "tif", "TIFF", "png", "jpeg"])

# Check if a file is uploaded
if uploaded_file:
    # Convert PDF to images
    images = convert_from_bytes(uploaded_file.read())

    # Check if images is a non-empty list
    if images:
        # Display the image using diagram_detector
        st.image(diagram_detector(image=np.array(images[1])), caption=f"Page {1}")
    else:
        st.warning("No valid images found in the uploaded file.")
else:
    st.info("Please upload a file.")
