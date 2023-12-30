import cv2
import fitz 
import numpy as np
from PIL import Image
import streamlit as st
from pdf2image import convert_from_bytes

from task_processor.face_detection_cv2.cv2_cascadeClassifier import face_detect_cv2
from task_processor.blob_diagram_detect import diagram_detector

Image.MAX_IMAGE_PIXELS = None

if 'page_session' not in st.session_state:
    st.session_state.page_session = 0

# Custom title
st.set_page_config(page_title="Process Docs", page_icon=":banana:")



st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #4d285b;
        }
        [class=main st-emotion-cache-uf99v8 ea3mdgi3] {
            background-color: #4d285b;
            
        }
        [data-testid=block-container] {
            background-color: #4d285b;
            
        }
    </style>
    """, unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["pdf", "tif", "TIFF", "png", "jpeg"])
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith((".tif", ".tiff")):
        # Read the TIFF file using PyMuPDF
        tiff_data = uploaded_file.read()
        pdf_document = fitz.open(stream=tiff_data, filetype="pdf")
        
        # Convert the pages of the multipage TIFF to PNG images
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_pixmap()
            img = Image.frombytes("RGB", [image_list.width, image_list.height], image_list.samples)
            images.append(img)

    elif uploaded_file.name.lower().endswith((".pdf")):
        # Convert the uploaded PDF to images
        images = convert_from_bytes(uploaded_file.read())
    else: # take the file as png, jpg etc
        # Convert the uploaded file to a numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Decode the byte array to an image
        # images = [cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)]  # You can use cv2.IMREAD_GRAYSCALE for grayscale images
        images = [cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)]  # You can use cv2.IMREAD_GRAYSCALE for grayscale images
        
    # Ensure page_session stays within the valid range
    st.session_state.page_session = max(0, min(st.session_state.page_session, len(images) - 1))
    
    # image = np.array(images[st.session_state.page_session])
    
    option_list = ['Original Image', 'Entity Detection', 'Face Detection']
    check_result = st.sidebar.selectbox("**Select Result**", option_list, index=0)      
    if check_result == "Original Image":
            st.image(cv2.cvtColor(images[st.session_state.page_session], cv2.COLOR_RGB2BGR), caption=f"Page {st.session_state.page_session + 1}")
            # images[st.session_state.page_session].save('temp.png')
        
    elif check_result == 'Face Detection':
        st.image(face_detect_cv2(images[st.session_state.page_session]), caption=f"Page {st.session_state.page_session + 1}")
    
    elif check_result == 'Entity Detection':   
        st.write("Note: Here the image will be converted to Gray scale.")     
        st.image(diagram_detector(image = np.array(images[st.session_state.page_session])), caption=f"Page {st.session_state.page_session + 1}")
        

    col1, _, col3 = st.columns(3)
    # Content for column 1
    with col3:
        if st.session_state.page_session != len(images)-1:
            next_but = st.button('Next Page')
        else:
            next_but = False
        
    with col1:
        if st.session_state.page_session != 0:
            previous_but = st.button('Previous Page')
        else:
            previous_but = False

    # Managing next button and previous button for multi page file.
    if next_but and st.session_state.page_session < len(images) - 1:
        st.session_state.page_session += 1
        st.rerun()

    if previous_but and st.session_state.page_session > 0:
        st.session_state.page_session -= 1
        st.rerun()