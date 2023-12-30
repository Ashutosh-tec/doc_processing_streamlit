import io
# import fitz 
import uuid
import requests
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO
from pdf2image import convert_from_bytes
                

# Custom title
st.set_page_config(page_title="About", page_icon=":apple:")

def main():
    """
    This is the main streamlit application.
    """
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #4d285b;
        }
    </style>
    """, unsafe_allow_html=True)
    # Custom title for the page
    
    st.title("About")
    st.write("Hello everyone, welcome to document processing streamlit application.")
    st.write("PDF, TIFF, PNG, JPG etc can be processed here.")
    st.write("Tasks to be done:")
    st.write("1. Face detection")
    st.write("2. Entity detection")

if __name__ == "__main__":
    main()

