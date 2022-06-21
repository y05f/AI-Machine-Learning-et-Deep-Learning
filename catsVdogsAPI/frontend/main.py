import requests
import streamlit as st
from PIL import Image


# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Cats vs Dogs web app")

# displays a file uploader widget
image = st.file_uploader("Choose an image")

# displays a button
if st.button("Upload"):
    if image is not None :
        files = {"file": image.getvalue()}
        res = requests.post(f"http://backend:8080/catsVdogs/predict", files=files)
        output = res.json()
        if output['model-prediction'] == 'cat':
            label = 'cat'
        elif output['model-prediction'] == 'dog':
            label = 'dog'
        st.header(label)
        st.image(image, width=500)
