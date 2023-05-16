import streamlit as st
import numpy as np
import tensorflow as tf
import h5py
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps


model = load_model('model (1).h5')

st.set_page_config(page_title='Handwritten Alphabet Identification', layout='wide')
st.header("Handwritten Alphabet Identification")

st.write("This is a CNN model which is trained to classify images of handwritten letters into the right english alphabets")

image = Image.open('Handwritten Alphabets.jpg')

st.image(image)

file = st.file_uploader("Kindly upload an image here", type = ['jpg', 'png'])

def pred(img):
    size = (28, 28,)
    img = ImageOps.fit(img, size, Image.ANTIALIAS).convert('L')
    img = np.array(img)
    img = img/255.
    img_reshape = img[np.newaxis, ...]
    p = model.predict(img_reshape)

    return p

if file is not None:
    img = Image.open(file)
    st.image(img, width = 300 )
    p = pred(img)
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    s = "This Image is Most Likely a : "+class_names[np.argmax(p)]
    st.success(s)