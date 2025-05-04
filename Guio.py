import streamlit as st
st.set_page_config(
    page_title='Image Classifier',
    page_icon='*',
    layout='wide'

)
st.title("Imagenette Image Classifier using CustomCNN")
st.write("Upload an  image from Imageneter dataset,and the model will predict class ")
col1,col2,col3=st.columns([1,1,1])
with col1:
    st.header("upload image")
    uploaded_file = st.file_uploader("choose an image...",type=['jpg','png'])