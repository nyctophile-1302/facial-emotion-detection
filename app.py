import streamlit as st 
import cv2
from PIL import Image
import numpy as np 
import time
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf

from my_model.model import FacialExpressionModel

st.set_option('deprecation.showfileUploaderEncoding', False)
face_cascade = cv2.CascadeClassifier('frecog/haarcascade_frontalface_default.xml')
model = FacialExpressionModel("my_model/model.json", "my_model/model_weights.h5")

labels = {0 : 'contempt', 1 :'surprise', 2: 'anger', 3: 'disgust', 4: 'fear', 5: 'happiness', 6: 'sadness'}

    def choose_image_and_predict(img):
    img = img.convert('L')  # Convert image to grayscale
    img = img.resize((48, 48))  # Resize image to 48x48
    img_array = np.array(img)  # Convert image to numpy array
    img_array = np.stack([img_array]*3, axis=-1)  # Convert to 3 channels
    img_array_expanded = np.expand_dims(img_array, axis=0)  # Add another dimension for batch size
    x = preprocess_input(img_array_expanded)
    pred = model.predict_emotion(x)
    label = np.argmax(pred, axis=1)[0]
    return labels[label]



def main():
    activities = ["Home","Detect your Facial expressions" ,"CNN Model Performance","About"]
    choice = st.sidebar.selectbox("Select Activity",activities)

    if choice == 'Home':
        html_temp = """
        <marquee behavior="scroll" direction="left" width="100%;">
        <h2 style= "color: #000000; font-family: 'Raleway',sans-serif; font-size: 62px; font-weight: 800; line-height: 72px; margin: 0 0 24px; text-align: center; text-transform: uppercase;">Try your own test! </h2>
        </marquee><br>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.subheader("Video Demo :")
        st.video("https://www.youtube.com/watch?v=M1uyH-DzjGE&t=46s")
    
    if choice == 'CNN Model Performance':
        st.title("Face Expression WEB Application :")
        st.image('images/model.png', width=700)
        st.image('images/dataframe.png', width=700)
        st.image("images/accuracy.png")
        st.image("images/loss.png")
    
    if choice == 'Detect your Facial expressions':
        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)
            st.image(our_image)
        
        if image_file is None:
            st.error("No image uploaded yet")

        task = ["Faces"]
        feature_choice = st.sidebar.selectbox("Find Features",task)
        if st.button("Process"):
            if feature_choice == 'Faces':
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)
                    progress.progress(i+1)
        
        prediction = choose_image_and_predict(our_image)
        if prediction == 'contempt':
            st.subheader("Aww! you are Contempt :smile: today! ")
            st.video("https://www.youtube.com/watch?v=4q1dgn_C0AU&t=24s")
        elif prediction == 'surprise':
            st.subheader("You seem to be surprised today! ")
        elif prediction == 'anger':
            st.subheader("You seem to be angry today! ")
            st.video("https://www.youtube.com/watch?v=d_5DU5opOFk")
        elif prediction == 'disgust':
            st.subheader("You seem to be disgusted today! ")
        elif prediction == 'fear':
            st.subheader("You seem to be scared today! ")
            st.video("https://www.youtube.com/watch?v=h_D6HhWiTiI")
        elif prediction == 'happiness':
            st.subheader("You seem to be happy today! ")
            st.video("https://www.youtube.com/watch?v=4q1dgn_C0AU&t=24s")
        elif prediction == 'sadness':
            st.subheader("You seem to be sad today! ")
            st.video("https://www.youtube.com/watch?v=ST97BGCi3-w")
        else 
            st.subheader("No Emotion Detected")
    elif choice == 'About':
        st.title("Face Expression WEB Application :")
        st.subheader("About Face Expression Detection App")

main()
