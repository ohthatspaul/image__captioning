import streamlit as st
import numpy as np
import os 
from os import listdir
from pickle import load
from keras.applications.vgg16 import VGG16
from keras.utils import load_img
from keras.utils import img_to_array
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras.utils import pad_sequences 
from collections import Counter
from io import BytesIO
from numpy import argmax
from PIL import Image, ImageOps


base_model = VGG16(include_top=True)
feature_extract_pred_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)


# cache imports

def extract_feature(model, file_name):
        img = load_img(file_name, target_size=(224, 224)) #size is 224,224 by default
        x = img_to_array(img) #change to np array
        x = np.expand_dims(x, axis=0) #expand to include batch dim at the beginning
        x = preprocess_input(x) #make input confirm to VGG16 input format
        fc2_features = model.predict(x)
        return fc2_features

# cache imports

def generate_caption(pred_model, caption_train_tokenizer, photo, max_length):
    in_text = '<START>'
    caption_text = list()
    for i in range(max_length):
            # integer encode input sequence
            sequence = caption_train_tokenizer.texts_to_sequences([in_text])[0]
            # pad input
            sequence = pad_sequences([sequence], maxlen=max_length)
            # predict next word
            model_softMax_output = pred_model.predict([photo,sequence], verbose=0)
            # convert probability to integer
            word_index = argmax(model_softMax_output)
            # map integer to word
            word = caption_train_tokenizer.index_word[word_index]
            #print(word)
            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += ' ' + word
            # stop if we predict the end of the sequence
            if word != 'end':
                caption_text.append(word)
            if word == 'end':
                break
    return caption_text

# Set containers
header = st.container()
dataset = st.container()
featureExtraction = st.container()
modelTraining = st.container()
modelValidation = st.container()
userValidation = st.container()
menuNavigation = st.container()


def main(): 

    with header:
        st.title('Image Captioning Using RNN + CNN ')
     

    with dataset:
        st.title("Dataset")
        st.text('Flicker8k_Dataset + Flicker8k_text')

    with featureExtraction:
        # load the tokenizer
        caption_train_tokenizer = load(open('caption_train_tokenizer.pkl', 'rb'))
        # pre-define the max sequence length (from training)
        max_length = 33
        pred_model = load_model('modelConcat_1_2.h5') 
        
    def upload_file():
        

        st.title("Try it out yourself!")

        caption_image_fileName = st.file_uploader("Upload an Image", type=["jpg","jpeg", "jpg","png"])
        if caption_image_fileName is not None:
             st.success("Image Uploaded")

        run = st.button("Caption this image")
        
        if run:
            
             photo = extract_feature(feature_extract_pred_model, caption_image_fileName)
             caption = generate_caption(pred_model, caption_train_tokenizer, photo, max_length)
             cap = (' '.join(caption))
             st.image(caption_image_fileName, caption=cap)

    with userValidation:
        st.text("Image and video option")
        upload_file()
        # camera_upload()

    

if __name__=='__main__':
     main()