#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 

st.header('Image Classification Model')
# Load the model
model_path = "C:\\Users\\SIDDHARTH\\image_classify.keras"
model = load_model(model_path)
data_cat = ['Corrosion', 'Cracked_tooth', 'Fatigue', 'Healthy']
img_height = 180
img_width = 180
image =st.text_input('Enter Image name',"C:\\Users\\SIDDHARTH\\Gear_Dataset\\test\\Fatigue\\gr1.jpg")

image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
st.write('With accuracy of ' + str(np.max(score)*100))


# In[ ]:




