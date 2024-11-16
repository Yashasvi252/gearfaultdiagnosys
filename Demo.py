import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np


st.header('Faulty Gear Identifier')
# Load the model
model_path = "/content/drive/MyDrive/Colab Notebooks/Faulty gear classification/Gear_Dataset/Gears.keras"
model = load_model(model_path)
data_cat = ['Corrosion', 'Cracked_tooth', 'Fatigue', 'Healthy']
img_height = 180
img_width = 180
image =st.text_input('Enter Image Path',"/content/drive/MyDrive/Colab Notebooks/Faulty gear classification/Gear_Dataset/test/Corrosion/C41.jpg")
image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
if data_cat[np.argmax(score)] == 'Healthy':
  st.write('No Defect found in Gear : Gear is Healty')
else:
  st.write('The Defect in Gear is ' + data_cat[np.argmax(score)])
  st.write('With accuracy of ' + str(np.max(score) * 100) + '%')