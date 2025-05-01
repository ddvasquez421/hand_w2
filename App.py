import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# App
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

# Streamlit 
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')

# Estilo Lovecraftiano y terrorífico con CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Uncial+Antiqua&display=swap');
        
        body {
            background-color: #121212;  /* Fondo oscuro */
            color: #00FF00;  /* Verde neón como si fuera sobrenatural */
            font-family: 'Uncial Antiqua', cursive;
            text-shadow: 0 0 10px #FF0000, 0 0 20px #FF0000, 0 0 30px #FF0000;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FF0000;
            text-shadow: 0 0 10px #FF0000, 0 0 20px #FF0000, 0 0 30px #FF0000;
        }
        .stTextInput input {
            color: #FF0000;
            background-color: #333333;
            border: 2px solid #FF0000;
        }
        .stButton>button {
            color: #000000;
            background-color: #FF0000;
            border: 2px solid #FF0000;
        }
        .stSlider>div>div>input {
            background-color: #333333;
            color: #FF0000;
            border: 2px solid #FF0000;
        }
        .stTextArea>textarea {
            color: #FF0000;
            background-color: #333333;
            border: 2px solid #FF0000;
        }
        .stCanvas {
            border: 2px solid #FF0000;
            box-shadow: 0px 0px 30px 10px rgba(255, 0, 0, 0.5);  /* Sombra de terror */
        }
    </style>
""", unsafe_allow_html=True)

# Título y subtítulo
st.title('El Abismo del Reconocimiento de Dígitos')
st.subheader("Dibuja un dígito y prepárate para la revelación")

# Add canvas component
drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FF0000'  # Rojo brillante para el trazo (terrorífico)
bg_color = '#121212'  # Fondo negro

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(0, 255, 0, 0.3)",  # Relleno verde tenue, asociado con lo sobrenatural
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# Botón de predicción
if st.button('Predecir'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.header('El Digito es : ' + str(res))
    else:
        st.header('Por favor dibuja el dígito en el lienzo.')

# Barra lateral (sidebar) con efectos de texto de terror
st.sidebar.title("¿Qué horror acecha en la oscuridad?")
st.sidebar.text("Esta aplicación evalúa la capacidad de un RNA de reconocer")
st.sidebar.text("los dígitos escritos a mano, una pequeña muestra del caos.")
st.sidebar.text("Desarrollado por Vinay Uniyal, con la bendición de lo oscuro.")
#st.sidebar.write("[GitHub Repo Link](https://github.com/Vinay2022/Handwritten-Digit-Recognition)")
