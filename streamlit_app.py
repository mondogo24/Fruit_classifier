import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Clasificador de Frutas", page_icon="🍎")

MODEL_PATH = './fruit_classifier.keras' 
CLASSES = {0: 'Manzana', 1: 'Plátano', 2: 'Limón', 3: 'Naranja', 4: 'Pera'}

@st.cache_resource
def load_trained_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def preprocess_image(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    
    image = tf.io.decode_image(file_bytes, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [64, 64])
    image = (image / 127.5) - 1
    
    return tf.expand_dims(image, axis=0)

FRUIT_CLASSIFIER = load_trained_model()

st.title("🍎 Clasificador de Frutas con TensorFlow")

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    display_image = Image.open(uploaded_file)
    st.image(display_image, caption='Imagen cargada', use_container_width=True)
    
    if FRUIT_CLASSIFIER:
        if st.button('Clasificar'):
            with st.spinner('Procesando...'):
                input_data = preprocess_image(uploaded_file)
                
                prediction = FRUIT_CLASSIFIER.predict(input_data)
                predicted_index = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction)
                
                st.success(f"Predicción: **{CLASSES[predicted_index]}**")
                st.write(f"Confianza: {confidence:.2%}")
                
                with st.expander("Ver salida cruda del modelo"):
                    st.write(prediction)