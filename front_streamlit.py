import requests
import streamlit as st
from PIL import Image
import io

# Definición de la URL de la API
API_URL = "http://127.0.0.1:8080/sentiment/image/" 
#API_URL = "ttp://127.0.0.1:8080/sentiment/image/" # Cambia la URL al nuevo endpoint si es necesario

# Mapeo de clases en español
class_to_idx = {
    0: 'enojo',
    1: 'desprecio',
    2: 'desagrado',
    3: 'miedo',
    4: 'felicidad',
    5: 'neutral',
    6: 'tristeza',
    7: 'sorpresa'
}

# Configuración de la página
st.title('Análisis de Expresiones Faciales')
st.markdown('A través de una foto te diremos qué expresión tiene la persona (formatos permitidos: png, jpeg): \n')

# Subir imagen
uploaded_file = st.file_uploader("Cargar tu imagen aquí:", type=["jpg", "jpeg", "png"])

if st.button('Cargar imágen'):
    if uploaded_file is not None:
        # Leer la imagen
        image = Image.open(uploaded_file)

        # Preparar los datos para enviar a la API
        buf = io.BytesIO()
        if uploaded_file.type == "image/png":
            image.save(buf, format='PNG')
            byte_image = buf.getvalue()
            files = {'file': ('image.png', byte_image, 'image/png')}
        else:
            image.save(buf, format='JPEG')
            byte_image = buf.getvalue()
            files = {'file': ('image.jpg', byte_image, 'image/jpeg')}

        # Hacer la solicitud POST a la API
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            results = response.json()
            expresion = class_to_idx[int(results['expresion'])]  # Convertir a entero y mapear
            st.markdown(f"Clasificación de expresión facial de la foto: {expresion}")
        else:
            st.error("Error en la respuesta de la API")
    else:
        st.warning("Por favor, carga una imagen para analizar.")
