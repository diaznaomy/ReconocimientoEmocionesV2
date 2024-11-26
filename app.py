from fastapi import FastAPI, HTTPException, File, UploadFile
import torch
import timm
from torchvision import transforms
import numpy as np
from PIL import Image
import io

# Inicializar el modelo ViT con timm
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=8)
model.load_state_dict(torch.load('VIT-modelo.pth'))  # Cargar los pesos (asegúrate de que los pesos coincidan)
model.eval()  # Cambiar a modo evaluación

# Inicializar la API
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API de reconocimiento de emociones está funcionando."}
# Definir las transformaciones para las imágenes
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar las imágenes
    transforms.ToTensor(),  # Convertir a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización
])

# Nuevo endpoint para análisis de imágenes
@app.post("/sentiment/image/")
async def predict_sentiment_from_image(file: UploadFile = File(...)):
    try:
        # Leer la imagen desde el archivo
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Aplicar las transformaciones a la imagen
        img_tensor = data_transforms(image).unsqueeze(0)  # Añadir dimensión de batch

        # Realizar la predicción
        with torch.no_grad():
            predictions = model(img_tensor)
            expresion = torch.argmax(predictions, dim=1).item()  # Obtener la clase con mayor probabilidad

        # Devolver el resultado
        return {"expresion": int(expresion)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
