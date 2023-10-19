import cv2
import numpy as np
import joblib

from skimage.transform import resize
from skimage.feature import hog

# Cargar el modelo entrenado
model = joblib.load('modelo_entrenado.joblib')

# Función para preprocesar la imagen de entrada
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = resize(img, (32, 32), anti_aliasing=True)
    hog_features = ExtractHOG(img)  # Extraer características HOG
    return hog_features

def ExtractHOG(img):
    ftr = hog(img, orientations=8, pixels_per_cell=(16, 16),
              cells_per_block=(1, 1), visualize=False)  # No necesitas visualizar en este caso
    return ftr


# Ruta de la imagen que deseas clasificar
input_image_path = 'ComidaDB/Pozole/Test/7.jpg'

# Preprocesar la imagen
preprocessed_image = preprocess_image(input_image_path)

# Realizar la clasificación
predicted_class = model.predict([preprocessed_image])

# Mapeo inverso del valor predicho a la clase original
class_mapping = {1: 'Torta', 2: 'Flauta', 3: 'Gordita', 4: 'Tamal', 5: 'Pozole'}
predicted_class_name = class_mapping[predicted_class[0]]

print(f"La imagen se clasificó como: {predicted_class_name}")
