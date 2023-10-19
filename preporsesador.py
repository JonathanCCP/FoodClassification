import cv2
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.feature import hog
import glob

import joblib


def ExtractHOG(img):
    ftr, _ = hog(img, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=True)
    return ftr

def preprocessing_part_two(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = resize(img_gray, (32, 32), anti_aliasing=True)
    hog_features = list(ExtractHOG(img_resized))
    return hog_features

def process_single_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_features = preprocessing_part_two(img)
    return img_features

def predict_single_image(image, model):
    prediction = model.predict([image])
    return prediction

def doPredict( img ):
    # Load your model
    model = joblib.load('modelo_entrenado.joblib')

    
    # Make a prediction for the single image
    prediction = predict_single_image(img, model)

    return prediction
# Example usage
if __name__ == "__main__":
    image_path = "ComidaDB/Pozole/Test/19.jpg"
    img_features = process_single_image(image_path)
    print("HOG Features:", img_features)
    
    print(doPredict(img_features))
    
