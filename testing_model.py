import cv2
import numpy as np
from skimage.transform import resize
from skimage.feature import hog
import os

import joblib

def preprocess_image(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_img = resize(gray_img, (36, 36), anti_aliasing=True)
    hog_features = ExtractHOG(resized_img)
    return hog_features

def ExtractHOG(img):
    ftr, _ = hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True)
    return ftr

def predict_single_image(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict([preprocessed_image])
    return prediction

def doSinglePredict(image_path, model):
    # Load and preprocess a single image
    image = cv2.imread(image_path)
    
    # Make a prediction for the single image using the provided model
    prediction = predict_single_image(image, model)

    return prediction


def doPredict(images_path):
    # Load the three classifiers
    svm_model = joblib.load('modelo_entrenado_LinearSVM.joblib')  #  SVM model separately
    mlp_model = joblib.load('modelo_entrenado_NeuralNet.joblib')  #  MLP model separately
    logreg_model = joblib.load('modelo_entrenado_LogisticRegression.joblib')  #  Logistic Regression model separately

    # Get the predictions using each classifier
    image_predictions = []
    file_names = os.listdir(images_path)
    
    for image_path in file_names:
        print("IMAGEN:" + images_path +"/"+ image_path)
        svm_prediction = doSinglePredict(images_path +"/"+ image_path, svm_model)
        mlp_prediction = doSinglePredict(images_path +"/"+ image_path, mlp_model)
        logreg_prediction = doSinglePredict(images_path +"/"+ image_path, logreg_model)
        image_predictions.append([image_path, svm_prediction[0], mlp_prediction[0], logreg_prediction[0]])

    results = []

    for image in image_predictions:
        predictions = [image[1], image[2], image[3]]
        class_vote = [['Torta',0], ['Flauta',0], ['Gordita',0], ['Tamal',0], ['Pozole',0]]
        for i in predictions:
            class_vote[i-1][1] += 1

        bestAnswer = class_vote[0]
        for j in class_vote:
            if bestAnswer[1] < j[1]:
                bestAnswer = j

        results.append([image[0], bestAnswer])
    print(results)
    return results



#def doPredict(image_path):
#    # Predictions list
#    predictions = []
#    # Load your model
#    model = joblib.load('modelo_entrenado.joblib')
#
#    # Load and preprocess a single image
#    image = cv2.imread(image_path)
#    
#    # Make a prediction for the single image
#    prediction = predict_single_image(image, model)
#
#    return prediction