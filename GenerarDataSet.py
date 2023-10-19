import cv2
import numpy as np
import pandas as pd

from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean

from sklearn.model_selection import train_test_split
from skimage import data, color, feature
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity



import glob

def load_data(pez, tipo):
    label=[]
    arr = []
    strr = "ComidaDB/"+pez+"/" + tipo + "/*"
    for file_ in glob.glob(strr):
      img = cv2.imread(file_)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      arr.append(img)
      label.append(pez)

    ##print (data_train_ftr)		
    ##print (len(label))
    ##print (label)

   
    return arr,label

def whole_train_data(tipo):
    Torta_data, Torta_label = load_data('Torta', tipo)
    Gordita_data, Gordita_label = load_data('Gordita', tipo)
    Flauta_data, Flauta_label = load_data('Flauta', tipo)
    Tamal_data, Tamal_label = load_data('Tamal', tipo)
    Pozole_data, Pozole_label = load_data('Pozole', tipo)

    data =np.concatenate((Torta_data,Gordita_data,Flauta_data,Tamal_data,Pozole_data))
    labels =np.concatenate((Torta_label, Gordita_label, Flauta_label, Tamal_label, Pozole_label))

    return data, labels

def preprocessing(arr):
    arr_prep=[]
    for i in range(arr.shape[0]):
        img=cv2.cvtColor(arr[i], cv2.COLOR_BGR2GRAY)
        img=resize(img, (36, 36),anti_aliasing=True)
        arr_prep.append(img)
    return arr_prep


def ExtractHOG(img):
#    ftr,_=hog(img, orientations=8, pixels_per_cell=(16, 16),
#            cells_per_block=(1, 1), visualize=True, multichannel=False)
    ftr,_=hog(img, orientations=8, pixels_per_cell=(4, 4),
            cells_per_block=(2, 2), visualize=True)
    return ftr
  
def preprocessing_part_two_orig(arr):
    arr_feature=[]
    for i in range(np.shape(arr)[0]):
        arr_feature.append(ExtractHOG(arr[i])) 
    return arr_feature

def preprocessing_part_two(arr):
    # En vez de que cada imagen sea un array, que sea una lista
    arr_feature=[]
    for i in range(np.shape(arr)[0]):
        arr_feature.append(list(ExtractHOG(arr[i]))) 
    return arr_feature


def GenerarDataset():
    #print ("Procesando dataser de Entrenamiento")
    data_train, labels_train = whole_train_data('Train')
    #print ("Procesando dataser de Prueba")
    data_test, labels_test = whole_train_data('Test')



    #print ("Realizando Preprocesamiento1 al Dataset de Entrenamiento")
    data_train_p = preprocessing(data_train)
    #print ("Realizando Preprocesamiento1 al Dataset de Prueba")
    data_test_p = preprocessing(data_test)

    #print (data_train_p)
    #print (data_test_p)				
    #exit()

    #print ("Realizando Preprocesamiento2 al Dataset de Entrenamiento")
    data_train_ftr = preprocessing_part_two(data_train_p)
    #print ("Realizando Preprocesamiento2 al Dataset de Prueba")
    data_test_ftr= preprocessing_part_two(data_test_p)

    print (len(data_train_ftr))
    print (len(data_test_ftr))	
    print (labels_train)
    print (len(labels_train))
    print (labels_test)		
    print (len(labels_test))

    entire_dataset = data_train_ftr+data_test_ftr
    entire_labels = list(labels_train)+list(labels_test)
    print (entire_labels)
    df = pd.DataFrame(entire_dataset)
    df['class']	=entire_labels	
    df.to_csv("comida.csv",index=False)

    if (len(data_train_ftr) <= 0):
        return False
    else:
        return True
