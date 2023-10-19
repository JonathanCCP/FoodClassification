"""
Version Original del Codigo:
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
TODO: Checar la lista completa de algoritmos en este enlace


https://scikit-learn.org/stable/supervised_learning.html
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib


from sklearn import datasets
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier





def clasifierComparison():

    names = [        
        "LinearSVM",
        "NeuralNet",
        "LogisticRegression",
    ]

    classifiers = [
        SVC(kernel="linear", C=0.025),
        MLPClassifier(alpha=1, max_iter=1000),
        LogisticRegression(),
    ]



    #df = pd.DataFrame()
    df = pd.read_csv('comida.csv')
    mymap = {'Torta':1, 'Flauta':2, 'Gordita':3, 'Tamal':4, 'Pozole':5}
    df = df.applymap(lambda s: mymap.get(s) if s in mymap else s)

    #print (df)
    X = df.drop(columns=['class'])
    y = df['class']

    # import some data to play with
    #iris = datasets.load_iris()
    #X = iris.data  # we only take the first two features.
    #y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    name_score = []

    # iterate over classifiers
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        # Guardar el clasificador entrenado en un archivo
        joblib.dump(clf, 'modelo_entrenado_' + name + '.joblib')
        Y_pred = clf.predict(X_test)
        score = clf.score(X_test, y_test)
        name_score.append([name, score])
    
    print(name_score)
        



    return name_score
