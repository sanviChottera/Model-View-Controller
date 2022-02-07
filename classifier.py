import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

print(pd.Series(y).value_counts())

classes = ['0', '1', '2','3', '4','5', '6', '7', '8','9']
nclasses = len(classes)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
xtrainscaled = xtrain/255
xtestscaled = xtest/255

clf = LogisticRegression(solver= "saga", multi_class= 'multinomial').fit(xtrainscaled, ytrain)

def get_prediction(image):
    impil = Image.open(image)
    imagebw = impil.convert('L')
    imagebwresize  = imagebw.resize( (28,28), Image.ANTIALIAS)
    pixelfilter = 20
    #percentile() converts the values in scalar quantity
    minpixel = np.percentile(imagebwresize, pixelfilter)
    
    #using clip to limit the values betwn 0-255
    imgInverted_scaled = np.clip(imagebwresize - minpixel, 0, 255)
    maxpixel = np.max(imagebwresize)
    imgInverted_scaled = np.asarray(imgInverted_scaled)/maxpixel
    #converting into an array() to be used in model for prediction
    testsample = np.array(imgInverted_scaled).reshape(1,784)
    testpred = clf.predict(testsample)
    return testpred[0]
    