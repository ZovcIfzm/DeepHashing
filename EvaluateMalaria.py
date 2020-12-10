import numpy as np
import pandas as pd
import cv2
import pickle
from skimage.color import rgb2gray
import GIST as cust_gist
import dataset as ds
from NetworkTF import *
import experiments as exp
from evaluation_metrics import *

def convertToGIST(images):
    param = {
            "orientationsPerScale": np.array([8,8,8,8]),
            "numberBlocks": [4,4],
            "fc_prefilt": 4,
            "boundaryExtension": 10
    }
    gist = cust_gist.GIST(param)
    resizedImages = [cv2.resize(img, dsize = (min(img.shape[0], img.shape[1]), min(img.shape[0], img.shape[1])), interpolation=cv2.INTER_CUBIC) for img in images]
    gist_vectors = [gist._gist_extract(rgb2gray(img)) for img in resizedImages]
    gist_vectors = np.array(gist_vectors, copy=False)
    return gist_vectors

try:
    classDict = pickle.load(open('malaria_classdict.pkl', 'rb'))
    print('Classdict dump found, loading previous')
except FileNotFoundError:
    print('First breaking classes since no existing dump was found')
    images, labels = ds.load_ds_raw('malaria_numpy.npz')
    gistData = np.array(convertToGIST(images), copy = False)
    classDict = ds.break_classes(gistData, labels)
    pickle.dump(classDict, open('malaria_classdict.pkl','wb'))
    
print('Generating Inputs')
inputs = ds.generateInputs(classDict, 'unsupervised')

print('Running Unsupervised Experiment')
exp.runExperiment('malaria_model', 'unsupervised', inputs)
    



