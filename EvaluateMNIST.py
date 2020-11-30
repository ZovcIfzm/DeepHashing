import numpy as np
from evaluation_metrics import *
import pandas as pd
import pickle

def load_ds_raw():
    raw_ds = np.load('mnist_numpy.npz')
    images,labels = raw_ds['images'], raw_ds['labels']
    print(images.shape)
    print(labels.shape)
    return images,labels

try:
    class_dict = pickle.load(open("mnist_classdict.pkl","rb"))
    print("Classdict dump found, loading previous")
except FileNotFoundError:
    print("First breaking classes since no existing dump was found")
    images,labels = load_ds_raw()
    images = images.reshape([70000,784])
    class_dict = break_classes(images,labels)
    pickle.dump(class_dict,open("mnist_classdict.pkl","wb"))


#let's split into a gallery set and query set as was done in the paper
gallery_dict = {}
query_dict = {}
for label, data in class_dict.items():
    np.random.shuffle(data)
    query, gallery = data[:100], data[100:]
    gallery_dict[label] = gallery
    query_dict[label] = query
    
