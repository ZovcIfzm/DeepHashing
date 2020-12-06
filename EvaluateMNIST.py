# Libraries
import numpy as np
import pandas as pd
import pickle

# Other python files
import experiments as exp
import dataset as ds
import constants as k

def evaluateMNIST():
    # Preprocess MNIST data into standardized form (classDict)
    try:
        classDict = pickle.load(open("mnist_classdict.pkl","rb"))
        print("Classdict dump found, loading previous")
    except FileNotFoundError:
        print("First breaking classes since no existing dump was found")
        images,labels = ds.load_ds_raw("mnist_numpy.npz")
        images = images.reshape([70000,784])
        classDict = ds.break_classes(images,labels)
        pickle.dump(classDict,open("mnist_classdict.pkl","wb"))

    # Generate inputs (supervised is a superset of unsupervised + +/- pairs)
    print("Generating Inputs")
    inputs = ds.generateInputs(classDict, "unsupervised")

    # Run experiments
    print("Running unsupervised experiment")
    exp.runExperiment("mnist_model", "unsupervised", inputs)
    #exp.runExperiment("mnist_model", "supervised", inputs)


if __name__ == '__main__':
  evaluateMNIST()