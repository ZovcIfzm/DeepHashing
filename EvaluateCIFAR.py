# Libraries
import numpy as np
import pickle
from skimage.color import rgb2gray

# Other python files
import constants as k
from evaluation_metrics import *
import experiments as exp
import dataset as ds
from NetworkTF import *
import GIST as cust_gist  # You have to install FFTW and lear-gist first before you can import
                          # Documentation here: https://github.com/tuttieee/lear-gist-python

# Converts numpy images (32x32x3) into 512-D GIST vector
# Converts images into grayscale first
def convertToGIST(images):
  param = {
        "orientationsPerScale": np.array([8,8,8,8]),
         "numberBlocks": [4,4],
        "fc_prefilt": 4,
        "boundaryExtension": 10
  }
  gist = cust_gist.GIST(param)
  gist_vectors = [gist._gist_extract(rgb2gray(img)) for img in images]
  gist_vectors = np.array(gist_vectors, copy=False)
  return gist_vectors

def evaluateCIFAR():
  # Preprocess CIFAR data into standardized form (classDict)
  try:
      classDict = pickle.load(open("cifar_classdict.pkl","rb"))
      print("Classdict dump found, loading previous")
  except FileNotFoundError:
      print("First breaking classes since no existing dump was found")
      images, labels = ds.load_ds_raw('cifar_numpy.npz')
      gistData = np.array(convertToGIST(images), copy=False)
      classDict = ds.break_classes(gistData, labels)
      pickle.dump(classDict, open("cifar_classdict.pkl","wb"))

  # Generate inputs (supervised is a superset of unsupervised + +/- pairs)
  print("Generating Inputs")
  inputs = ds.generateInputs(classDict, "unsupervised")

  # Run experiments
  print("Running unsupervised experiment")
  exp.runExperiment("cifar_model", "unsupervised", inputs)
  #exp.runExperiment("cifar_model", "supervised", inputs)

if __name__ == '__main__':
  evaluateCIFAR()