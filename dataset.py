
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pandas as pd
import random
import itertools
import constants as k

def load_ds_raw(filename):
    raw_ds = np.load(filename)
    images,labels = raw_ds['images'], raw_ds['labels']
    print(images.shape)
    print(labels.shape)
    return images,labels

def break_classes(X,y):
    class_dict = {}
    for i, sample in enumerate(X):
        label = y[i]
        if label in class_dict:
            class_dict[label].append(sample)
        else:
            class_dict[label] = [sample]
    return class_dict

def splitDataset(class_dict):
  #let's split into a gallery set and query set as was done in the paper
  # gallery_dict = {}
  # query_dict = {}
  galleryX, galleryY, queryX, queryY = [],[],[],[]
  for label, data in class_dict.items():
      np.random.shuffle(data)
      query, gallery = data[:100], data[100:]
      # gallery_dict[label] = gallery
      # query_dict[label] = query
      galleryX+=gallery
      queryX+=query
      for _ in range(100):
          queryY.append(label)
      for _ in range(len(gallery)):
          galleryY.append(label)

  galleryX, galleryY, queryX, queryY = np.array(galleryX), np.array(galleryY), np.array(queryX), np.array(queryY)
  return galleryX, galleryY, queryX, queryY


def generatePairs(class_dict):
  numPos = k.NUMPOS
  numNeg = k.NUMNEG

  numClasses = len(class_dict)
  if numPos < numClasses:
    print("ERROR")
    print("Number of positive pairs less than number of classes")
    raise ValueError

  numPosPerClass = int(numPos/numClasses)
  positivePairs = []
  for each_class in class_dict:
    classImages = class_dict[each_class]
    lenOfClass = len(classImages)

    pairs = list(itertools.combinations(list(range(0,lenOfClass)), 2)) 
    random.shuffle(pairs) 
    for i in range(0, numPosPerClass):
      positivePairs.append((classImages[pairs[i][0]], classImages[pairs[i][1]]))

  positivePairs = np.array(positivePairs, copy=False)

  a = numClasses - 1
  if numNeg < (a+1)*a*0.5:
    print("ERROR")
    print("Number of negative pairs is less than 0.5(a+1)*a")
    print("Where a = numClasses - 1")
    raise ValueError

  pairsPerClassPair = int(numNeg/(0.5*a*(a+1)))
  lenOfClass = len(class_dict[each_class])  
  pairs = list(itertools.combinations(list(range(0,lenOfClass)), 2)) 
  random.shuffle(pairs)
  negativePairs = []

  for each_class in range(numClasses):
    for each_other_class in range(each_class+1, numClasses):      
      for i in range(pairsPerClassPair):
        negativePairs.append((class_dict[each_class][pairs[i][0]], class_dict[each_other_class][pairs[i][1]]))

  negativePairs = np.array(negativePairs, copy=False)
  return positivePairs, negativePairs

def generateInputs(class_dict, training_type):
  inputs = {}

  # Split dataset into gallery and query sets
  inputs["galleryX"], inputs["galleryY"], \
    inputs["queryX"], inputs["queryY"] = splitDataset(class_dict)

  # Generate positive and negative pairs
  # Positive pairs are pairs of images of the same class, negative the opposite.
  if training_type == "supervised":
    inputs["positivePairs"], inputs["negativePairs"] = generatePairs(class_dict)

  return inputs