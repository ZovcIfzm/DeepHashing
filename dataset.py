
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pandas as pd
import random
import itertools
import constants as k

def load_ds_raw(filename):
    raw_ds = np.load(filename, allow_pickle=True)
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
  print("making pairs")
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
    classImages = np.array(class_dict[each_class])
    lenOfClass = len(classImages)
    indices = np.random.choice(lenOfClass,numPosPerClass*2,replace=False).astype(np.int32)
    left_indices, right_indices = indices[:numPosPerClass].astype(np.int32), indices[numPosPerClass:].astype(np.int32)
    left_ims,right_ims = classImages[left_indices,], classImages[right_indices,]
    for left,right in zip(left_ims,right_ims):
      positivePairs.append((left, right))

  positivePairs = np.array(positivePairs, copy=False)
  print("made positive pairs!")

  numNegPerClass = int(numNeg / numClasses)
  negativePairs = []
  for each_class in class_dict.keys():
      other_keys = list(class_dict.keys())
      other_keys.remove(each_class)
      other_classes = np.concatenate([np.array(class_dict[k]) for k in other_keys])
      lenOfClass = len(class_dict[each_class])
      indices = np.random.choice(lenOfClass,numNegPerClass,replace=False).astype(np.int32)
      other_indices = np.random.choice(other_classes.shape[0],numNegPerClass,replace=False)
      left_ims = np.array(class_dict[each_class])[indices]
      right_ims = other_classes[other_indices]
      for left, right in zip(left_ims, right_ims):
          negativePairs.append((left, right))

  return np.array(positivePairs), np.array(negativePairs)

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