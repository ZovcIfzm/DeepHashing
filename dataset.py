
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pandas as pd

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
  positivePairs = []
  for each_class in class_dict:
    classImages = class_dict[each_class]
    lenOfClass = len(classImages)
    for i in range(lenOfClass):
      for j in range(i+1, lenOfClass):
        positivePairs.append((classImages[i], classImages[j]))
  positivePairs = np.array(positivePairs, copy=False)

  negativePairs = []
  numClasses = len(class_dict)
  for cur_class_num in range(numClasses):
    classImages = class_dict[cur_class_num]
    for img in classImages:
      for other_class_num in range(cur_class_num+1, numClasses):
        for other_img in class_dict[other_class_num]:
          negativePairs.append((img, other_img))
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