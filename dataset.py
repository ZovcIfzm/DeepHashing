
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
  