import tensorflow as tf
import numpy as np

import evaluation_metrics as em
from NetworkTF import *
import constants as k
import dataset as ds

def runExperiment(modelName, trainingType, inputs):
  # Extract input data
  galleryX = inputs["galleryX"]
  queryX = inputs["queryX"]
  queryY = inputs["queryY"]

  positivePairs = None
  negativePairs = None
  if trainingType == "supervised":
    positivePairs = inputs["positivePairs"]
    negativePairs = inputs["negativePairs"]

  # Apply model
  if k.USE_EXISTING_MODEL:
      model = tf.keras.models.load_model(modelName)
  else:
      #train the model
      model = DeepHash([60,30,16],.1,.1,100,initialize_W(galleryX, 60))
      opt = tf.keras.optimizers.SGD(.0001)

      if trainingType == "unsupervised":
        train_unsupervised(model,300,galleryX,opt,k.CONV_ERROR)
      elif trainingType == "supervised":
        train_supervised(model,100,positivePairs, negativePairs, galleryX, opt, k.ALPHA, k.CONV_ERROR)

      model.save(modelName)

  #run on queries
  query_preds_raw = model(queryX)
  #print(query_preds_raw)
  query_preds = query_preds_raw[1]

  #print(np.array_equal(query_preds.numpy(), np.ones_like(query_preds.numpy())))

  metrics = em.generate_metrics(query_preds,queryY)
  print(metrics)
  print("mAP: ", metrics[0])
  print("Precision at N: ", metrics[1])
  print("Hamming rank: ", metrics[2])