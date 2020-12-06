import numpy as np
from evaluation_metrics import *
import pandas as pd
import pickle
from NetworkTF import *
from dataset import *

USE_EXISTING_MODEL = False

try:
    class_dict = pickle.load(open("mnist_classdict.pkl","rb"))
    print("Classdict dump found, loading previous")
except FileNotFoundError:
    print("First breaking classes since no existing dump was found")
    images,labels = load_ds_raw("mnist_numpy.npz")
    images = images.reshape([70000,784])
    class_dict = break_classes(images,labels)
    pickle.dump(class_dict,open("mnist_classdict.pkl","wb"))


#let's split into a gallery set and query set as was done in the paper
galleryX, galleryY, queryX, queryY = splitDataset(class_dict)

if USE_EXISTING_MODEL:
    model = tf.keras.models.load_model("mnist_model")

else:
    #train the model
    model = DeepHash([60,30,16],0,0,100,initialize_W(galleryX, 60))
    opt = tf.keras.optimizers.SGD(.001)
    train_unsupervised(model,100,galleryX,opt,.01)
    model.save("mnist_model")

#run on queries
query_preds_raw = model(queryX)
print(query_preds_raw)
query_preds = query_preds_raw[1]

print(np.array_equal(query_preds.numpy(), np.ones_like(query_preds.numpy())))

metrics = generate_metrics(query_preds,queryY)
print(metrics)

