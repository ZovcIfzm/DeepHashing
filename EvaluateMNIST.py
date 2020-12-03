import numpy as np
from evaluation_metrics import *
import pandas as pd
import pickle
from NetworkTF import *

USE_EXISTING_MODEL = False

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
galleryX, galleryY, queryX, queryY = [],[],[],[]
for label, data in class_dict.items():
    np.random.shuffle(data)
    query, gallery = data[:100], data[100:]
    gallery_dict[label] = gallery
    query_dict[label] = query
    galleryX+=gallery
    queryX+=query
    for _ in range(100):
        queryY.append(label)
    for _ in range(len(gallery)):
        galleryY.append(label)

galleryX, galleryY, queryX, queryY = np.array(galleryX), np.array(galleryY), np.array(queryX), np.array(queryY)

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

