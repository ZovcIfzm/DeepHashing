import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pandas as pd

ds = tfds.load('mnist', split='train+test', shuffle_files=True)
assert isinstance(ds, tf.data.Dataset)
arr = tfds.as_numpy(ds)
images = []
labels = []
for item in arr:
    images.append(item["image"])
    labels.append(item['label'])

images = np.array(images)
labels = np.array(labels)
np.savez("mnist_numpy",images=images,labels=labels)