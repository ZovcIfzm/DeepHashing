import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pandas as pd

ds = tfds.load('malaria', split = 'train', shuffle_files = True)
assert isinstance(ds, tf.data.Dataset)
arr = tfds.as_numpy(ds)
images = []
labels = []
for item in arr:
    images.append(item['image'])
    labels.append(item['label'])
print(len(images))

images = np.array(images)
labels = np.array(labels)
np.savez('malaria_numpy', images = images, labels = labels)
