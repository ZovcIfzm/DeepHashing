# Used for turning images into GIST descriptors
# Our model uses 512-D GIST feature vectors to represent the images
'''
  Don't use this package, it's too old (last updated 2010).
import leargist # you need to manually install pyleargist by going to the package
                # and running 
                # "python setup.py build"
                # "sudo python setup.py install"
'''
import gist # You have to install FFTW and lear-gist first before you can import
            # Documentation here: https://github.com/tuttieee/lear-gist-python
import numpy as np
import constants as k
from evaluation_metrics import *
from dataset import *

from NetworkTF import *
import pickle

USE_EXISTING_MODEL = False

# Straight from CIFAR-10 python3 documentation
# Cifar-10 contains 5 data_batch_# files, a test_batch file,
#   and a batches.meta file that gives meaningful names to the data labels
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# Returns 
#{
# data: a 10000x3072 numpy array of uint8s. Each row of the array 
#   stores a 32x32 colour image. The first 1024 entries contain the 
#   red channel values, the next 1024 the green, and the final 1024 
#   the blue. The image is stored in row-major order, so that the first 
#   32 entries of the array are the red channel values of the first 
#   row of the image.
# labels: a list of 10000 numbers in the range 0-9. The number at 
#   index i indicates the label of the ith image in the array data.
#}

# Takes in raw data from unpickle (10000x3072 vector)
def convertToGIST(images):
  gist_vectors = [gist.extract(img) for img in images]
  return np.asarray(gist_vectors)

def evaluateCifar():
  try:
      class_dict = pickle.load(open("cifar_classdict.pkl","rb"))
      print("Classdict dump found, loading previous")
  except FileNotFoundError:
      print("First breaking classes since no existing dump was found")
      images,labels = load_ds_raw('cifar_numpy.npz')
      gist_data = np.asarray(convertToGIST(images))
      print("gd shape", gist_data.shape)
      print("label shape", labels.shape)
      class_dict = break_classes(gist_data, labels)
      pickle.dump(class_dict,open("cifar_classdict.pkl","wb"))

  galleryX, galleryY, queryX, queryY = splitDataset(class_dict)

    
  if USE_EXISTING_MODEL:
      model = tf.keras.models.load_model("cifar_model")

  else:
      #train the model
      model = DeepHash([60,30,16],0,0,100,initialize_W(galleryX, 60))
      opt = tf.keras.optimizers.SGD(.001)
      train_unsupervised(model,100,galleryX,opt,.01)
      model.save("cifar_model")

  #run on queries
  query_preds_raw = model(queryX)
  print(query_preds_raw)
  query_preds = query_preds_raw[1]

  print(np.array_equal(query_preds.numpy(), np.ones_like(query_preds.numpy())))

  metrics = generate_metrics(query_preds,queryY)
  print(metrics)



if __name__ == '__main__':
  evaluateCifar()