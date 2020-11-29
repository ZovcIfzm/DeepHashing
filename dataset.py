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
from constants import CIFAR_BATCH_LOCATIONS

# Straight from CIFAR-10 python3 documentation
# Cifar-10 contains 5 data_batch_# files, a test_batch file,
#   and a batches.meta file that gives meaningful names to the data labels
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpickleData(file):
  return unpickle(file).data
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

def loadCifar():
  batches = unpickleData(CIFAR_BATCH_LOCATIONS)
  print(batches.shape)

# Takes in raw data from unpickle (6*10000x3072 vector)
def convertToGIST(data_vec):
  num_imgs = data_vec.shape[0]
  # convert (60000*3072) to (60000x3x32x32) to (60000,32x32x3)
  imgs_reshaped = np.transpose(np.reshape(data_vec,(num_imgs,3,32,32)), (0,2,3,1))
  # gist_vectors = leargist.color_gist(imgs_reshaped)
  gist_vectors = gist.extract(imgs_reshaped)
  return gist_vectors

if __name__ == '__main__':
  loadCifar()