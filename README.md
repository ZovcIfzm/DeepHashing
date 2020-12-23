# DeepHashing
Image hashing is the one-way process of converting an image into a binary hash such that similar images have similar hashes. This has promising applications in speeding up approximate nearest neighbor search when trying to retrieve similar images from a database as well as in security for verifying an image hasn’t been perceptually modified. Utilizing deep learning, we implement a model that learns these binary hashes under three primary constraints. First, we minimize the loss in information between the continuous model output and the quantized binary hash. Second, we make sure the binary values are distributed evenly on each bit. Third, we ensure different bits are as independent as possible through a relaxed orthogonality constraint on each fully connected layer of the model. In addition, we implement a variant of the same model that takes advantage of training data labeled for classification tasks in order to generate hashes that are near one another for images of the same class and far away for images of different classes. We evaulate the supervised and unsupervised variants of this model on the MNIST and CIFAR-10 datasets, as was done in the original paper, as well as a recent malaria diagnosis dataset from the NLM.
