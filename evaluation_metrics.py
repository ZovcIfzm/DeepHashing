import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import pairwise_distances as pdist

def break_classes(X,y):
    class_dict = {}
    for i, sample in enumerate(X):
        label = y[i]
        if label in class_dict:
            class_dict[label].append(sample)
        else:
            class_dict[label] = [sample]
    return class_dict

#so scores is just a way to rank the distances, the actual values don't matter, just the relative order
#larger scores are higher ranked here, which is why we take the reciprocal
#the label should just be 1 if it's the same class, 0 otherwise
def mean_average_precision(pairwise_dists,labels):
    mAP = 0
    for i, label in enumerate(labels.tolist()):
        scores = 1/(np.concatenate([pairwise_dists[i][:i],pairwise_dists[i][i+1:]])+1) #plus one to avoid div zeros
        curr_labels = np.where(labels==label,1,0)
        curr_labels = np.concatenate([curr_labels[:i],curr_labels[i+1:]])
        AP = average_precision_score(curr_labels,scores)
        mAP+=AP
    mAP = mAP/labels.shape[0]
    return mAP



def hamming_radius(pairwise_dists,labels,N):
    pass

def precision_at_sample():
    pass

def generate_metrics(hashes,labels,hamming_N = 100):
    dists = pdist(hashes,metric="hamming")
    mAP = mean_average_precision(dists,labels)
    return mAP
