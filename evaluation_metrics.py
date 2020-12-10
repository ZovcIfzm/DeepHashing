import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import pairwise_distances as pdist

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

def precision_at_sample(pairwise_dists,labels,N):
    total_to_average = 0
    for i, label in enumerate(labels.tolist()):
        sorted_inds_dists = np.argsort(pairwise_dists[i])
        numerator = 0
        for closest in sorted_inds_dists[1:1+N]:
            if labels[closest]==label:
                numerator+=1
        total_to_average+=numerator/(N-1)
    return total_to_average/labels.shape[0]

def hamming_radius(pairwise_dists, labels,r):
    to_average = 0
    numma = 0
    denna = 0
    for i, label in enumerate(labels.tolist()):
        numerator = 0
        denominator = 0
        for j, dist in enumerate(pairwise_dists[i]):
            if i==j:
                continue
            if dist<r:
                denominator+=1
                denna+=1
                if labels[j]==label:
                    numerator+=1
                    numma+=1
        if denominator!=0:
            to_average+=numerator/denominator
        else:
            to_average+=1
    return numma/denna

def generate_metrics(hashes,labels,hamming_N = 500, hamming_R=2):
    dists = pdist(hashes,metric="hamming") * hashes.shape[1]
    mAP = mean_average_precision(dists,labels)
    precision_at_N = precision_at_sample(dists,labels,hamming_N)
    hamming_rank = hamming_radius(dists,labels,hamming_R)
    return mAP, precision_at_N, hamming_rank
