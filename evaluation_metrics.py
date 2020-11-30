import numpy as np

def break_classes(X,y):
    class_dict = {}
    for i, sample in enumerate(X):
        label = y[i]
        if label in class_dict:
            class_dict[label].append(sample)
        else:
            class_dict[label] = [sample]
    return class_dict

def mean_average_precision():
    pass

def hamming_radius():
    pass

def precision_at_sample():
    pass
