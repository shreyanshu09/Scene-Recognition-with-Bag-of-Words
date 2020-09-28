from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):

    CATEGORIES = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
              'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
              'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']
    K = 1
    
    N = train_image_feats.shape[0]
    M = test_image_feats.shape[0]
    d = train_image_feats.shape[1] # d are same in both train and test
    
    dist = distance.cdist(test_image_feats, train_image_feats, metric='euclidean')

    test_predicts = []
    
    for each in dist:
        label = []
        idx = np.argsort(each)
        for i in range(K):
            label.append(train_labels[idx[i]])
        

        amount = 0
        for item in CATEGORIES:
            if label.count(item) > amount:
                label_final = item
    
        test_predicts.append(label_final)
        

    return test_predicts