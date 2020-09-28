from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import pdb

def svm_classify(train_image_feats, train_labels, test_image_feats):

    
    SVC = LinearSVC(C=700.0, class_weight=None, dual=True, fit_intercept=True,
                    intercept_scaling=1, loss='squared_hinge', max_iter= 2000,
                    multi_class='ovr', penalty='l2', random_state=0, tol= 1e-4,
                    verbose=0)
    SVC.fit(train_image_feats, train_labels)
    
    pred_label = SVC.predict(test_image_feats)

    return pred_label