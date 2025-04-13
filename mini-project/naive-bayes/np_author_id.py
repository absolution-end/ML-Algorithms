

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
import os
from time import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
# from tools import email_preprocess as ep
from email_preprocess import preprocess 
# from email_preprocess import strong

### features_train and features_test are the features for the training
from sklearn.naive_bayes import GaussianNB
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


##############################################################
# Enter Your Code Here
t0 = time()
clf = GaussianNB()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print("Accuracy:", accuracy)
print("predicting Time:", round(time()-t0, 3), "s")

##############################################################

##############################################################


# Great job! Udacity is proud of you!