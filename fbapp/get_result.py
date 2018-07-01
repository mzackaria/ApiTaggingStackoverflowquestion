import pickle
file = open('logistic_regression.pyc', 'rb')
lr = pickle.load(file)
file.close()

file = open('multilabel_binarizer.pyc', 'rb')
multilabel_binarizer = pickle.load(file)
file.close()

file = open('vectorizer.pyc', 'rb')
vectorizer = pickle.load(file)
file.close()
