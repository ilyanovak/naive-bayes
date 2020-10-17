# Test

import numpy as np
# X = np.array([[1, 1],
#               [2, 3],
#               [4, 7],
#               [1, 1],
#               [2, 1],
#               [3, 2]])
# Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict([[4, 5]]))
print(clf_pf.predict_proba([[4, 5]]))
