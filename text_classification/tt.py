import numpy as np

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

enc.fit([[0], [1], [2]])

print ("enc.n_values_is:",enc.n_values_)

print ("enc.feature_indices_is:",enc.feature_indices_)

print (enc.transform([[0]]).toarray())
