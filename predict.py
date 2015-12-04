import data_analyze
import pickle
from sklearn import svm, preprocessing, neighbors

data = data_analyze.ComputeStats()
data = data.stats_ar

#Extracting features from array
features = data[:, 1:6]
features = features.astype('S')

#Coding the boolean variables Retweeted and Verified
features[features[:, 0] == 'True', 0] = 1
features[features[:, 0] == 'False', 0] = 0
features[features[:, 1] == 'True', 1] = 1
features[features[:, 1] == 'False', 1] = 0

#Converting the entire array from unicode to floats
features = features.astype(float)

#Scaling the features possibly to help with knn classifier
features[:,2] = preprocessing.scale(features[:,2])
features[:,3] = preprocessing.scale(features[:,3])
features[:,4] = preprocessing.scale(features[:,4])


#Coding target
target_y = data[:, 6]
target_y[target_y == u"HootSuite"] = 1
target_y[target_y != u'1'] = 0
target_y = target_y.astype(int)

clf2 = pickle.load(open("fitted_model.p", "rb"))

a = clf2.predict(features)
