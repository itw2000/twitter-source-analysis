
from sklearn import preprocessing, linear_model, tree, neighbors
from sklearn.cross_validation import KFold, StratifiedKFold
import data_analyze
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import pylab as pl
from pylab import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib
from scipy import stats
import perceptron
import cProfile


reload(data_analyze)
data1 = data_analyze.ComputeStats()
data = data1.stats_ar
visual_data = data1.stats_df

#Extracting features from array
features = data[:, 1:7]
features = np.char.encode(features, 'ascii', 'ignore')
target_y = np.char.encode(data[:, 7], 'ascii', 'ignore')

def sty(axis):

    axis.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
    axis.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.7)
    axis.patch.set_facecolor('0.85')
    axis.set_axisbelow(True)

    for child in axis.get_children():
         if isinstance(child, matplotlib.spines.Spine):
            child.set_alpha(0)
    for line in axis.get_xticklines() + axis.get_yticklines():
        line.set_markersize(5)
        line.set_color("gray")
        line.set_markeredgewidth(1.4)
    for line in axis.xaxis.get_ticklines(minor=True) + axis.yaxis.get_ticklines(minor=True):
        line.set_markersize(0)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    axis.xaxis.set_ticks_position('bottom')
    axis.yaxis.set_ticks_position('left')


#Taking a quick look at the data
def explore (stats_df = visual_data):
    a = stats_df['followers'].describe()
    b = stats_df['friends'].describe()
    c = stats_df['stat_count'].describe()
    d = stats_df['listed'].describe()

    print a,b,c,d
    print len(stats_df)

    fig, axes = pl.subplots(nrows=2, ncols=2)



    stats_df['followers'].hist(bins=50,ax=axes[0,0], alpha=0.5, color='0.2'); axes[0,0].set_title('Followers', family='light'); axes[0,0].set_ylabel('Frequency')
    sty(axes[0,0])
    stats_df['friends'].hist(bins=50,ax=axes[0,1], alpha=0.5,color='0.2'); axes[0,1].set_title('Friends'); axes[0,1].set_ylabel('Frequency');axes[0,1].patch.set_facecolor('0.85')
    sty(axes[0,1])
    stats_df['stat_count'].hist(bins=50,ax=axes[1,0], alpha=0.5,color='0.2'); axes[1,0].set_title('Number of Tweets'); axes[1,0].set_ylabel('Frequency');axes[1,0].patch.set_facecolor('0.85')
    sty(axes[1,0])
    stats_df['listed'].hist(bins=50,ax=axes[1,1], alpha=0.5,color='0.2'); axes[1,1].set_title('Number of Lists User is On'); axes[1,1].set_ylabel('Frequency');axes[1,1].patch.set_facecolor('0.85')
    sty(axes[1,1])

    pl.show()






#Simple models built on the data sampled from tweets collection.
#It applies some common algorithims from the scikit-learn package
#to that data to predict the tweet came from a platform.
# Metric used to evaluate is AUC due to large class imbalances.
def classify (features=features, target_y=target_y):
    print features.shape
    #Coding the boolean variables Retweeted and Verified
    #Redundent remove
    features[features[:, 0] == 'True', 0] = 1
    features[features[:, 0] == 'False', 0] = 0
    features[features[:, 1] == 'True', 1] = 1
    features[features[:, 1] == 'False', 1] = 0

    #Converting the entire array from unicode to floats
    features = features.astype(float)
    minmax = preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0))
    features[:,2:5] = minmax.fit_transform(features[:,2:5])
    #Scaling the features possibly to help with knn classifier
    #features[:,2] = minmax.fit_transform(features[:,2])
    #features[:,3] = minmax.fit_transform(features[:,3])
    #features[:,4] = minmax.fit_transform(features[:,4])
    #features[:,5] = minmax.fit_transform(features[:,5])


    n_samples, n_features = features.shape
    #Coding target
    positive = np.where(target_y == "Twitter for iPhone")
    negative = np.where(target_y != "Twitter for iPhone")
    target_y[positive] = 1
    target_y[negative] = 0
    target_y = target_y.astype(float)
    (sum(target_y))

    #Shuffling the features and target and creating a training and test set.
    #Need to perform k-fold validation
    features, target_y = shuffle(features, target_y)
    kf = StratifiedKFold(target_y, n_folds=5, indices=False)

    half = (n_samples/2)

    X_train, y_train = features[half:], target_y[half:]
    X_test, y_test = features[:half], target_y[:half]



    clsf = RandomForestClassifier(n_estimators=500)
    probs = clsf.fit(X_train, y_train).predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Area : %f" % roc_auc)
    pl.plot(fpr, tpr, color='grey', label='Random Forest (auc = %0.2f' % roc_auc)

    clsf2 = tree.DecisionTreeClassifier(criterion='entropy')

    for i, (train, test) in enumerate(kf):

        probs = clsf2.fit(features[train], target_y[train]).predict_proba(features[test])

        #Calculation the tpr, fpr and for the thresholds
        fpr, tpr, thresholds = roc_curve(target_y[test], probs[:, 1])
        roc_auc = auc(fpr, tpr)
        print("Area : %f" % roc_auc)

        pl.plot(fpr, tpr, lw=1, label='Fold %d (auc = %0.2f)' % (i, roc_auc))
        pl.plot([0,1],[0,1],'k--')

    pl.xlabel('FPR')
    pl.ylabel('TPR')
    pl.legend(loc="lower right")
    pl.show()





if __name__ == "__main__":
    classify()























