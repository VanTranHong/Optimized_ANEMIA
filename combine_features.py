import data_preprocess as dp
import numpy as np
import pandas as pd
import featureselection as fselect
from sklearn.model_selection import StratifiedKFold
import sffs
import os
from joblib import Parallel, delayed
import normal_run as nr
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import  ElasticNet
from sklearn.linear_model import  SGDClassifier


# files = ['/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_200.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_201.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_202.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_203.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_204.csv','/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_400.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_401.csv','/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_402.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_403.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_404.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_600.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_601.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_602.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_603.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_604.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_800.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_801.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_802.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_803.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_804.csv','/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_960.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_961.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_962.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_963.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/infogain_964.csv']

# files = ['/Users/vantran/Desktop/anemia/STATS/NONMOTHER/mrmr_00.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/mrmr_01.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/mrmr_02.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/mrmr_03.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/mrmr_04.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/cfs_00.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/cfs_01.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/cfs_02.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/cfs_03.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/cfs_04.csv']
files = ['/Users/vantran/Desktop/anemia/STATS/NONMOTHER/elasticnetaccuracy0.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/elasticnetaccuracy1.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/elasticnetaccuracy2.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/elasticnetaccuracy3.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/elasticnetaccuracy4.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/elasticnetf10.csv','/Users/vantran/Desktop/anemia/STATS/NONMOTHER/elasticnetf11.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/elasticnetf12.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/elasticnetf13.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/elasticnetf14.csv','/Users/vantran/Desktop/anemia/STATS/NONMOTHER/naive_bayesaccuracy0.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/naive_bayesaccuracy1.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/naive_bayesaccuracy2.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/naive_bayesaccuracy3.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/naive_bayesaccuracy4.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/naive_bayesf10.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/naive_bayesf11.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/naive_bayesf12.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/naive_bayesf13.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/naive_bayesf14.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/knnf10.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/knnf11.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/knnf12.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/knnf13.csv', '/Users/vantran/Desktop/anemia/STATS/NONMOTHER/knnf14.csv']
df = pd.DataFrame()
for f in files:
    temp = pd.read_csv(f, index_col=0)
    col = temp['0']
    df = pd.concat([df,col],axis=1)
df.to_csv("/Users/vantran/Desktop/anemia/STATS/NONMOTHER/sffs.csv")
