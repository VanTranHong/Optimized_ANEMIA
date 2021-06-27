import data_preprocess as dp
import numpy as np
import pandas as pd
import scoring as score
import normal_run as nr
import ranking_subset_run as rsr
import sfs_run as sfs_r
import boost_bag_run as bbr
import featureselection as fselect
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import  SGDClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import statsmodels.api as sm
import uni_multiStats as stats

######## READING IN THE DATA #############


data = pd.read_csv('gis_added.csv',skipinitialspace=True, header = 0)
target = 'ChildAnemia'
# selected=['Residence','Maternal Education Level','Maternal occupation (recode)','Husband/paternal occupation (recode)','Water source (recode)','Time to get water source (recode)','Household amenities (recode)','Household transportation (recode)','Floor material (recode)','Religion (recode)','Sex of household head','Listening to radio','Reading News','Watching TV','Currently breastfeeding (from mother)','Maternal Anemia (Binary)','Birth size grouped (recode)','Child IP Drugs (recode)','Birth order (recode)',target]

###     Pre process data ##

# nominal=['Cluster','Region','Residence','Maternal Education Level','Husband/partner education','Maternal occupation (recode)','Husband/paternal occupation (recode)','Water source (recode)','Time to get water source (recode)','Toilet type (recode)','Household amenities (recode)','Household transportation (recode)','Floor material (recode)','Religion (recode)','Number of household members (recode)' ]
# continuous=[]

data1 = dp.modify_data(data,['Age in months'],['GCAT','GCOW','GDOG','Chicken','Disease'],target)
print(data1)





stats.gen_stats(data1, target)
n_features = data1.shape[1]-1

# # #################### RUNNING WITHOUT BOOSTING AND BAGGING for all ranking feature selections and CFS###############
n_seed = 1
splits =2
runs = stats.runSKFold(n_seed,splits,data=data1,target=target)
score.score(rsr.normal_run( n_seed, splits, ['infogain_3'], ['naive_bayes'], runs, n_features),n_seed,splits)
# score.score(sfs_r.subset_run(n_seed, splits,['elasticnet'],['accuracy'],runs,n_features),n_seed,splits)
# sfs_r.subset_features(n_seed,splits, ['elasticnet'],['accuracy'],runs, n_features)
# score.score(bbr.boostbag_run(n_seed,splits,['infogain_20'],['elasticnet'],runs,'boost',n_features), n_seed,splits)
# # subset_features(n_seed, splits, estimators,metrics, runs, n_features):


    
# # score.score(nr.normal_run(data1,n_seed=1,splits=2,methods=['cfs_0'],estimators=['knn']),1,2)
# num = data1.shape[1]
# score.score(parallel.normal_run( 1, 10, ['fcbf_0'], ['naive_bayes'], runs),1,10)

# parallel.normal_run( 1, 10, ['mrmr_0'], ['naive_bayes'], runs)
#, 'infogain_20','reliefF_10','reliefF_20','infogain_'+str(num),'cfs_0'
# # , 'svm','naive_bayes','knn','xgboost'
# # , 'infogain_20', 'reliefF_10','reliefF_20','infogain_'+str(num)
# score.score(parallel.boostbag_run( 2,3, ['infogain_10'], ['elasticnet'], runs,'bag'),2,3)
# parallel.subset_run(5,10,['elasticnet'],['f1'],runs)
# parallel.subset_run(1,2,['knn'],['f1'],runs)

# nr.normal_run(data1,n_seed=1,splits=2,methods=['infogain_10'],estimators=['rdforest'])
# 'rdforest','knn','xgboost','svm',
