import data_preprocess as dp
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import statsmodels.api as sm
from joblib import Parallel, delayed



#######    main    ####
def runSKFold(n_seed, splits, data,target):
    
    '''
    splitting the data into n_seed of splits folds cross validarion
    Args:
    n_seed(int): number of cross-validation
    splits(int): number of folds in each cross validation
    
    Returns:
    List: data for each of n_seed * splits folds
    '''
    runs = []

    X = np.array(data.drop(target,axis=1))
    y = np.array(data[target])
    result = Parallel(n_jobs=-1)(delayed(execute_skfold)(X,y, splits, seed) for seed in range(n_seed))
    for i in  result:   
        for j in i:
            runs.append(j)
    return runs

def execute_skfold(X,y, splits, seed):
    skf = StratifiedKFold(n_splits=splits, random_state=seed, shuffle=True)
    result = Parallel(n_jobs=-1)(delayed(execute_split)(X,y, train, test) for train, test in skf.split(X,y))
    return result

def execute_split(X,y, train, test):
    X_train, X_test = X[train], X[test]
    X_train, X_test = dp.impute(X_train), dp.impute(X_test)
    y_train, y_test = y[train], y[test]
    arr = [X_train, X_test, y_train, y_test]
    return arr
    

def stats(data, column, category):   
    '''
    reporting p-values and odds ratio for multivariate logistic regression
    '''
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis = 1)
    endog = data[column]
    features = data.drop(columns=[column],axis=1).columns
    exog = sm.add_constant(data.loc[:,features])
    logit = sm.Logit(endog,exog)
    result = logit.fit()
    p_values = result.pvalues.to_frame()
    p_values.columns =["P_values"]
    CI = result.conf_int(alpha = 0.05)
    CI.columns = ["LowerCI","UpperCI"] 
    coeff = np.exp(result.params.to_frame())
    coeff.columns=["Coefficients"]
    final = pd.concat([p_values,CI,coeff], axis=1)
    final.to_csv("/Users/vantran/Desktop/ANEMIA/processed_data/"+category+"/multivariate_stats.csv")
  
   
    


def univariate_stats(data,column, groups, category):
    '''
    reporting p-values and odds ration for univariate logistic regression
    '''
    results = pd.DataFrame(columns=["P_values","LowerCI","UpperCI","Coefficients"])
    for group in groups:
    
        endog = data[column]
        exog = sm.add_constant(data.loc[:,group])
        logit = sm.Logit(endog,exog)
        result = logit.fit()
        p_values = result.pvalues.to_frame()
        p_values.columns =["P_values"]
        CI = result.conf_int(alpha = 0.05)
        CI.columns = ["LowerCI","UpperCI"] 
        coeff = np.exp(result.params.to_frame())
        coeff.columns=["Coefficients"]
        final = pd.concat([p_values,CI,coeff], axis=1)
        results = results.append([final])
    results.to_csv("/Users/vantran/Desktop/ANEMIA/processed_data/"+category+"/univariate_stats.csv")


     
def impute(data):
    '''
    imputing the data
    '''
    
    columns = data.columns
    imputed_data = []
    for i in range(2):
        imputer = IterativeImputer(sample_posterior=True, random_state=i, verbose=1)
        imputed_data.append(imputer.fit_transform(data))
    returned_data = np.round(np.mean(imputed_data,axis = 0))
    return_data = pd.DataFrame(data = returned_data, columns=columns)

    return return_data



def gen_stats(data, target, nominals, category):
    '''
    generating baseline accuracy and f1 score and multivariate and univariate
    '''
  
    data1 = impute(data)
    data1.to_csv('/Users/vantran/Desktop/ANEMIA/processed_data/'+category+'/imputed_data.csv')   
    ###     multivariate    ##
    stats(data1,target,category)
    ###     univariate      ###
    univariate_stats(data1,target,nominals, category)



# # # ######## baseline accuracy ##########
    f = open('/Users/vantran/Desktop/ANEMIA/processed_data/'+category+'/baselineresult.txt','w')
    rate = sum(data1[target])/data1.shape[0]
    rate2 = 1-rate
    f.write('base line accuracy is '+str( max(rate,1-rate))+'\n')
    f1 = 2*rate/(1+rate)
    f.write('base line f1 value is '+str(f1))
    f.close()