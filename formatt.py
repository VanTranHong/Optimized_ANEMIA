import pandas as pd
import numpy as np
import math
import csv
# import sklearn
# import random
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression, Lasso
# from sklearn.model_selection import KFold
# from sklearn import metrics
# from sklearn import preprocessing
# from sklearn.model_selection import cross_validate
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn import svm, datasets
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
# import scipy.stats as stats
# from scipy.stats import chi2_contingency
# from sklearn.feature_selection import SelectKBest 
# from sklearn.feature_selection import chi2 
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_selection import mutual_info_classif
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
# from statsmodels.stats.multicomp import pairwise_tukeyhsd


##importing relevant fuctions from other files#####
# from algorithm import svm,rdforest,lasso
# from modifydata import checkcategory,fillin
# from featureselection import infogains,infogain,featureselection,selectffeature




######this code is used to convert sav file to csv file when necessary######
import pyreadstat as py
from openpyxl.workbook import Workbook
df, meta = py.read_sav('/Users/vantran/Downloads/ETAR71FL.SAV')
# writer=pd.ExcelWriter ("converted.xlsx")
# df.to_excel(writer, 'df')
df.to_csv('/Users/vantran/Desktop/HIVstatus.csv')
# writer.save()

##### THIS IS DONE FOR MATT######

# #### this code convert excel to sav file when necessary####
# read_excel = pd.read_excel('Child Anemia Data Uncleaned.xlsx')
# read_excel.to_csv('Child Anemia not cleaned.csv', index = None, header = True)
# read_excel = pd.read_excel('Maternal Data Not Cleaned.xlsx')
# read_excel.to_csv('Marternal not cleaned data.csv', index = None, header = True)



# ##dropping rows with the same CASE ID done for Maternal
# motherdata = pd.read_csv('Marternal not cleaned data.csv',skipinitialspace=True, header = 0)

# dropduplicated = motherdata.drop_duplicates(subset = ['CASE ID'], keep = 'first')
# dropduplicated.to_excel("motheroutput.xlsx") 


# ##selecting the youngest child of each unique ID for Child
# childdata = pd.read_csv('Child Anemia not cleaned.csv',skipinitialspace=True, header = 0)
# print(childdata.shape)
# uniqueID = childdata['CASE ID'].unique()
# print(uniqueID.shape)
# returnframe = pd.DataFrame(columns=childdata.columns)

# for ID in uniqueID:
#     duplicates = childdata[childdata['CASE ID']==ID]   
#     duplicates = duplicates.sort_values(by = 'Child Age in Months')
#     returnframe = returnframe.append(duplicates.iloc[0,:])
    
                           

# returnframe.to_excel("childoutput.xlsx") 



    
   
        







# ######    main    ####
# data = pd.read_csv(datafile,skipinitialspace=True, header = 0)
# data = data[data.columns[data.isnull().mean()<0.05]] #excluding columns with too many missing data
# data = data.select_dtypes(exclude=['object']) #excluding columns with wrong data type
# fillin(data)
# data = data.drop(labels = ['COLGATEID']) 
# print(data)
# # data = data.astype(int)  

 


# fixdata(data) 
# print(data)
# ###obtaining relevant features#####
# #items = infogains(data) 
# items = selectffeature(data, 30)
# items = np.concatenate((items[0:20],data_consider),axis = 0)




# ###running algorithm

# rdforest(data.filter(items = items))
# svm(data.filter(items = items))
# lasso(data.filter(items = items))







            

    
    



# ####handling the old data

# #reading files and removing rows or columns with too many missing places
# datafile = r'/Users/vantran/Documents/vantran/coverted.csv'
# min_max_scaler = preprocessing.MinMaxScaler()


   



        


# #######    main    ####
# data = pd.read_csv(datafile,skipinitialspace=True, header = 0)

# ##fixing data####
# data = data.loc[data['Hpyloriantigen'].isnull()==False]
# data = data.loc[data['Hpyloriantibody'].isnull()==False]           
# #data_consider = ['Hpyloriantigen','Hpyloriantibody','age','sex','maternaleducation','maternaloccupation','historyofvaccination','breastfeeding','dewormingmedicationinthelast6month','howmanypeoplelivinginyourhome','cigarettesmokersinthehouse','numberofsmokersinthehouse','typeofroof','wallsaremadeof','floortypeofthehouse','floorcoveredbyamaterial','wheredoyoucook','howoftenusecharcoalforcooking','howoftenusewoodforcooking','howoftenuseleavesforcooking','howoftenusedungforcooking','howoftenusenaftaorlambaforcooking','howoftenusegasforcooking','howoftenuseelectricityforcooking','mainsourceofwater','typeoftoilet','cat','dog','hen','coworox','sheep','horse','pig','goat','muleordonkey']   
# data = data[data.columns[data.isnull().mean()<0.05]] #excluding columns with too many missing data
# data = data.select_dtypes(exclude=['object']) #excluding columns with wrong data type
# fillin(data)
# data = data.astype(int)   

# data_consider = ['Hpyloriantigen','Hpyloriantibody']
# data = data.drop(labels=['spsscode','originalcode'], axis = 1)
# fixdata(data) 
# ###obtaining relevant features#####
# #items = infogains(data) 
# items = selectffeature(data, 30)
# items = np.concatenate((items[0:20],data_consider),axis = 0)




# ####running algorithm

# #rdforest(data.filter(items = items))
# svm(data.filter(items = items))
# #lasso(data.filter(items = items))







            

    
    
        

      

        
        
        
        

       
    
    




  









  
  
  
  
  
  
  
  
  
  
  

        
        
    
    















# losreg.fit(X=x_train, y=y_train)
# predictions = linreg.predict(X=x_test)
# error = predictions-y_test
# rmse = np.sqrt((np.sum(error**2)/len(x_test)))
# coefs = linreg.coef_
# features = x_train.columns
# '''


# '''
# #regularization
# alphas = np.linspace(0.0001, 1,100)
# rmse_list = []
# best_alpha = 0

# for a in alphas:
#     lasso = Lasso(fit_intercept = True, alpha = a, max_iter= 10000 )
    
#     kf = KFold(n_splits=10)
#     xval_err =0
    
    
    
    
#     for train_index, validation_index in kf.split(x_train):
    
#         lasso.fit(x_train.loc[train_index,:], y_train[train_index])
      
#         p = lasso.predict(x_train.iloc[validation_index,:])
#         err = p-y_train[validation_index]
#         xval_err = xval_err+np.dot(err,err)
#         rmse_10cv = np.sqrt(xval_err/len(x_train))
#         rmse_list.append(rmse_10cv)
#         best_alpha = alphas[rmse_list==min(rmse_list)]
      
        
# #using the alpha calculated to calculate accuracy of the test
# lasso = Lasso(fit_intercept = True, alpha = best_alpha)
# lasso.fit(x_train, y_train)
# predictionsOnTestdata_lasso = lasso.predict(x_test)
# predictionErrorOnTestData_lasso = predictionErrorOnTestData_lasso-y_test
# rmse_lasso 


        
        










    
