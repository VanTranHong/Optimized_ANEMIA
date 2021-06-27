import data_preprocess as dp
import numpy as np
import pandas as pd
# import scoring as score
# import normal_run as nr
import ranking_subset_run as rsr
import sfs_run as sfs_r
# import boost_bag_run as bbr
# import featureselection as fselect
# from sklearn.model_selection import StratifiedKFold
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import LinearSVC, SVC
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.naive_bayes import GaussianNB, BernoulliNB
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.linear_model import  SGDClassifier
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# import statsmodels.api as sm
import uni_multiStats as stats
# import pickle
# from openpyxl import Workbook, load_workbook
import pyreadstat as py
# from openpyxl.workbook import Workbook

############### DEALING WITH HIV DATA ############
#relabel hiv into hivstatus with 0 for no, 1 for yes, 2 for indeterminate

# hiv, meta = py.read_sav('/Users/vantran/Desktop/ANEMIA/raw_data/hiv.SAV')
# hiv['HIV03'].replace({2:1,3:1},inplace = True)
# hiv['HIV03'].replace({4:0,5:0, 6:0, 7:0, 8:0,9:0},inplace = True)
# hiv.to_csv('/Users/vantran/Desktop/ANEMIA/raw_data/hiv.csv')


############################# This converts SAV to CSV #################
###mother####
# df, meta = py.read_sav('/Users/vantran/Desktop/ANEMIA/raw_data/mother.SAV')
# df.to_csv('/Users/vantran/Desktop/ANEMIA/raw_data/mother_convert.csv')

####child#####
# df, meta = py.read_sav('/Users/vantran/Desktop/ANEMIA/raw_data/child.SAV')
# df.to_csv('/Users/vantran/Desktop/ANEMIA/raw_data/child_convert.csv')


######################### MERGING HIV DATA INTO THE DATASET ###############
# mother = pd.read_csv('/Users/vantran/Desktop/ANEMIA/raw_data/mother_convert.csv',skipinitialspace=True, header = 0)

# hiv = pd.read_csv('/Users/vantran/Desktop/ANEMIA/raw_data/hiv.csv',skipinitialspace=True, header = 0)
# CASEID = []
# for index in range(hiv.shape[0]):
#     row = hiv.iloc[index]
#     CASEID.append(str(int(row['HIVCLUST']))+"-"+str(int(row['HIVNUMB']))+"-"+str(int(row['HIVLINE'])))
# hiv['CASEID'] = CASEID
# HIV_stat = []
# for index in range(mother.shape[0]):
#     id = "-".join(mother.iloc[index]['CASEID'].split())
#     row = hiv[hiv['CASEID'] == id]
#     if row.shape[0]>0:
#         HIV_stat.append(row.iloc[0]['HIV03'])
#     else:
#         HIV_stat.append('N/A')
# mother['HIV'] = HIV_stat
# mother.to_csv('/Users/vantran/Desktop/ANEMIA/raw_data/mother_hivadded.csv')




# child = pd.read_csv('/Users/vantran/Desktop/ANEMIA/raw_data/child_convert.csv',skipinitialspace=True, header = 0)
# hiv = pd.read_csv('/Users/vantran/Desktop/ANEMIA/raw_data/hiv.csv',skipinitialspace=True, header = 0)
# CASEID = []
# for index in range(hiv.shape[0]):
#     row = hiv.iloc[index]
#     CASEID.append(str(int(row['HIVCLUST']))+"-"+str(int(row['HIVNUMB']))+"-"+str(int(row['HIVLINE'])))
# hiv['CASEID'] = CASEID
# HIV_stat = []
# for index in range(child.shape[0]):
#     id = "-".join(child.iloc[index]['CASEID'].split())
#     row = hiv[hiv['CASEID'] == id]
#     if row.shape[0]>0:
#         HIV_stat.append(row.iloc[0]['HIV03'])
#     else:
#         HIV_stat.append('N/A')
# child['HIV'] = HIV_stat
# child.to_csv('/Users/vantran/Desktop/ANEMIA/raw_data/child_hivadded.csv')





#####################       ADDING GIS DATA TO HIV ADDED DATA       ##########

# data = pd.read_csv('/Users/vantran/Desktop/ANEMIA/raw_data/mother_hivadded.csv',skipinitialspace=True, header = 0)
# data2 =pd.read_csv('/Users/vantran/Desktop/ANEMIA/raw_data/GIS.csv', skipinitialspace = True, header=0)
# selected = ['DHSCLUST','All_Population_Count_2015','Annual_Precipitation_2015','Day_Land_Surface_Temp_2015','Diurnal_Temperature_Range_2015','Enhanced_Vegetation_Index_2015','Global_Human_Footprint','Gross_Cell_Production','Growing_Season_Length','Irrigation','ITN_Coverage_2015','Land_Surface_Temperature_2015','Malaria_Prevalence_2015','Mean_Temperature_2015','Nightlights_Composite','Proximity_to_Water']
# GIS = data2[selected]
# df = pd.DataFrame(columns=selected)
# samples  = data.shape[0]
# clusters = GIS['DHSCLUST']
# num_GIS_variables = len(selected)
# zero_row = pd.DataFrame(0, index =[0],columns = selected )
# for i in range(samples):
#     clusnum = data.iloc[i,:]['V001']### This is the cluster number in the child or mother dataset#####
    
#     if clusnum in clusters:
#         row = GIS.loc[GIS['DHSCLUST']==clusnum]
        
#         df = df.append(row, ignore_index=True)
#     else:
#         df = df.append(zero_row, ignore_index=True)      
# GIS_added = pd.concat([data,df], axis =1)
# GIS_added = GIS_added.drop(['DHSCLUST'], axis=1)
# GIS_added.to_csv('/Users/vantran/Desktop/ANEMIA/raw_data/mother_hivgis_added.csv')


    
  

################# This filters out desired columns for child ##############
# data_child = pd.read_csv('/Users/vantran/Desktop/ANEMIA/raw_data/child_hivgis_added.csv',skipinitialspace=True, header = 0)
# child_desired = ['All_Population_Count_2015','Annual_Precipitation_2015','Day_Land_Surface_Temp_2015','Diurnal_Temperature_Range_2015','Enhanced_Vegetation_Index_2015','Global_Human_Footprint','Gross_Cell_Production','Growing_Season_Length','Irrigation','ITN_Coverage_2015','Land_Surface_Temperature_2015','Malaria_Prevalence_2015','Mean_Temperature_2015','Nightlights_Composite','Proximity_to_Water','V025','ToiletNew','HIV','HouseholdAmenities','AnyTransportation','FloorNew','V151','V404','PotatoesBeansLentils','MeatorFishNew','AnyFruitNew','anyDairyNew','AnyVegetables','MaternalAnemia','B4','ChildIPdrugs','ChildAnemia','ChildAgeGroups','V024','V106','HusbandEducationNew','MaternalOccNew','HusbandOccNew','WaterSourceNew','TimetoWaterNew','ReligionRecode','HouseholdNumGroup','V158','V159','V157','V190','BirthSizeGrouped','M45','HeightNew','WeightGroupNew','BirthOrderGrouped','V013']
# data_child = data_child[child_desired]
# data_child.to_csv('/Users/vantran/Desktop/ANEMIA/processed_data/child_selectedcolumns_processed.csv')

#####   MODIFYING DATA AND GEN STATISTICS     ########
# data = pd.read_csv('/Users/vantran/Desktop/ANEMIA/processed_data/child_selectedcolumns_processed.csv',skipinitialspace=True, header = 0)
# continuous = ['All_Population_Count_2015','Annual_Precipitation_2015','Day_Land_Surface_Temp_2015','Diurnal_Temperature_Range_2015','Enhanced_Vegetation_Index_2015','Global_Human_Footprint','Gross_Cell_Production','Growing_Season_Length','Irrigation','ITN_Coverage_2015','Land_Surface_Temperature_2015','Malaria_Prevalence_2015','Mean_Temperature_2015','Nightlights_Composite','Proximity_to_Water']
# nominal = data.columns.drop(continuous)
# target = 'ChildAnemia'
# category = "child"
# data1, nominal_groups = dp.modify_data(data,continuous,nominal,target)
# data1.to_csv('/Users/vantran/Desktop/ANEMIA/processed_data/'+category+'/modified_data.csv')
# stats.gen_stats(data1, target, nominal_groups, category)


################# This filters out desired columns for mother ##############
# data_mother = pd.read_csv('/Users/vantran/Desktop/ANEMIA/raw_data/mother_hivgis_added.csv',skipinitialspace=True, header = 0)
# mother_desired = ['All_Population_Count_2015','Annual_Precipitation_2015','Day_Land_Surface_Temp_2015','Diurnal_Temperature_Range_2015','Enhanced_Vegetation_Index_2015','Global_Human_Footprint','Gross_Cell_Production','Growing_Season_Length','Irrigation','ITN_Coverage_2015','Land_Surface_Temperature_2015','Malaria_Prevalence_2015','Mean_Temperature_2015','Nightlights_Composite','Proximity_to_Water','HannahAnemia','WaterTime','MaternalOccRecode','HusbandOccNew','HannahReligion','HannahHouseholdNum','V158','V157','V159','V190','HannahRecentBirths','HannahAllChildrenBorn','HannahContraception','HannahBMI','V106','HannahWater','V025','HannahToilet','HIV','HouseholdAmenities','TransportationAccess','HannahFloor','V151','V404','V213','V216','HannahMaritalStatus','MaternalIronRecode','V013','V024']
# data_mother = data_mother[mother_desired]
# data_mother.to_csv('/Users/vantran/Desktop/ANEMIA/processed_data/mother_selectedcolumns_processed.csv')

#####   MODIFYING DATA AND GEN STATISTICS     ########
# data = pd.read_csv('/Users/vantran/Desktop/ANEMIA/processed_data/mother_selectedcolumns_processed.csv',skipinitialspace=True, header = 0)
# continuous = ['All_Population_Count_2015','Annual_Precipitation_2015','Day_Land_Surface_Temp_2015','Diurnal_Temperature_Range_2015','Enhanced_Vegetation_Index_2015','Global_Human_Footprint','Gross_Cell_Production','Growing_Season_Length','Irrigation','ITN_Coverage_2015','Land_Surface_Temperature_2015','Malaria_Prevalence_2015','Mean_Temperature_2015','Nightlights_Composite','Proximity_to_Water']
# nominal = data.columns.drop(continuous)
# target = 'HannahAnemia'
# category = "mother"
# data1, nominal_groups = dp.modify_data(data,continuous,nominal,target,category)
# data1.to_csv('/Users/vantran/Desktop/ANEMIA/processed_data/'+category+'/modified_data.csv')
# stats.gen_stats(data1, target, nominal_groups, category)










############ GENERATING THE RUNS  AND FEATURES #####################
n_seed = 5
splits =10
target = 'ChildAnemia'
category = "child"
data1 = pd.read_csv('/Users/vantran/Desktop/ANEMIA/processed_data/'+category+'/modified_data.csv',skipinitialspace=True, header = 0)
data1 = data1.drop(columns = ['Unnamed: 0'], axis=1)

runs = stats.runSKFold(n_seed,splits,data=data1,target=target)
with open('/Users/vantran/Desktop/ANEMIA/processed_data/'+category+'/runs.txt',"wb") as  fp:
    pickle.dump(runs,fp)





    


# # # # #################### RUNNING WITHOUT BOOSTING AND BAGGING for all ranking feature selections and CFS###############

# sfs_r.subset_features(n_seed, splits, ['elasticnet'], ['f1'], runs, n_feature)

# # # 
# rsr.execute_feature_selection(runs, ['infogain_20'], n_features,n_seed,splits)
# rsr.execute_feature_selection(runs, ['infogain_40'], n_features,n_seed,splits)
# rsr.execute_feature_selection(runs, ['infogain_60'], n_features,n_seed,splits)
# rsr.execute_feature_selection(runs, ['infogain_80'], n_features,n_seed,splits)

# rsr.execute_feature_selection(runs, ['cfs_0'], n_features,n_seed,splits)
# rsr.execute_feature_selection(runs, ['mrmr_0'], n_features,n_seed,splits)
# score.score(rsr.normal_run(n_seed, splits, ['cfs_0'],['knn'],runs,n_feature),n_seed, splits)





# score.score(rsr.normal_run(n_seed, splits, ['mrmr_0'], ['svm'], runs,n_feature), n_seed, splits)
# score.score(sfs_r.subset_run(n_seed, splits,['elasticnet'],['f1'],runs,n_feature),n_seed,splits)
# sfs_r.subset_features(n_seed,splits, ['elasticnet'],['accuracy'],runs, n_features)
# score.score(bbr.boostbag_run(n_seed,splits,['_20'],['xgboost'],runs,'bag',n_features), n_seed,splits)
# score.score(bbr.boostbag_run(n_seed,splits,['cfs_0'],['knn'],runs,'bag',n_features), n_seed,splits)
# score.score(bbr.boostbag_run(n_seed,splits,['mrmr_0'],['naive_bayes'],runs,'boost',n_features), n_seed,splits)
# # subset_features(n_seed, splits, estimators,metrics, runs, n_features):





        
    
   
    
    
    

