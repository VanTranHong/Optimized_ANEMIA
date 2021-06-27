import pandas as pd 
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def remove_invalid_column(data,target):
    """Function that removes columns that aren't suitable for machine learning.
    This includes features with more than 5% missing values, wrong data type,
    and the indices.

    Args:
        data (Pandas DataFrame): The DataFrame that contains data that hasn't been preprocessed.
        target: the dependent variable of the dataset

    Returns:
        DataFrame: Preprocessed DataFrame.
    """
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis = 1)
    data1 = data[data.columns[data.isnull().mean()<1]]
    # data2 = data2.drop('Unnamed: 0', axis = 1)
    if target not in data1.columns:
        data1 = pd.concat([data1,data[target]], axis =1)
    data2 = data1.select_dtypes(exclude=['object'])
    data2 = data2[data2[target].notna()] 
    return data2

def recategorize(data,category):
    
    """Recategorizes the data according to the standards in the SPSS file.
    This functions aim to relavel the reference point as 0 so that in the dummification process, it will be removed

    Args:
        data (DataFrame): DataFrame containing data that hasn't been preprocessed.

    Returns:
        DataFrame: A DataFrame with replaced values.
    """
    
    #################### This is for child data ###############
    if category=="child":
        data['V025'].replace({1:0,2:1},inplace = True)
        data["FloorNew"].replace({0: -1, 1: 0}, inplace=True)
        data["FloorNew"].replace({-1: 1}, inplace=True) 
        data["HouseholdAmenities"].replace({0: -1, 1: 0}, inplace=True)
        data["HouseholdAmenities"].replace({-1: 1}, inplace=True) 
        data['V151'].replace({1:0,2:1},inplace = True)
        data['B4'].replace({1:0,2:1},inplace = True)
        data["ChildIPdrugs"].replace({0: -1, 1: 0}, inplace=True)
        data["ChildIPdrugs"].replace({-1: 1}, inplace=True) 
        data["ChildAgeGroups"].replace({0: -1, 5: 0}, inplace=True)
        data["ChildAgeGroups"].replace({-1: 5}, inplace=True) 
        data["V024"].replace({11: 0}, inplace=True) 
        data["MaternalOccNew"].replace({0: -1, 2: 0}, inplace=True)
        data["MaternalOccNew"].replace({-1: 2}, inplace=True) 
        data["HusbandOccNew"].replace({0: -1, 2: 0}, inplace=True)
        data["HusbandOccNew"].replace({-1: 2}, inplace=True) 
        data["V190"].replace({5: 0}, inplace=True) 
        data["HeightNew"].replace({0: -1, 3: 0}, inplace=True)
        data["HeightNew"].replace({-1: 3}, inplace=True) 
        data["WeightGroupNew"].replace({0: -1, 4: 0}, inplace=True)
        data["WeightGroupNew"].replace({-1: 4}, inplace=True) 
        data["M45"].replace({8: 0}, inplace=True) 
        data["V013"].replace({7: 0}, inplace=True) 
    
    else:
    
    ########## This is for mother data ##############
    
        data['V025'].replace({1:0,2:1},inplace = True)
        data["HannahFloor"].replace({0: -1, 1: 0}, inplace=True)
        data["HannahFloor"].replace({-1: 1}, inplace=True) 
        data["HouseholdAmenities"].replace({0: -1, 1: 0}, inplace=True)
        data["HouseholdAmenities"].replace({-1: 1}, inplace=True) 
        data['V151'].replace({1:0,2:1},inplace = True)
        data["HannahMaritalStatus"].replace({0: -1, 1: 0}, inplace=True)
        data["HannahMaritalStatus"].replace({-1: 1}, inplace=True) 
        data["MaternalIronRecode"].replace({0: -1, 1: 0}, inplace=True)
        data["MaternalIronRecode"].replace({-1: 1}, inplace=True) 
        data["V013"].replace({7: 0}, inplace=True) 
        data["V024"].replace({11: 0}, inplace=True) 
        data["V190"].replace({5: 0}, inplace=True) 
        data["MaternalOccRecode"].replace({0: -1, 2: 0}, inplace=True)
        data["MaternalOccRecode"].replace({-1: 2}, inplace=True) 
        data["HusbandOccNew"].replace({0: -1, 2: 0}, inplace=True)
        data["HusbandOccNew"].replace({-1: 2}, inplace=True) 
    return data


def getnominal(data,nominal):
    """Finds the features that are:
    1. have nominal values
    2. have more than 2 distinc values so it needs to be dummified

    Args:
        data (DataFrame): DataFrame containing the dataset.
        nominal: the list of nominal columns

    Returns:
        List: A list that contains the nominal features.
    """
    
    
    
    returnarr = []
    non_dummy =[]
    number_distinct =[]
    for col in nominal:
        if col in data.columns:
            distinct = data[col].dropna().nunique()
            if distinct > 2:
                returnarr.append(col)  
                number_distinct.append(distinct-1)
            else:
                non_dummy.append([col])  
    return returnarr, number_distinct, non_dummy

def create_dummies (data):
    """Creates dummy variables.

    Args:
        data (DataFrame): DataFrame containing the dataset

    Returns:
        DataFrame: DataFrame containing the dataset with dummy variables.
    """
    dummy = pd.get_dummies(data, columns = data.columns, drop_first= True) 
    return dummy




def normalize_data(data, continuous):
    
    """
    normalize continuous data so mean is 0 and standard deviation is 1
    
    Args:
        data (DataFrame): DataFrame containing the dataset
        continuous: list of continuous columns

    Returns:
        DataFrame: DataFrame whose continuous columns are already normalized 
    
    
    """
    returnarr =[]
    for col in continuous:
        if col in data.columns:
            data[col].replace({-9999:None}, inplace = True)
            returnarr.append(col)
    
    
    for col in returnarr:
        mean = np.mean(data[col])
        std = np.std(data[col])
        normed = (data[col]-mean)/std
        data[col] = normed
            
    
    return data
def impute(data):
    """Multivariate imputer that estimates each feature from all the others

    Args:
        data (Numpy Array): A numpy array containing the dataset.

    Returns:
        Numpy Array: A matrix with imputed values.
    """
    imputed_data = []
    for i in range(5):
        imputer = IterativeImputer(sample_posterior=True, random_state=i)
        imputed_data.append(imputer.fit_transform(data))
       
    returned_data = np.round(np.mean(imputed_data,axis = 0))
    return returned_data

def group_nominals(names, num_distinct):
    start = 0
    groups = []
    for num in num_distinct:
        group =[]
        for i in range(num):
            group.append(names[start])
            start+=1
        groups.append(group)
    return groups

            

def modify_data(data, numerical, nominal,target, category):
    """Runs all the preprocessing functions on the dataset.

    Args:
        data (DataFrame): DataFrame containing the dataset with no preprocessing.
        numerical (List): List containing all the features that are ordinal.
        nominal (List): List containing all the features that are nominal.
        target: the dependent variable  of the dataset

    Returns:
        DataFrame: DataFrame with all the preprocessing done.
    """
    data1 = remove_invalid_column(data, target)
    data2 = recategorize(data1,category)
    nominal, num_distinct, non_dummy = getnominal(data2,nominal) 
    nominal_data = create_dummies (data2[nominal])
    new_nominal_groups = group_nominals(nominal_data.columns, num_distinct)
    groups = new_nominal_groups + non_dummy
    groups.remove([target])
    data3 = data2.drop(nominal, axis = 1)
    data4 = pd.concat([data3,nominal_data], axis =1)
    data4 = normalize_data(data4, numerical)
    return data4, groups
