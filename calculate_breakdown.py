import data_preprocess as dp
import numpy as np
import pandas as pd

target = 'ChildAnemia'
data = pd.read_csv('data_child.csv',skipinitialspace=True, header = 0)
names = list(data.columns)
names.remove('Unnamed: 0')
names.remove('Unnamed: 0.1')
names.remove(target)
print(len(data[data['ChildAnemia']==1]))
print(len(data))


newnames = []
for name in names:
    newnames.append(name+"positive")
    newnames.append(name+"negative")
    
d = {'Features': newnames, 'Positive': [0 for i in range(2*len(names))], 'Negative': [0 for i in range(2*len(names))], 'Total': [0 for i in range(2*len(names))]}
results = pd.DataFrame(data=d)
for i in range(len(names)):
    name = names[i]
    group = data[data[name]==1]
    total = len(group)
    positive = len(group[group[target]==1])
    negative =total-positive
    results.loc[2*i,'Total']=total
    results.loc[2*i,'Positive']=positive
    results.loc[2*i,'Negative']=negative
    neggroup = data[data[name]!=1]
    total = len(neggroup)
    positive = len(neggroup[neggroup[target]==1])
    negative =total-positive
    results.loc[2*i+1,'Total']=total
    results.loc[2*i+1,'Positive']=positive
    results.loc[2*i+1,'Negative']=negative
    
results.to_csv('child_breakdown.csv')