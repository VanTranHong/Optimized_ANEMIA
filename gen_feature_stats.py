import glob, os

import numpy as np
import pandas as pd
root = "/Users/vantran/Desktop/anemia/STATS/NONMOTHER/"


##########GENERATING FEATURES PROBABILITY AND RENAME COLUMNS
folder = root+"features"
os.chdir(folder)
df = pd.DataFrame()
for f in glob.glob("*.csv"):
    dat = pd.read_csv(f,skipinitialspace=True, header = 0)
    
    filename = ""
    if "cfs_0" in f:
        filename = f.replace("cfs_0","CFS")
    elif "mrmr_0" in f:
        filename = f.replace("mrmr_0","MRMR")
    else:
        filename = f.replace("infogain_","IG-")
    filename = filename.replace(".csv","")
    df[filename] = dat["0"]/10

    



############ COMBINING 5 RUNS############


nrows = df.shape[0]

df0 = pd.DataFrame()

for col in df.columns:
    name = col[:-1]
    if name not in df0.columns:
        df0[name] = df[col]
    else:
        df0[name] = df[col]+df0[name]
# df0 = df0.drop(columns = ['Unnamed: '], axis=1)
df0["Average"] = df0.mean(axis=1)
df0["Features"] = list(range(1,nrows+1))
df0["Percentage"] = df0["Average"]/(df.shape[1]/5)*100

df0.to_csv(folder+"_average.csv")



# ############GENERATING TABLES TO BE USED IN HEATMAP FEATURES #######


# nrows = df0.shape[0]
# # data = df0.drop(columns = ['Unnamed: 0'], axis=1)
# feature_selection = df0.columns.drop('Features')



# df = pd.DataFrame(columns=['Feature',"Feature Selection","Probability"])
# for col in feature_selection:
#     stat = df0[col]
#     for i in range(nrows):
#         nw = {'Feature': i+1,'Feature Selection': col,"Probability":stat[i]/5}
#         df = df.append(nw, ignore_index=True)

# df.to_csv(root+"feature_finished.csv")





        
        
    