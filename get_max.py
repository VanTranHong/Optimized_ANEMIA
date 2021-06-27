import glob, os

import numpy as np
import pandas as pd





os.chdir("/Users/vantran/Desktop/anemia/STATS/mother/resultsparallel")
df = pd.DataFrame(columns=['filename','Accuracy', 'F1'])
for file in glob.glob("*finalscore.csv"):
    data = pd.read_csv(file,skipinitialspace=True, header = 0)
    max_ac = data['Average Accuracy'].max()
    max_f1 = data['Average F1'].max()
    df = df.append({'filename':file,'Accuracy':max_ac,'F1': max_f1}, ignore_index=True)
df.to_csv('/Users/vantran/Desktop/anemia/STATS/mother/max_scores_in_summary.csv',index=True)
print(df) 
