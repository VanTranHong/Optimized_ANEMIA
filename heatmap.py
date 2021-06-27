import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 



######## reading csv file to draw the heapmap of accuracy or F1 score either for no boost or bag, boost or bag

root = "/Users/vantran/Desktop/RESEARCH/anemia/STATS/revised_stats/"
df = pd.read_csv(root+'child_accuracy_f1.csv')
df['Accuracy'] = round(df['Accuracy'],3)
df['F1'] = round(df['F1'],3)
filenames = df["filename"]
featureselection = []
classifier = []
boostbag = []
for name in filenames:
  
        
    if "elasticnet" in name:
        classifier.append("LR")
    elif "knn" in name:
        classifier.append("KNN")
    elif "naive_bayes" in name:
        classifier.append("NB")
    elif "rdforest" in name:
        classifier.append("RF")
    elif "svm" in name:
        classifier.append("SVM")
    elif "xgboost" in name:
        
        classifier.append("XGB")
        
    if "bag_finalscore" in name:
        boostbag.append("bag")
    elif "boost_finalscore" in name:
        boostbag.append("boost")
    else:  
        boostbag.append("none")
    
    if "cfs" in name:
        featureselection.append("CFS")
    elif "mrmr" in name:
        featureselection.append("MRMR")
    elif "infogain" in name:
        num = ''.join(i for i in name if i.isdigit())
        featureselection.append("IG-"+num)
    else:
        featureselection.append("notselected")
        
size=20
root = root+'images/child/'
        
df["Feature Selection"] = featureselection
df["Classifier"] = classifier
df["Boost or Bag"] = boostbag
df = df[df["Feature Selection"]!="notselected"]

# df.to_csv(root+"filenamesorted.csv")

boost = df[df["Boost or Bag"] == "boost"]
result = boost.pivot(index = 'Feature Selection',columns = 'Classifier', values = 'Accuracy')

fig,ax = plt.subplots(figsize=(12,7))
plt.xlabel(" Feature Selection", fontsize = 25)
plt.ylabel(" Classifier", fontsize = 25)
ax.set_xticks([])
ax.set_yticks([])
res = sns.heatmap(result,fmt ='.1%',cmap ='RdYlGn',annot = True,annot_kws={"size": 16},linewidths=0.30,ax=ax )
res.set_xticklabels(res.get_xmajorticklabels(),fontsize =size)
res.set_yticklabels(res.get_ymajorticklabels(),fontsize=size)
plt.savefig(root+'boost_Accuracy.png')
plt.show()
result = boost.pivot(index = 'Feature Selection',columns = 'Classifier', values = 'F1')

fig,ax = plt.subplots(figsize=(12,7))
plt.xlabel(" Feature Selection", fontsize = 25)
plt.ylabel(" Classifier", fontsize = 25)
ax.set_xticks([])
ax.set_yticks([])
res = sns.heatmap(result,fmt ='.1%',cmap ='RdYlGn',annot = True,annot_kws={"size": 16},linewidths=0.30,ax=ax )
res.set_xticklabels(res.get_xmajorticklabels(),fontsize =size)
res.set_yticklabels(res.get_ymajorticklabels(),fontsize=size)
plt.savefig(root+'boost_F1.png')
plt.show()
        
        
bag = df[df["Boost or Bag"] == "bag"]
result = bag.pivot(index = 'Feature Selection',columns = 'Classifier', values = 'Accuracy')

fig,ax = plt.subplots(figsize=(12,7))
plt.xlabel(" Feature Selection", fontsize = 25)
plt.ylabel(" Classifier", fontsize = 25)
ax.set_xticks([])
ax.set_yticks([])
res = sns.heatmap(result,fmt ='.1%',cmap ='RdYlGn',annot = True,annot_kws={"size": 16},linewidths=0.30,ax=ax )
res.set_xticklabels(res.get_xmajorticklabels(),fontsize =size)
res.set_yticklabels(res.get_ymajorticklabels(),fontsize=size)
plt.savefig(root+'bag_Accuracy.png')
plt.show()
result = bag.pivot(index = 'Feature Selection',columns = 'Classifier', values = 'F1')

fig,ax = plt.subplots(figsize=(12,7))
plt.xlabel(" Feature Selection", fontsize = 25)
plt.ylabel(" Classifier", fontsize = 25)
ax.set_xticks([])
ax.set_yticks([])
res = sns.heatmap(result,fmt ='.1%',cmap ='RdYlGn',annot = True,annot_kws={"size": 16},linewidths=0.30,ax=ax )
res.set_xticklabels(res.get_xmajorticklabels(),fontsize =size)
res.set_yticklabels(res.get_ymajorticklabels(),fontsize=size)
plt.savefig(root+'bag_F1.png')
plt.show()
        
none = df[df["Boost or Bag"] == "none"]
# print(none)
result = none.pivot(index = 'Feature Selection',columns = 'Classifier', values = 'Accuracy')

fig,ax = plt.subplots(figsize=(12,7))
plt.xlabel(" Feature Selection", fontsize = 25)
plt.ylabel(" Classifier", fontsize = 25)

ax.set_xticks([])
ax.set_yticks([])
res = sns.heatmap(result,fmt ='.1%',cmap ='RdYlGn',annot = True,annot_kws={"size": 16},linewidths=0.30,ax=ax )
res.set_xticklabels(res.get_xmajorticklabels(),fontsize =size)
res.set_yticklabels(res.get_ymajorticklabels(),fontsize=size)
plt.savefig(root+'noboostbag_Accuracy.png')
plt.show()
result = none.pivot(index = 'Feature Selection',columns = 'Classifier', values = 'F1')

fig,ax = plt.subplots(figsize=(12,7))
plt.xlabel(" Feature Selection", fontsize = 25)
plt.ylabel(" Classifier", fontsize = 25)
ax.set_xticks([])
ax.set_yticks([])
res = sns.heatmap(result,fmt ='.1%',cmap ='RdYlGn',annot = True,annot_kws={"size": 16},linewidths=0.30,ax=ax )
res.set_xticklabels(res.get_xmajorticklabels(),fontsize =size)
res.set_yticklabels(res.get_ymajorticklabels(),fontsize=size)
plt.savefig(root+'noboostbag_F1.png')
plt.show()
        
        
        
        
        
    







# # result = result.reindex(index = Feature Selection,columns = Columns)
# # plt.title(title,fontsize = 30)

# # ttl = ax.title
# # ttl.set_position([0.5,1.05])



# # sns.set(font_scale=1.4)
# # res = sns.clustermap(result, linewidths=0.30,figsize=(10,7),annot = True,annot_kws={"size": 16},cbar_pos=None,cmap="vlag")




# # res.set_xticklabels(res.get_xmajorticklabels(),fontsize =20)
# # res.set_yticklabels(res.get_ymajorticklabels(),fontsize =20,rotation =45)
# # res.colorbar(False)
# # colorbar  = res.collections[0].colorbar
# # colorbar.ax.locator_params(nbins =4)
# # colorbar.ax.tick_params(labelsize = 15)


# # sns.clustermap(result, cbar=True,annot = True,annot_kws={"size": 16},linewidths=0.30)
# # plt.savefig('cluster.png')


# #RdYlGn
