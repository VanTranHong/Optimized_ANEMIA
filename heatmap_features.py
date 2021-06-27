import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 



############## THIS IS FOR UNIVARIATE AND MULTIVARIATE##########

###### reading from csv file that contains 3 columns, featur
#  selection, risk factor and probability ######
# df = pd.read_csv('/Users/vantran/Desktop/oddsratio.csv')

# # df['P-Value'] = df['P-Value']<0.05
# # df['P-Value'] = df['P-Value']


# result = df.pivot('method','index','Modified')

# Methods = ['uni','multi' ]
# Columns = [5,4,7,12,1,18,13,24,16,27,10,14,9,25,6,21,26,2,3,11,15,22,17,23,8,19,20]
# result = result.reindex(index = Methods,columns = Columns)

# fig,ax = plt.subplots(figsize = (12,7))
# sns.heatmap(result, linewidths=0.30,cmap="vlag")
# plt.savefig('oddsratio.png')
# plt.show()

################ THIS IS FOR NORMAL GENERATION OF FEATURE HEATMAP
root = "/Users/vantran/Desktop/RESEARCH/anemia/STATS/revised_stats/"
df = pd.read_csv(root+'child_GIS_sum_part2.csv')
root = root+'images/child/'




###### reading from csv file that contains 3 columns, feature selection, risk factor and probability ######
# df = pd.read_csv('/Users/vantran/Desktop/PLOTS/heatmapfeatures.csv')
result = df.pivot('Feature','Feature Selection', 'Probability')


fig,ax = plt.subplots(figsize=(16,9))
###### reordering the rows and columns #######
FeaturesName = df['Feature'].unique()


result = result.reindex(index = FeaturesName)
# plt.title(title,fontsize = 30)
# df['Feature Selection']=pd.Categorical(df['Feature Selection'], categories=Features,ordered=True)


plt.xlabel("Risk Factor", fontsize = 20)
plt.ylabel("Feature Selection", fontsize = 20)
# ttl = ax.title
# ttl.set_position([0.5,1.05])


ax.set_xticks([])
ax.set_yticks([])



# res = sns.heatmap(pd.crosstab(df['Feature Selection'],df['Feature']),cmap ='RdYlGn',annot = False,linewidths=0.30,ax=ax)
# res = sns.heatmap(result,fmt ='.0%',cmap ='RdYlGn',cbar=False,annot = True,annot_kws={"size": 16},linewidths=0.30,ax=ax )
res = sns.heatmap(result,cmap ='RdYlGn',annot = False,linewidths=0.30,ax=ax )#annot_kws={"size": 10},fmt ='.2%'
res.set_xticklabels(res.get_xmajorticklabels(),fontsize =20)
res.set_yticklabels(res.get_ymajorticklabels(),fontsize =8,rotation =45)
colorbar  = res.collections[0].colorbar
colorbar.ax.locator_params(nbins =5)
plt.savefig(root+'featurechild_part2.png')
# sns.clustermap(result, linewidths=0.30)
# plt.savefig('cluster.png')

plt.show()




















# plt.subplots(figsize=(4,4))
# ###### reordering the rows and columns #######

# # plt.title(title,fontsize = 30)
# # df['Feature Selection']=pd.Categorical(df['Feature Selection'], categories=Features,ordered=True)
# plt.xlabel("Risk Factor", fontsize = 20)
# plt.ylabel("Feature Selection", fontsize = 20)
# # # ttl = ax.title
# # # ttl.set_position([0.5,1.05])
# # ax.set_xticks([])
# # ax.set_yticks([])
# # ax.set


# res = sns.heatmap(pd.crosstab(df['Feature Selection'],df['Features']),cmap ='RdYlGn',annot = False,linewidths=0.30,ax=ax)

# # res = sns.heatmap(result,cmap ='RdYlGn',annot = False,linewidths=0.30,ax=ax )#annot_kws={"size": 10},fmt ='.2%'
# res.set_xticklabels(res.get_xmajorticklabels(),fontsize =15)
# res.set_yticklabels(res.get_ymajorticklabels(),fontsize =15,rotation =45)
# colorbar  = res.collections[0].colorbar
# colorbar.ax.locator_params(nbins =3)
# plt.savefig('feature.png')
####### sns.clustermap(result, linewidths=0.30,figsize=(10,7),cbar_pos=(0, .2, .03, .4),cmap="vlag")
# cm.set_axis_labels('Risk Factor','Feature Selection')

 
# hm = cm.ax_heatmap.get_position()
# plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), fontsize=15)
# plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), fontsize=15)
# plt.setp(cm.ax_heatmap.xaxis., fontsize=20)
# cm.ax_heatmap.set_position([hm.x0, hm.y0, hm.width, hm.height])
# col = cm.ax_col_dendrogram.get_position()
# cm.ax_col_dendrogram.set_position([col.x0, col.y0, col.width, col.height])





########plt.savefig('/Users/vantran/Desktop/anemia/STATS/NONMOTHER/cluster_feature.png')

#######plt.show()

#RdYlGn
