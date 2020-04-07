import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 

# def createDataPoints(centroidLocation, numSamples, clusterDeviation):
#     # Create random data and store in feature matrix X and response vector y.
#     X, y = make_blobs(n_samples=numSamples, centers=centroidLocation, 
#                                 cluster_std=clusterDeviation)
    
#     # Standardize features by removing the mean and scaling to unit variance
#     X = StandardScaler().fit_transform(X)
#     return X, y

# X, y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)    

# epsilon = 0.3
# minimumSamples = 7
# db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
# labels = db.labels_
# print(labels)

# # Firts, create an array of booleans using the labels from db.
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# print(core_samples_mask)

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print(n_clusters_)

# # Remove repetition in labels by turning it into a set.
# unique_labels = set(labels)
# print(unique_labels)

# colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

# # Plot the points with colors
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = 'k'

#     class_member_mask = (labels == k)

#     # Plot the datapoints that are clustered
#     xy = X[class_member_mask & core_samples_mask]
#     plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

#     # Plot the outliers
#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker=u'o', alpha=0.5)
#     plt.savefig("Data/DBSCAN_result_for_K_means_comparisssion.png")
#     plt.show(block=False)
#     plt.pause(2)

# # Leaners code block - K Means clustering
# from sklearn.cluster import KMeans 

# k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)

# k_means.fit(X)

# k_means_labels = k_means.labels_
# # print(k_means_labels)

# k_means_cluster_centers = k_means.cluster_centers_
# # print(k_means_cluster_centers)

# # Initialize the plot with the specified dimensions.
# fig = plt.figure(figsize=(6, 4))

# # Colors uses a color map, which will produce an array of colors based on
# # the number of labels there are. We use set(k_means_labels) to get the
# # unique labels.
# colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# # Create a plot
# ax = fig.add_subplot(1, 1, 1)

# # For loop that plots the data points and centroids.
# # k will range from 0-3, which will match the possible clusters that each
# # data point is in.
# for k, col in zip(range(3), colors):

#     # Create a list of all data points, where the data poitns that are 
#     # in the cluster (ex. cluster 0) are labeled as true, else they are
#     # labeled as false.
#     my_members = (k_means_labels == k)
    
#     # Define the centroid, or cluster center.
#     cluster_center = k_means_cluster_centers[k]
    
#     # Plots the datapoints with color col.
#     ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
#     # Plots the centroids with specified color, but with a darker outline
#     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# # Title of the plot
# ax.set_title('KMeans')

# # Remove x-axis ticks
# ax.set_xticks(())

# # Remove y-axis ticks
# ax.set_yticks(())

# plt.savefig("Data/K_means_result_for_DBSCAN_comparisssion.png")

# # Show the plot
# plt.show(block=False)
# plt.pause(2)



# #########################################

import csv
import pandas as pd

filename='Data/weather-stations20140101-20141231.csv'

#Read csv
pdf = pd.read_csv(filename)
# print(pdf.head(5))

pdf = pdf[pd.notnull(pdf["Tm"])]
# print(pdf)
pdf = pdf.reset_index(drop=True)
# print(pdf)


###### Visualisation
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = (14,10)

llon=-140
ulon=-50
llat=40
ulat=65

pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) &(pdf['Lat'] < ulat)]

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

# To collect data based on stations        

xs,ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))
pdf['xm']= xs.tolist()
pdf['ym'] =ys.tolist()

#Visualization1
for index,row in pdf.iterrows():
#   x,y = my_map(row.Long, row.Lat)
   my_map.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)
#plt.text(x,y,stn)
plt.show()

#### CLustering

from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)
Clus_dataSet = pdf[['xm','ym']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# Compute DBSCAN
db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels)) 


# A sample of clusters
pdf[["Stn_Name", "Tx", "Tm", "Clus_Db"]].head(5)


### Visualisation of stations

set(labels)

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = (14,10)

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
#my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

# To create a color map
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))



#Visualization1
for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]                    
    my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm) 
        ceny=np.mean(clust_set.ym) 
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print("Cluster " + str(clust_number) + ', Avg Temp: ' + str(np.mean(clust_set.Tm)))
        
### Clustering based on temp and location

from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)
Clus_dataSet = pdf[['xm','ym','Tx','Tm','Tn']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels)) 


# A sample of clusters
pdf[["Stn_Name", "Tx", "Tm", "Clus_Db"]].head(5)

### Visualisation based on temp and location

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = (14,10)

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
#my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

# To create a color map
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))



#Visualization1
for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]                    
    my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm) 
        ceny=np.mean(clust_set.ym) 
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))











