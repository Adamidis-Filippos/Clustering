import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

path = r'C:\Users\santo\Desktop\ML Course\Part 4 - Clustering\Section 25 - Hierarchical Clustering\Python\Mall_Customers.csv'
df = pd.read_csv(path)
x = df.iloc[:,[3,4]].values

dendrogram = sch.dendrogram(sch.linkage(x, method= 'ward'))
plt.title('Dendrogram')
plt.xlabel('customers')
plt.ylabel('euclidean distances')
plt.show()

# optimal clusters= 5 according to dendrogram
hc = AgglomerativeClustering(n_clusters=5, affinity= 'euclidean', linkage= 'ward')
y_hc = hc.fit_predict(x)


plt.scatter(x[y_hc==0,0],x[y_hc==0,1], s =100, c= 'red', label= 'cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1], s =100, c= 'green', label= 'cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1], s =100, c= 'blue', label= 'cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1], s =100, c= 'orange', label= 'cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1], s =100, c= 'yellow', label= 'cluster 5')
plt.title('Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

hc = AgglomerativeClustering(n_clusters=3, affinity= 'euclidean', linkage= 'ward')
y_hc = hc.fit_predict(x)


plt.scatter(x[y_hc==0,0],x[y_hc==0,1], s =100, c= 'red', label= 'cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1], s =100, c= 'green', label= 'cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1], s =100, c= 'blue', label= 'cluster 3')
plt.title('Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


