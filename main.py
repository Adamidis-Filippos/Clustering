import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

path = r'C:\Users\santo\Desktop\ML Course\Part 4 - Clustering\Section 24 - K-Means Clustering\Python\Mall_Customers.csv'
df = pd.read_csv(path)
x = df.iloc[:,[3,4]].values


#elbow method
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init= 'k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init= 'k-means++', random_state=42)
y_pred = kmeans.fit_predict(x)


plt.scatter(x[y_pred == 0,0],x[y_pred == 0,1],s=100, c= 'red', label= 'Cluster 1')
plt.scatter(x[y_pred == 1,0],x[y_pred == 1,1],s=100, c= 'blue', label= 'Cluster 2')
plt.scatter(x[y_pred == 2,0],x[y_pred == 2,1],s=100, c= 'green', label= 'Cluster 3')
plt.scatter(x[y_pred == 3,0],x[y_pred == 3,1],s=100, c= 'orange', label= 'Cluster 4')
plt.scatter(x[y_pred == 4,0],x[y_pred == 4,1],s=100, c= 'cyan', label= 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s= 300, c= 'yellow', label= 'Centroids')
plt.xlabel('Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()




