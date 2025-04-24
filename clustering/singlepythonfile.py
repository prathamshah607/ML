import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

orignal = pd.read_csv("Clustering_Data.csv")
data = orignal.copy()
titles = list(data)

def z(X):
    means     = np.mean(X, axis=0)
    stdevs  = np.std(X, axis=0)+0.00001
    X_norm = (X - means) / stdevs    

    return X_norm, means, stdevs

data, means, stddevs = z(data)

def return_closest_centroid (point, centroids):
    distances = []
    for index, item in enumerate(centroids):
        distances.append(np.linalg.norm(point - item))
    return (centroids[distances.index(min(distances))], sum(distances))

def map_points(centroids, points):
    mapping = []
    for index, point in enumerate(points):
        mapping.append({"point" : point, "centroid" : return_closest_centroid(point, centroids)[0]})
    return mapping

def get_means_from_clusters(pointmap, corrcentroids):
    means = []
    for centroid in corrcentroids:
        means.append(np.mean(np.array([point["point"] for point in pointmap if (point["centroid"] == centroid).all()]), axis=0))
    return means

def graphing (points, centroids):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.grid()
    for centroid in centroids:
        values = [point["point"] for point in points if (point['centroid'] == centroid).all()]   
        xs = list(map(lambda x:x[0], values))
        ys = list(map(lambda x:x[1], values))
        zs = list(map(lambda x:x[2], values))
        ax.scatter(xs, ys, zs, s = 50)
    ax.set_title('3D Scatter Plot')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()
    
def calculate_cluster_variance(data):
    points = np.array([d['point'] for d in data])
    centroids = np.array([d['centroid'] for d in data])
    unique_centroids, indices = np.unique(centroids, axis=0, return_inverse=True)
    variances = []
    for i in range(len(unique_centroids)):
        cluster_points = points[indices == i]
        if len(cluster_points) > 0:
            centroid = unique_centroids[i]
            squared_distances = np.sum((cluster_points - centroid) ** 2, axis=1)
            variance = np.mean(squared_distances)
            variances.append(variance)
        else:
            variances.append(0)

    return variances

distances = []
def kmeans (points, init_centres, max_iterations):
    global distances
    pointmap = []
    centres = init_centres
    for i in range(0, max_iterations):
        pointmap = map_points(centres, points)
        centres = get_means_from_clusters(pointmap, centres)
        
        sumofpoints = 0
        for point in points:
            sumofpoints += return_closest_centroid(point, centres)[1]
        print(f"iter. {i} : sum of distance of all (normalised) points from their nearest centroid: {sumofpoints}")
        distances.append(sumofpoints)
        try:
            if(distances[-1] == distances[-2]):
                print("\n*** The model has reached maximum efficiency. There is no need to run more iterations now. ***\n")
                break
        except:
            continue
    pointmap = map_points(centres, points)
    print("CENTRES OF CLUSTERS:")
    print(centres)
    print(f"\n\nSum of variances with {len(init_centres)} clusters: {sum(calculate_cluster_variance(pointmap))}")
    graphing(pointmap, centres)

pca = PCA(n_components=3)
data_pca = pca.fit(data)
data_trans = pca.transform(data)
data_trans = pd.DataFrame(data=data_trans)
data = data_trans
data.head()

kmeans(data.values, data.values[0:3], 20)



