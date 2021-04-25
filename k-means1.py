import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

sns.set()
'''
模型的初步建立
'''
# 导入数据
X, y_true = make_blobs(n_samples=400,
                       centers=4,
                       cluster_std=0.60,
                       random_state=0)
X = X[:, ::-1]  # 调整坐标系以获得更好图片

# 作图：关于kmeans
kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')

plt.show()

#测试数据
plt.clf()
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))
plt.scatter(X_stretched[:, 0],
            X_stretched[:, 1],
            c=labels,
            s=40,
            cmap='viridis')
plt.show()


# 定义一个作图函数
def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # 作图：输入数据
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # 作图：kmeans的中心演算
    centers = kmeans.cluster_centers_
    radii = [
        cdist(X[labels == i], [center]).max()
        for i, center in enumerate(centers)
    ]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))


'''
通过GMM寻找聚类：数据集1
'''
# 作图
# 在这里可以通过调整n_cluster的值，改变得到的分类数
# 取n_cluster=4可以得到最好的结果，取3或5的结果也不错
kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X)
plt.show()
'''
通过GMM寻找聚类：拉伸后的数据集
'''
# 作图
# 在这里可以通过调整n_cluster的值，改变得到的分类数
# 取n_cluster=4可以得到最好的结果，取3或5的结果也不错

plt.clf()
kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X_stretched)
plt.show()
