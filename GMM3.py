import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse
from sklearn import mixture
from sklearn.datasets import make_moons

sns.set()
'''

'''
# 导入数据
X, y_true = make_blobs(n_samples=400,
                       centers=4,
                       cluster_std=0.60,
                       random_state=0)
X = X[:, ::-1]  # 调整坐标系以获得更好图片

rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))

plt.clf()

gmm = mixture.GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
# plt.show()

probs = gmm.predict_proba(X)
print(probs[:5].round(3))

size = 50 * probs.max(1)**2  # square emphasizes differences
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size)


# 定义一个作图函数
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # 作图：设置参数
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # 作图：画椭圆
    for nsig in range(1, 4):
        ax.add_patch(
            Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


# 定义一个GMM的椭圆方程拟合函数
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


# 生成一份make_moons的数据集
plt.clf()
Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)
plt.scatter(Xmoon[:, 0], Xmoon[:, 1])
plt.show()

# 用2个中心聚类
plt.clf()
gmm2 = mixture.GaussianMixture(n_components=2, covariance_type='full')
plot_gmm(gmm2, Xmoon)
plt.show()

# 用16个中心拟合这组数据集
plt.clf()
gmm16 = mixture.GaussianMixture(n_components=16, covariance_type='full')
plot_gmm(gmm16, Xmoon, label=False)
plt.show()