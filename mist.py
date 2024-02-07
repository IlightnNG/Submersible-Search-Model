import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# 生成随机的三维轨迹线
def generate_trajectory():
    num_points = 100
    x = np.random.randn(num_points)
    y = np.random.randn(num_points)
    z = np.random.randn(num_points)
    return x, y, z

# 绘制三维轨迹线
def plot_trajectory(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

# 目标分布函数（假设为多元正态分布）
def target_distribution(x, y, z):
    mean = np.zeros(3)
    cov = np.eye(3)
    return multivariate_normal.pdf([x, y, z], mean=mean, cov=cov)

# Metropolis-Hastings算法
def metropolis_hastings(target_distribution, num_samples, step_size):
    samples = np.zeros((num_samples, 3))
    current_sample = np.random.randn(3)

    for i in range(num_samples):
        proposed_sample = current_sample + np.random.randn(3) * step_size
        acceptance_ratio = min(
            1,
            target_distribution(*proposed_sample) / target_distribution(*current_sample)
        )
        if np.random.rand() < acceptance_ratio:
            current_sample = proposed_sample
        samples[i] = current_sample

    return samples

# 生成并绘制随机的三维轨迹线
x, y, z = generate_trajectory()
plot_trajectory(x, y, z)

# 生成MCMC抽样
num_samples = 10000
step_size = 0.1
samples = metropolis_hastings(target_distribution, num_samples, step_size)

# 计算某个时间点在三维空间中出现的概率
target_time = 20 # 假设目标时间为10
target_point = np.array([x[target_time], y[target_time], z[target_time]])

# 计算每个样本点的概率
probabilities = np.exp(-np.sum((samples - target_point) ** 2, axis=1))

# 绘制点阵表示概率
fig = plt.figure()
fig.suptitle('Submersible position prediction')
ax = fig.add_subplot(111, projection="3d")

# 将概率映射到颜色
colormap = plt.get_cmap("viridis")
norm = plt.Normalize(vmin=probabilities.min(), vmax=probabilities.max())
colors = colormap(norm(probabilities))

# 绘制点阵
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=colors)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
