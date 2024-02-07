import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
import random
import decimal
import math
import numpy as np  
from sklearn.naive_bayes import GaussianNB  
from sklearn.preprocessing import LabelEncoder  
from scipy.stats import multivariate_normal


x0=0    #初始位置x
y0=0    #初始y
z0=0    #初始z

t=0     #时间
t_interval=2  #通讯时间间隔
t_limit=30  #时间限制
num=0
num_max=40

#陆地坐标系
p=0
q=0
theta=0     #x与m夹角

v_p=2
v_q=2

#潜水艇坐标系
x=0
y=0
z=0

v_x=2
v_y=2
v_z=-2 #下潜速度 m/s

#潜水器参数
length=8  #长 m
m=324000      #质量 kg  323
r=3         #等效半径 m
V=3.14*r*r*length+4/3*3.14*r**3*0.8 #体积

#雾状图
ppp = np.array([])  
qqq = np.array([])  
zzz = np.array([])  


def predict(x,y,z,v_x,v_y,v_z):
    #海水温度随深度关系式
    T0=18       #海面温度
    h=-z        #深度
    T=T0-16*(1-2.71**(-0.01*h))

    #盐度随深度变化
    S=0.035+0.0000005*h

    #海水密度
    rho=1020+1*(S-0.035)-1*(T-18)

    #洋流速度
    vC=random.random()*10+1
    currentX=random.random()*2-1    #x方向上的分量比例
    currentY=random.random()*2-1    
    ux=currentX*vC+v_x              #x方向上的分速度
    uy=currentY*vC+v_y

    #x
    #mu0=1.005*10**-3                   #20℃时动力粘度  Pa*s    
    mu=0.001779/(1+0.03368*T+0.0002210*t**2)    #动力粘度
    H=r*0.8       #球表面到中心距离
    fHr=0.7*(H/r)**-1.082+1.001
    F_front=fHr*6*10**5*3.14*mu*ux*r
    ax=F_front/m
    print(ax)
    #y
    ay=0.5*7*0.1**8*rho*uy*uy*2*3.14*r*length
    print(ay)
    #z
    g=9.8   #N/kg
    az=rho*g*V/m-g
    print(az)

    v_x=v_x+ax*t_interval
    v_y=v_y+ay*t_interval
    v_z=v_z+az*t_interval

    x = x+v_x*t_interval
    y = y+v_y*t_interval
    z = z+v_z*t_interval
    return x,y,z,v_x,v_y,v_z

#draw
fig = plt.figure()  
#fig.suptitle('Submersible position imitate',fontsize=24)
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Submersible position imitate',fontsize=20)
isPro = True


for num in range(0,num_max):
    t=t+t_interval
    if isPro:
        z = z + v_z*t_interval
        theta = math.atan(v_q/v_p)
        v_x = v_p*math.cos(theta)+v_q*math.sin(theta)
        v_y =-v_p*math.sin(theta)+v_q*math.cos(theta)   #将pq转为xy
    else:

        x,y,z,v_x,v_y,v_z=predict(x,y,z,v_x,v_y,v_z)

        v_p=v_x*math.cos(theta)-v_y*math.sin(theta) #将xy转为pq
        v_q=v_x*math.sin(theta)+v_y*math.cos(theta)
    
    p = p+v_p*t_interval
    q = q+v_q*t_interval

    if num >= 20 : 
        isPro=False
        ax.scatter(p,q,z,color='red') 

    else:
        ax.scatter(p,q,z,color='blue') 

    ppp = np.append(ppp, p/100-1)  
    qqq = np.append(qqq, q/100-1)  
    zzz = np.append(zzz, z/10+4)  


# 设置坐标轴标签  
ax.set_xlabel('P(m)')  
ax.set_ylabel('Q(m)')  
ax.set_zlabel('Z(m)')  


# 显示图形  
plt.show()



#Metropolis-Hastings算法


#绘制曲线
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
    current_sample = np.random.randn(3)        #范围

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
#plot_trajectory(ppp, qqq, zzz)

# 生成MCMC抽样
num_samples = 10000
step_size = 0.1
samples = metropolis_hastings(target_distribution, num_samples, step_size)

# 计算某个时间点在三维空间中出现的概率
target_time = 25 # 假设目标时间为10
target_point = np.array([ppp[target_time], qqq[target_time], zzz[target_time]])

# 计算每个样本点的概率
probabilities = np.exp(-np.sum((samples - target_point) ** 2, axis=1))

# 绘制点阵表示概率
fig = plt.figure()
#fig.suptitle('Submersible position prediction',fontsize=24)
ax = fig.add_subplot(111, projection="3d")
ax.set_title('Submersible position prediction',fontsize=20)


# 将概率映射到颜色
colormap = plt.get_cmap("viridis")
norm = plt.Normalize(vmin=probabilities.min(), vmax=probabilities.max())
colors = colormap(norm(probabilities))

# 绘制点阵
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=colors)

# 绘制曲线
ax.plot(ppp,qqq,zzz)

ax.set_xlabel("P (×10^2+100 m)")
ax.set_ylabel("Q (×10^2+100 m)")
ax.set_zlabel("Z (×10-40 m)")

plt.show()


