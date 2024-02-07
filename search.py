import numpy as np  
import matplotlib.pyplot as plt 
import random
import math
t=1             #时间
vr=8            #机器人速度
vs=2            #潜水器速度

r0=10           #点间距
r=15            #搜查半径
rM=t**2        #雾半径 !!!!!!
rM0=rM
s=0             #需搜素的路程

#num=rM/r
n=0             #已搜素点个数
n0=0

v0=4/3*3.14*r**3   #探索体积
beta=1          #修正因子
v=n*beta*r**3*v0        #已搜素体积
v=1
#vM=4/3*3.14*rM**3      #雾团体积

a=1             #修正因子
p=1

#fig=plt.figure()
plt.title('Probability of successful rescue',fontsize=20)
plt.xlabel('t (min)')  
plt.ylabel('P')  

x=2

for _ in range(100):
    t=t+0.7
    if t > 20 :
        if n0 > 0:
            if s > 0:
                s=s-vr
                v=n*beta*v0            #已搜素体积
            else:
                n=n+1
                n0=n0-1
                s=rM0*random.random()*x
        else:
            rM0=rM
            s=rM0*random.random()*x
            n0=(rM0/r0)**3
            
    
        rM=t          #雾半径 !!!!!!
        vM=4/3*3.14*rM**3           #雾团体积
        p=v/(vM-v)/1.8
        plt.scatter(t,p)

plt.show()