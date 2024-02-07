import matplotlib.pyplot as plt
import numpy as np
import random

from matplotlib import cbook, cm
from matplotlib.colors import LightSource


# Load and format data
filename='testData.asc'
#filename='gebco_2023_n39.6606_s34.3521_w16.2598_e20.8125.asc'
#dem = np.loadtxt(filename)

z = np.loadtxt(filename,skiprows=0,delimiter=' ')
nrows, ncols = z.shape

#map parameters
ncols =       1093
nrows =       1275
xllcorner =   16.258333333333
yllcorner =   34.350000000000
cellsize =    0.004166666667
NODATA_value = -32767

print (z.shape)
x = np.linspace(xllcorner, xllcorner+cellsize*ncols, ncols)
y = np.linspace(yllcorner, yllcorner+cellsize*nrows, nrows)
x, y = np.meshgrid(x, y)

#region = np.s_[0:2000:10, 0:2000:10]
region = np.s_[400:600, 750:950]
x, y, z = x[region], y[region], z[region]/10




# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
#ax.set_title('Elevation chart of the Ionian Sea',fontsize=20)
ax.set_title('Block Optimal Retrieval Models',fontsize=20)



ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)


#随机个点
num_points = 10  
random_x = np.random.rand(num_points) * 0.4+19.5  
random_y = np.random.rand(num_points) * 0.4+36.2  
random_z = np.random.rand(num_points) *-200-100
#ax.scatter(random_x,random_y,random_z)
scatter = ax.scatter(random_x, random_y, random_z, color='black', marker='o', s=50)
#ax.plot(random_x,random_y,random_z)
ax.plot(random_x, random_y, random_z, color='red', marker=None, linestyle='-', linewidth=2) 

ax.set_xlabel('longitude (°)')  
ax.set_ylabel('latitude (°)')  
ax.set_zlabel('h (m)') 

plt.show()