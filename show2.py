import numpy as np
import matplotlib.pyplot as plt

# 3D diagrams require additional import modules
from mpl_toolkits.mplot3d import Axes3D

# Convert the default figure diagram to a 3D diagram
fig = plt.figure()
ax = Axes3D(fig)

# Get the coordinates of x and y
X = np.arange(-2, 2, 0.25)
Y = np.arange(-2, 2, 0.25)
# Draw the grid lines
X, Y = np.meshgrid(X, Y)

# Get the value of Z
R2 = X ** 2 + Y ** 2
Z = R2

# Draw the picture
# rstride: horizontal dividing line span (the smaller the denser)
# cstride: longitudinal dividing line span (the smaller the denser)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), edgecolor='grey')

# Projection
# zdir is deciding which direction to project
# offset indicates the coordinate plane corresponding to which point is projected to the azimuth coordinate
ax.contourf(X, Y, Z, zdir='z', offset=-5, cmap='rainbow')

# Limit the range of axes for drawing
ax.set_zlim(-5, 10)
# Get rid of the axes
# ax.axis("off")

plt.savefig("poincare_embedding2")
plt.show()
