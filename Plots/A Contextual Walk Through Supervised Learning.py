import numpy as np
import matplotlib.pyplot as plt

ds_size = 50
x = np.random.uniform(0, 10, size = (2,ds_size) )
y = 2 + x[0] + 3*x[1] + 5*np.random.randn(ds_size)

# Figure 1
fig = plt.figure(figsize = (30,30))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[0], x[1], y, alpha = 1)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')



# + for Figure 2
point = [0,0,2] #point for point normal equation of a line
normal = np.array([1, 3, -1])  #normal for point normal equation of a line
d = -np.dot(point,normal)
xx, yy = np.meshgrid(range(11), range(11))
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2] #equation of plane
ax.plot_surface(xx, yy, z, color = 'orange',  alpha=0.5)


# + for Figure 3
for i in range(ds_size):
    xlin = [x[0,i],x[0,i] ]
    ylin = [x[1,i], x[1,i]]
    zlin = [y[i], 2+x[0,i]+3*x[1,i]] #line from x,y to the plane
    ax.plot(xlin,ylin,zlin,'r-',alpha=0.8, linewidth=1)


# + for rotate
ax.axis('off')
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(0.1)

# Figure 4
propx0, propx1 = np.meshgrid([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])
outz = (1/ds_size)*np.sum(np.array([0.5*(2 + propx0*x[0, i] + propx1*x[1, i] - y[i])**2 for i in range(len(y))]), axis = 0 ) #cost for given x, y
fig = plt.figure(figsize = (30,30))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('$a_1$')
ax.set_ylabel('$a_2$')
ax.set_zlabel('MSE @ $a_0 = 2$')
ax.plot_surface(propx0, propx1, outz)



