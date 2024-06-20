#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplg
from Laplace import LaplDiscr
from time import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.tri import Triangulation
from rungekutta4 import step

animate_solution = True
plot_interval = 0.02 # update the plot in these intervals
Tend = 2

use_LU = False
mesh_filename = "mesh.mat"

lapl = LaplDiscr(mesh_filename)

# The boundary data.
def g(t):
    return 0

# Build discretization matrices
M,A,G = lapl.get_system(g)

# plt.spy(A)
# plt.show()

# Initial data
def u0(x,y):
    x0 = 0.5
    y0 = -0.5
    r = 0.1
    return np.exp(-(x - x0)**2/r**2 - (y - y0)**2/r**2)

v0 = u0(lapl.p[0,:],lapl.p[1,:])
vt0 = 0*v0

# First order form
B = spsp.bmat([[spsp.eye(lapl.N),None],[None,M]],format='csc')
D = spsp.bmat([[None,spsp.eye(lapl.N)],[A,None]],format='csc')
w = np.hstack((v0,vt0))


# LU factorization
B_lu = spsplg.splu(B)

# plt.spy(B_lu.L)
# plt.show()

def rhs(w):
    return B_lu.solve(D@w)

# Time stepping
dt_try = 1e-3
Nt = int(np.ceil(Tend/dt_try))
dt = Tend/Nt

if animate_solution:
    zmin = -0.2
    zmax = 0.2
    tri = Triangulation(lapl.p[0,:],lapl.p[1,:],lapl.t[0:3,:].T)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(tri,v0, cmap=cm.jet, linewidth=0)
    plt.axis([-1,1,-1,1])
    ax.set_zlim(zmin,zmax)
    surf.set_clim(zmin,zmax)
    plt.xlabel('x')
    plt.ylabel('y')
    fig.colorbar(mappable=surf)
    plt.title("t = 0")

start_time = time()
t = 0
for tidx in range(Nt):
    w, t = step(rhs,w,t,dt)

    if animate_solution and ((tidx % int(plot_interval/dt)) == 0 or tidx == Nt-1):
        v = w[:lapl.N]
        surf.remove()
        surf = ax.plot_trisurf(tri,v, cmap=cm.jet, linewidth=0)
        surf.set_clim(zmin,zmax)
        plt.title("t = " + '{0:.2f}'.format(dt*(tidx+1)))
        plt.pause(1e-4)
    
print('Execution time: {0:.4f} seconds'.format(time() - start_time))

# To stop the program from finishing until figure is closed
if animate_solution:
    plt.show()