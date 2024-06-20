#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.tri import Triangulation
from scipy.sparse import csc_matrix

def K(x,y):
    outer_circ_rad = 1
    if np.any(np.abs(np.sqrt(x**2 + y**2) - outer_circ_rad) < 1e-12): # on outer circle, Neumann BC.
        return 0
    else: # on inner circle, Robin BC.
        return 1

class LaplDiscr:
    """
    A class providing a FE spatial discretization of the Laplace operator
    
    u_xx + u_yy

    with boundary condition 
    
    K(x,y)*u + du/dn = g(t)
    
    for a given mesh.
    
    Example usage:
        
        from Laplace import LaplDiscr
        mesh_filename = "mesh.mat"
        lapl = LaplDiscr(mesh_filename)
        M,A,G = lapl.get_system()

    Attributes
    ----------
    p : NumPy array
        the x- and y-coordinates of the points in the mesh
    e : NumPy array
        indices of the starting and ending points of the edges of the mesh
    t : NumPy array
        indices of the corner points of each triangle in the mesh
    N : int
        number of points (degrees of freedom)

    Methods
    -------
    get_system(g)
        Assembles and returns the discretization matrices M (mass matrix) and 
        A (stiffness matrix), and the function G(t) corresponding to the ODE
        M v_t = A v + G(t) (heat equation) or
        M v_tt = A v + G(t) (wave equation)
        with boundary data given by g(t).
        
    plot_solution(v,title_str="")
        Plots the solution represented in v on the mesh. The string in title_str
        is added as the title to the plot.
        
    plot_mesh()
        Plots the mesh.
    """
    
    def __init__(self,filename):
        mesh_file = loadmat(filename)
        self.p = mesh_file['p']
        self.e = mesh_file['e']
        self.t = mesh_file['t']
        
        # adjust for matlab indexing
        self.t[0:3,:] = self.t[0:3,:] - 1
        self.e[0:2,:] = self.e[0:2,:] - 1
        
        self.N = self.p.shape[1]
        
    def get_system(self,g):
        A = StiffMat2D(self.p,self.t)
        M = MassMat2D(self.p,self.t)
        R = RobinMat2D(self.p,self.e)
        r_inner = RobinVec2D(self.p,self.e)

        # Scale the Robin boundary data constribution by the time-dependent 
        # function g(t).
        def G(t):
            return r_inner*g(t)
        
        return csc_matrix(M),csc_matrix(-(A+R)),G
    
    def get_initial(self):
        return np.zeros(self.N)
    
    def plot_solution(self,v,title_str=""):
        tri = Triangulation(self.p[0,:],self.p[1,:],self.t[0:3,:].T)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(tri,v, cmap=cm.jet, linewidth=0)
        plt.axis([-1,1,-1,1])
        ax.set_zlim(-1,1)
        surf.set_clim(-1,1)
        plt.xlabel('x')
        plt.ylabel('y')
        fig.colorbar(mappable=surf)
        plt.title(title_str)
        plt.show()

    def plot_mesh(self):
        tri = Triangulation(self.p[0,:],self.p[1,:],self.t[0:3,:].T)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.triplot(tri)
        plt.axis([-1,1,-1,1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

# Functions for assembling the matrices. Taken from
#
# Larson, M.G., Bengzon, F. (2013). The Finite Element. In: The Finite Element 
# Method: Theory, Implementation, and Applications. Texts in Computational 
# Science and Engineering, vol 10. Springer, Berlin, Heidelberg. 
#
def RobinMat2D(p,e):
    Np = p.shape[1]
    Ne = e.shape[1]
    R = np.zeros((Np,Np))
    
    E_idx = 0
    for E_idx in range(Ne):
        loc2glb = e[0:2,E_idx].astype(int)
        x = p[0,loc2glb]
        y = p[1,loc2glb]
        length = np.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)
        RK = K(x,y)/6*np.array([[2,1],[1,2]])*length
        mat_idx = np.ix_(loc2glb.tolist(),loc2glb.tolist())
        R[mat_idx] = R[mat_idx] + RK
        
    return R

def RobinVec2D(p,e):
    Np = p.shape[1]
    Ne = e.shape[1]
    r = np.zeros(Np)
    
    E_idx = 0
    for E_idx in range(Ne):
        loc2glb = e[0:2,E_idx].astype(int)
        x = p[0,loc2glb]
        y = p[1,loc2glb]
        length = np.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)
        rK = K(x,y)*np.array([1,1])*length/2
        r[loc2glb] = r[loc2glb] + rK
        
    return r

def StiffMat2D(p,t):
    Np = p.shape[1]
    Nt = t.shape[1]
    A = np.zeros((Np,Np))
    for T_idx in range(Nt):
        loc2glb = t[0:3,T_idx].astype(int)
        x = p[0,loc2glb]
        y = p[1,loc2glb]
        area = np.abs(np.sum((x[[1,2,0]] - x)*(y[[1,2,0]] + y)))/2
        b = np.array([y[1]-y[2],y[2]-y[0],y[0]-y[1]])/(2*area)
        c = np.array([x[2]-x[1],x[0]-x[2],x[1]-x[0]])/(2*area)
        AK = (np.tensordot(b,b,axes=0) + np.tensordot(c,c,axes=0))*area
        mat_idx = np.ix_(loc2glb.tolist(),loc2glb.tolist())
        A[mat_idx] = A[mat_idx] + AK
    return A

def MassMat2D(p,t):
    Np = p.shape[1]
    Nt = t.shape[1]
    M = np.zeros((Np,Np))
    for T_idx in range(Nt):
        loc2glb = t[0:3,T_idx].astype(int)
        x = p[0,loc2glb]
        y = p[1,loc2glb]
        area = np.abs(np.sum((x[[1,2,0]] - x)*(y[[1,2,0]] + y)))/2
        MK = area*np.array([[2,1,1],[1,2,1],[1,1,2]])/12
        mat_idx = np.ix_(loc2glb.tolist(),loc2glb.tolist())
        M[mat_idx] = M[mat_idx] + MK
    return M