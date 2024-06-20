
import matplotlib.pyplot as plt
from operators_np import sbp_cent_4th
import numpy as np
from math import pi, inf, ceil

mm=1000

m = 51
xl = 0
xr = 1
sigma = -100
xvec,h = np.linspace(xl,xr,m,retstep=True)
H,HI,D1,D2,M,e_l,e_r,d1_l,d1_r = sbp_cent_4th(m,h)
A = -M + sigma*(np.tensordot(e_l,e_l, axes=0) + np.tensordot(e_r,e_r,axes=0))

SBP_SAT=HI@A
ee, ev=np.linalg.eig(SBP_SAT)
fig = plt.figure()
plt.plot(np.real(ee),np.imag(ee),'b*')
plt.xlabel('$\Re$')
plt.ylabel('$\Im$')
plt.title('Eigenvalues of SBP-SAT approximation ')


L=np.tril(A,-1);U=np.triu(A,1);D=np.diag(np.diag(A))
B_GS=-np.linalg.inv(D+L)@U
B_J =-np.linalg.inv(D)@(L+U)

ee, ev=np.linalg.eig(B_GS)
e_max_GS=np.max(np.absolute(ee))

ee, ev=np.linalg.eig(B_J)
e_max_J=np.max(np.absolute(ee))


ww=np.linspace(1,2,mm)
e_max=np.zeros(mm)
for i in range(mm):
    w=ww[i]
    B_SOR=-np.linalg.inv(D+w*L)@(w*U+(w-1)*D)
    ee, ev=np.linalg.eig(B_SOR)
    e_max[i]=np.max(np.absolute(ee))
    #print(ee)
    

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([1,2])
plt.plot(ww,e_max,'r*')
plt.xlabel('$\omega$')
plt.ylabel('rho')
plt.title('Spectral radius SOR ')
#plt.legend()


R=np.min(e_max)
Ri=np.argmin(e_max)
w=ww[Ri]
B_SOR=-np.linalg.inv(D+w*L)@(w*U+(w-1)*D)
ee, ev=np.linalg.eig(B_SOR)
CondS=np.linalg.cond(ev)
print(CondS)

print(e_max_J)    # Spectral radius Jacobi
print(e_max_GS)   # Spectral radius Gauss-Seidel
print(R)          # Spectral radius optimal SOR
N=np.ceil(-(6+np.log10(CondS))/np.log10(R))   # Tolerance 10^{-6} leads to N iterations
print(Ri)

#Show plot
plt.show()

