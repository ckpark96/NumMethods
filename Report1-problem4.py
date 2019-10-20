import numpy as np
import matplotlib.pyplot  as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la


h = 0.02
Nx = int(2/h)
Ny = int(1/h)

Dx = np.zeros((Nx, Nx-1))
np.fill_diagonal(Dx, 1)
np.fill_diagonal(Dx[1:,:], -1)

Dy = np.zeros((Ny, Ny-1))
np.fill_diagonal(Dy, 1)
np.fill_diagonal(Dy[1:,:], -1)

Lxx = Dx.T@Dx
Lyy = Dy.T@Dy
L = sp.kronsum(Lxx, Lyy)
L = L/h**2
print(L)
plt.figure()
plt.spy(L,marker = 'o' , markersize = 6,color = 'green')

gridy = np.arange(0+h,1,h)
gridx = np.arange(0+h,2,h)


grid = np.mgrid[0+h:2:h,0+h:1:h]
x = grid[0]
y = grid[1]

f = 20*np.multiply(np.sin(np.pi*y),np.sin(1.5*np.pi*x+np.pi))
f = np.transpose(f)

plt.figure()
plt.imshow(f,origin='lower')
plt.colorbar()

f = np.transpose(f)
x0 = np.sin(2*np.pi*gridy)/h**2
x2 = np.sin(2*np.pi*gridy)/h**2
y0 = np.sin(0.5*np.pi*gridx)/h**2
y1 = 0*gridx/h**2

for i in np.arange(0,Ny-1):
    f[0,i] = f[0,i] + x0[i]
    f[-1,i] = f[-1,i] + x2[i]
    
for j in np.arange(0,Nx-1):
    f[j,0] = f[j,0] + y0[j]

f= np.reshape(f, ((Nx-1)*(Ny-1),1), order='F')


u = la.spsolve(L,f)
u = np.reshape(u,(Ny-1,Nx-1))[::-1,:]

plt.figure()
plt.imshow(u,origin='lower')
plt.colorbar()

plt.show()