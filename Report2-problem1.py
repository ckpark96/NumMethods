import numpy as np
import matplotlib.pyplot  as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la


# FLIP PLOTS
alpha = -5

def k(x,y):
    return 1 + 4*x + 6*y

def f(x,y):
    return \
    np.exp(alpha*(x-1)**2+alpha*(y-1)**2) + \
    np.exp(alpha*(x-3)**2+alpha*(y-1)**2) + \
    np.exp(alpha*(x-5)**2+alpha*(y-1)**2) + \
    np.exp(alpha*(x-7)**2+alpha*(y-1)**2) + \
    np.exp(alpha*(x-1)**2+alpha*(y-3)**2) + \
    np.exp(alpha*(x-3)**2+alpha*(y-3)**2) + \
    np.exp(alpha*(x-5)**2+alpha*(y-3)**2) + \
    np.exp(alpha*(x-7)**2+alpha*(y-3)**2)

h = 0.5
hx = h
hy = h
gridx = np.arange(h,8,h)
gridy = np.arange(h,4,h)
Nx = len(gridx)
Ny = len(gridy)

diag0 = np.zeros((len(gridx)*len(gridy)))
diagup = np.zeros((len(gridx)*len(gridy)))
diagdown = np.zeros((len(gridx)*len(gridy)))
diagmax = np.zeros((len(gridx)*(len(gridy)-1)))
diagmin = np.zeros((len(gridx)*(len(gridy)-1)))


m = 0
for j in gridy:
    for i in gridx:
        
        diag0[m] = (k(i-hx/2,j)+k(i+hx/2,j))/hx**2 + (k(i,j-hy/2)+k(i,j+hy/2))/hy**2

        if i != max(gridx) and m <= len(diagup):
            diagup[m] = -k(i+hx/2,j)/hx/hx

        if i != min(gridx) and m <= len(diagdown):
            diagdown[m] = -k(i-hx/2,j)/hx/hx

        if j != max(gridy) and m <= len(diagmin)-1:
            diagmax[m] = -k(i,j+hy/2)/hy/hy

        if j != min(gridy):
            diagmin[m-Nx] = -k(i,j-hy/2)/hy/hy

        m += 1

diagup = diagup[:-1]
diagdown = diagdown[1:]
diagonals = [diagmin, diagdown, diag0, diagup, diagmax]

L = sp.diags(diagonals, [-len(gridx), -1, 0, 1, len(gridx)], shape = (len(gridx)*len(gridy), len(gridx)*len(gridy)), format = 'csc')
    
check1 = np.allclose(diagup, diagdown, 0.0001, 0.0001)
check2 = np.allclose(diagmax, diagmin, 0.0001, 0.0001)

plt.figure()
plt.spy(L, markersize = 1)

grid = np.mgrid[h:8:h,h:4:h]
x = grid[0]
y = grid[1]
f = f(x,y)

plt.figure()
plt.imshow(f, origin = 'lower')
plt.colorbar()

f= np.reshape(f, (Nx*Ny,1), order='F')

u = la.spsolve(L,f)
u = np.reshape(u,(Ny,Nx)).T

plt.figure()
plt.imshow(u, origin = 'lower')
plt.colorbar()


k = k(x,y)
plt.figure()
plt.imshow(k, origin = 'lower')
plt.colorbar()
plt.show()