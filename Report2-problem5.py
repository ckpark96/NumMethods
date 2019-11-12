import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.sparse.linalg as la

xmax = 10
ymax = 10
h = 0.1
Nx = int(xmax/h)  #Nx is total number of points (including boundary) - 1
Ny = int(ymax/h)

def f(x,y):
    return np.exp(-10*(x-5)**2 - 10*(y-5)**2)

reshaper0 = lambda u: np.reshape(u, (Nx-1)*(Ny-1), order = 'F')
reshaper1 = lambda u: np.reshape(u,(int(Ny)-1,int(Nx)-1))[::-1,:]

def grid_form(h, xmax, ymax):
    return np.mgrid[h:xmax:h,h:ymax:h]

def L_form(h, Nx, Ny):
    Dx = np.zeros((Nx, Nx-1))
    np.fill_diagonal(Dx, 1)
    np.fill_diagonal(Dx[1:,:], -1)
    
    Dy = np.zeros((Ny, Ny-1))
    np.fill_diagonal(Dy, 1)
    np.fill_diagonal(Dy[1:,:], -1)
    
    Lxx = Dx.T@Dx
    Lyy = Dy.T@Dy
    L = sp.kronsum(Lxx, Lyy)/h**2    
    return -L

def sourcefn(h, Nx, Ny,grid, func):
    x = grid[0]
    y = grid[1]
    
    sourcefn = func(x,y)
    return np.reshape(sourcefn, ((int(Nx)-1)*(int(Ny)-1)), order='F')

grid = grid_form(h, xmax, ymax)
u0 = reshaper0( f(grid[0],grid[1]) )
A = L_form(h, Nx, Ny)
source = sourcefn(h, Nx, Ny, grid, f)
residuals = []

def cback(rk):
    residuals.append(rk)

xk, info = sp.linalg.gmres(A, source, tol = 1e-12, callback = cback, maxiter = 5000, restart = 5000)
save = residuals

fig = plt.figure()
fig.suptitle("GMRES", fontsize=10)
sour = plt.subplot(221)
sour.title.set_text('source function')
plt.imshow(reshaper1(source))
numba = 2
logresi = []
spsolvesol = []
eig = []

for i in [-40, 0, 40]:
    key = 'key' + str(numba)
    residuals = []
    A += sp.eye(A.shape[0])*i
    eigvalues = la.eigs(A)[0]
    eig.append(eigvalues)
    x = la.spsolve(A, source)
    spsolvesol.append(x)
    xk = sp.linalg.gmres(A, source, tol = 1e-12, callback = cback, maxiter = 5000, restart = 5000)[0]
    lastrk = residuals[-1]
    verify = abs(lastrk - np.linalg.norm(source - A*xk)/np.linalg.norm(source))
    logresi.append(residuals)
    print("Verification: Absolute difference = ", verify)
    key = plt.subplot(220+numba)
    key.title.set_text('$\gamma = %d$'%(i))
    plt.imshow(reshaper1(x))
    numba += 1
    A -= sp.eye(A.shape[0])*i


plt.figure()
plt.semilogy(logresi[0], label = '$\gamma$ = -40')
plt.semilogy(logresi[1], label = '$\gamma$ = 0')
plt.semilogy(logresi[2], label = '$\gamma$ = +40')
plt.legend()




zeros = np.zeros(eig[0].shape)
fig3 = plt.figure()
fig3.suptitle("Eigenvalues", fontsize=10)
one = plt.subplot(311)
plt.scatter(eig[0],zeros)
one.title.set_text('$\gamma$ = -40')
two = plt.subplot(312)
plt.scatter(eig[1],zeros)
two.title.set_text('$\gamma$ = 0')
three = plt.subplot(313)
plt.scatter(eig[2],zeros)
three.title.set_text('$\gamma$ = 40')


plt.show()





