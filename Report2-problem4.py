import time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.sparse.linalg as la

xmax = 16
ymax = 8
a = 1
h = 0.2
t_instants = [0, 1, 2, 3, 5, 10, 20, 40]
dt = 0.99*h**2/4
Nx = int(xmax/h)  #Nx is total number of points (including boundary) - 1
Ny = int(ymax/h)
tmax = 40

def u(x,y):
    return np.exp(-2*(x-1.5)**2 - 2*(y-1.5)**2)

reshaper0 = lambda u: np.reshape(u, (Nx-1)*(Ny-1), order = 'F')
reshaper1 = lambda u: np.reshape(u,(int(Ny)-1,int(Nx)-1))[::-1,:]

def grid_form(h, xmax, ymax):
    return np.mgrid[h:xmax:h,h:ymax:h]

def L_form(Nx, Ny):
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

def forward(u0, dt, A, k):
    return u0 + dt*(A@u0) + dt*(u0*(np.ones(u0.shape)-u0))*k

def backward_NR(u0, dt, A, k):
    ui = u0.copy()
    i = 0
    
    while True:
        i += 1
        iterations.append(i)
        J = A + sp.eye(A.shape[0]) - 2*sp.diags(ui) 
        f = (A@ui) + (ui*(1-ui))*k
        vi = la.spsolve(sp.eye(J.shape[0])-dt*J , ui-u0-dt*f)
        ui = ui - vi
        NR_error.append(np.linalg.norm(vi))
        if np.linalg.norm(vi) < 0.001:
            return ui

def backward_P(u0, dt, A, k):
    ui = u0
    while True:
        f = (A@ui) + (ui*(np.ones(ui.shape)-ui))*k
        uh = dt*f + u0
        if (abs(uh-ui) < 0.001).all():
            return uh
        ui = uh
        

def Fisher_solver(u0, tmax, dt, A, t_instants, k, method='forward'):
    algo = forward
    if method == 'backward_NR':
        algo = backward_NR
        dt = 0.4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    if method == 'backward_P':
        algo = backward_P
        dt *= 1
        
    t = 0
    instants = [u0]
    
    while t <= 40:
        t += dt
        print(t)
        uh = algo(u0, dt, A, k)        
        u0 = uh
        for i in t_instants:
            if abs(t-i) < dt/2:
                instants.append(uh)
    return instants

grid = grid_form(h, xmax, ymax)
u0 = reshaper0( u(grid[0],grid[1]) )
A = L_form(Nx, Ny)

grid = np.zeros((Nx-1,Ny-1))

def k(h, Nx, Ny):
    grid = np.zeros((Nx-1,Ny-1))
    R1 = np.ones(((int(1/h)+1),(int(1/h)+1)))
    R2 = np.ones(((int(2/h)+1),(int(2/h)+1)))
    R3 = np.ones(((int(3/h)+1),(int(3/h)+1)))
    R4 = np.ones(((int(3/h)+1),(int(2/h)+1)))
    R5 = np.ones(((int(2/h)+1),(int(2/h)+1)))
    
    grid[int(1/h)-1:int(2/h),int(1/h)-1:int(2/h)] = R1
    grid[int(1/h)-1:int(3/h),int(3/h)-1:int(5/h)] = R2
    grid[int(4/h)-1:int(7/h),int(4/h)-1:int(7/h)] = R3
    grid[int(9/h)-1:int(12/h),int(4/h)-1:int(6/h)] = R4
    grid[int(13/h)-1:int(15/h),int(1/h)-1:int(3/h)] = R5
    
    return reshaper0(grid)
    
    
    
    
    
k = k(h, Nx, Ny)    

#start = time.time()
#instants = Fisher_solver(u0, tmax, dt, A, t_instants, k)
#print('Forward time = ', time.time()-start)



NR_error = []
iterations = []
start = time.time()
Fisher_solver(u0, tmax, dt, A, t_instants, k, 'backward_NR')
print('Backward NR time = ', time.time()-start)

fig = plt.figure()
fig.suptitle("h = %f" % h, fontsize=10)
plt.scatter(iterations, NR_error)
plt.show()

#start = time.time()
#Fisher_solver(u0, tmax, dt, A, t_instants, k, 'backward_P')
#print('Backward Picard time = ', time.time()-start)

#mx = np.max(np.array(instants))
#mn = np.min(np.array(instants))


#fig = plt.figure()
#fig.suptitle("h = %d" % h, fontsize=10)
#
#plt.subplot(331)
#plt.imshow(reshaper1(instants[0]))
#plt.subplot(332)
#plt.imshow(reshaper1(instants[1]))
#plt.subplot(333)
#plt.imshow(reshaper1(instants[2]))
#plt.subplot(334)
#plt.imshow(reshaper1(instants[3]))
#plt.subplot(335)
#plt.imshow(reshaper1(instants[4]))
#plt.subplot(336)
#plt.imshow(reshaper1(instants[5]))
#plt.subplot(337)
#plt.imshow(reshaper1(instants[6]))
#plt.subplot(338)
#plt.imshow(reshaper1(instants[7]))
#
#plt.show()