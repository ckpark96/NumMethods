import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

#variables
h = 0.02
c = 1
xmax = 4
ymax = 12
Nx = xmax/h  #Nx is total number of points (including boundary) - 1
Ny = ymax/h
dt = 0.99*h/c/np.sqrt(2)
timeinstants = [1,2,3,4,5,6,7,8]

def grid_form(h, xmax, ymax):
    return np.mgrid[h:xmax:h,h:ymax:h]
    
def sourcefn(t, h, xmax, ymax):
    alpha = -50
    v = 4
    grid = np.mgrid[h:xmax:h,h:ymax:h]
    x = grid[0]
    y = grid[1]
    Nx = xmax/h
    Ny = ymax/h
    
    sourcefn = np.sin(2*np.pi*v*t)*np.exp(alpha*(x-2)**2+alpha*(y-2)**2)
    return np.reshape(sourcefn, ((int(Nx)-1)*(int(Ny)-1)), order='F')
    

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

def step0(dt, c, A, f, u0):
    return (c*dt)**2/2 * (A@u0 + f)

def step(dt, c, A, f, u1, u0):
    return 2*u1 - u0 + (c*dt)**2 * (A@u1 + f) 
    
    
def wave_solver(xmax, ymax, h, A, u0, c, dt, timeinstants):
    instants = []
    t = 0
    f = sourcefn(t, h, xmax, ymax)
    u1 = step0(dt, c, A, f, u0)
    u0 = u1
    
    while t <= 8:
        t += dt
        f = sourcefn(t, h, xmax, ymax)
        uh = step(dt, c, A, f, u1, u0)
        u0 = u1
        u1 = uh
        
        for i in timeinstants:
            if abs(t-i) < dt/2:
                instants.append(uh)
    
    return instants

u0 = np.zeros(((int(Nx)-1)*(int(Ny)-1)))
L = L_form(int(Nx), int(Ny))
instants = wave_solver(xmax, ymax, h, L, u0, c, dt, timeinstants)

mx = np.max(np.array(instants))
mn = np.min(np.array(instants))
reshaper = lambda u: np.reshape(u,(int(Ny)-1,int(Nx)-1))[::-1,:]

fig = plt.figure()
fig.suptitle("c = %d" %(c), fontsize=10)

one = plt.subplot(181)
plt.imshow(reshaper(instants[0]), vmin=mn, vmax=mx)
one.title.set_text('t = 1')
two = plt.subplot(182)
plt.imshow(reshaper(instants[1]), vmin=mn, vmax=mx)
two.title.set_text('t = 2')
three = plt.subplot(183)
plt.imshow(reshaper(instants[2]), vmin=mn, vmax=mx)
three.title.set_text('t = 3')
four = plt.subplot(184)
plt.imshow(reshaper(instants[3]), vmin=mn, vmax=mx)
four.title.set_text('t = 4')
five = plt.subplot(185)
plt.imshow(reshaper(instants[4]), vmin=mn, vmax=mx)
five.title.set_text('t = 5')
six = plt.subplot(186)
plt.imshow(reshaper(instants[5]), vmin=mn, vmax=mx)
six.title.set_text('t = 6')
seven = plt.subplot(187)
plt.imshow(reshaper(instants[6]), vmin=mn, vmax=mx)
seven.title.set_text('t = 7')
eight = plt.subplot(188)
im = plt.imshow(reshaper(instants[7]), vmin=mn, vmax=mx)
eight.title.set_text('t = 8')

cbar_ax = fig.add_axes([0.12, 0.2, 0.785, 0.05])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
plt.show()