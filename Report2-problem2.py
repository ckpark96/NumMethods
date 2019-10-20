# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:23:17 2019

@author: changkyupark
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
import time


alpha = -5
h = 0.08
dt = 0.003

def u(x,y): #t=0
    return np.exp(alpha*(x-2)**2+alpha*(y-2)**2)

Nx = int(4/h)
Ny = int(4/h)

Dx = np.zeros((Nx, Nx-1))
np.fill_diagonal(Dx, 1)
np.fill_diagonal(Dx[1:,:], -1)

Dy = np.zeros((Ny, Ny-1))
np.fill_diagonal(Dy, 1)
np.fill_diagonal(Dy[1:,:], -1)

Lxx = Dx.T@Dx
Lyy = Dy.T@Dy
A = -1*sp.kronsum(Lxx, Lyy)/h**2

grid = np.mgrid[h:4:h,h:4:h]
x = grid[0]
y = grid[1]

u = u(x,y)
u = np.reshape(u, ((Nx-1)*(Ny-1)), order='F')

def forward(u0, A, dt):
    return (sp.identity(A.shape[0]) + dt*A)@u0


def backward(u0, A, dt):
    return la.spsolve(sp.identity(A.shape[0]) - dt*A, u0)

    
timeinst = [0, 0.045, 0.09, 0.15]

def time_evolution(u0, A, dt, tmax, timeinstants, algo, stabilizer=None):
    if algo == 'FE':
        method = forward
    if algo == 'BE':
        method = backward
    if stabilizer is not None and algo == "FE":
        dt = stabilizer
    
    t = 0
    instants = [u0]
    uh = u0
    
    while t <= tmax:
        t += dt
        uh = method(uh, A, dt)
        for i in timeinstants:
            if abs(i-t) <= dt/2:
                instants.append(uh)
    return instants

start = time.time()
FE = time_evolution(u, A, dt, 0.15, timeinst, 'FE', stabilizer = h**2/4)
print('forward time =', time.time()-start)

start = time.time()
BE = time_evolution(u, A, dt, 0.15, timeinst, 'BE')
print('backward time =', time.time()-start)
#reshaper = lambda u: np.reshape(u,(Ny-1,Nx-1)).T
reshaper = lambda u: np.reshape(u, [grid[0].shape[0], grid[0].shape[1]])[::-1,:]

mx = max(np.max(np.array(FE)), np.max(np.array(BE)))
mn = max(np.min(np.array(FE)), np.min(np.array(BE)))

fig = plt.figure()
fig.suptitle(dt, fontsize=10)

plt.subplot(241)
plt.imshow(reshaper(FE[0]), vmin=mn, vmax=mx)
plt.subplot(242)
plt.imshow(reshaper(FE[1]), vmin=mn, vmax=mx)
plt.subplot(243)
plt.imshow(reshaper(FE[2]), vmin=mn, vmax=mx)
plt.subplot(244)
plt.imshow(reshaper(FE[3]), vmin=mn, vmax=mx)

plt.subplot(245)
plt.imshow(reshaper(BE[0]), vmin=mn, vmax=mx)
plt.subplot(246)
plt.imshow(reshaper(BE[1]), vmin=mn, vmax=mx)
plt.subplot(247)
plt.imshow(reshaper(BE[2]), vmin=mn, vmax=mx)
plt.subplot(248)
plt.imshow(reshaper(BE[3]), vmin=mn, vmax=mx)

plt.show()

#try slight above stable t and slight below