import numpy as np
from matplotlib import pyplot as plt

# b
k = 4
h = 0.2/(2**k)
grid = np.arange(0,1.01,h)



n = len(grid) - 1

L = np.zeros((n-1,n-1))
np.fill_diagonal(L,2)
np.fill_diagonal(L[:,1:], -1)
np.fill_diagonal(L[1:,:], -1)

L = L/h**2
f1 = np.ones((n-1)).T
f2 = np.exp(grid[1:-1]).T

f1[0] += 1/h**2
f1[-1] += 2/h**2

f2[0] += 1/h**2
f2[-1] += 2/h**2

print('L = ', L, 'f1 = ', f1)
print('L = ', L, 'f2 = ', f2)

# c
u1 = np.linalg.solve(L,f1)
u1 = np.insert(u1,0,1,axis=0)
u1 = np.insert(u1,n,2,axis=0)

u2 = np.linalg.solve(L,f2)
u2 = np.insert(u2,0,1,axis=0)
u2 = np.insert(u2,n,2,axis=0)


u1exact = lambda x : -0.5*x**2 + 1.5*x + 1
u1exactgrid = u1exact(grid)
u2exact = lambda x : -np.exp(x) + np.exp(1)*x + 2
u2exactgrid = u2exact(grid)

x = np.arange(0,1.01,0.01)
u1exactx = u1exact(x)
u2exactx = u2exact(x)

fig1 = plt.figure(1)
fig1.suptitle('u1')
plt.plot(grid,u1,label='numerical')
plt.plot(x,u1exactx,label='exact')
plt.legend()

fig2 = plt.figure(2)
fig2.suptitle('u2')
plt.plot(grid,u2,label='numerical')
plt.plot(x,u2exactx,label='exact')
plt.legend()

globalerror1 = np.linalg.norm(u1exactgrid-u1)*np.sqrt(h)
globalerror2 = np.linalg.norm(u2exactgrid-u2)*np.sqrt(h)

print('global error 1 =', globalerror1)
print('global error 2 =', globalerror2)

plt.show()
