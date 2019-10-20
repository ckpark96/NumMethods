import numpy as np
import matplotlib.pyplot as plt

# a
grid = np.arange(0,1.1,0.1)
h = 0.1

# b
grid = np.linspace(0,1.0,11)


# d
n = len(grid) - 1
L = np.zeros((n-1,n-1))
np.fill_diagonal(L,2)
np.fill_diagonal(L[:,1:], -1)
np.fill_diagonal(L[1:,:],-1)
L = L/(h**2)

row0 = L[0,:]
row1 = L[1,:]
rowlast = L[-1,:]
print(row0)
print(row1)
print(rowlast)

plt.figure(1)
plt.spy(L, marker = 'o' , color = 'green' )

# e
k = np.arange(1,n)
D = h*n
oplambda = (np.pi*k/D)**2
matlambda = 4/h**2*(np.sin(k*np.pi*h/2))**2
numlambda = np.linalg.eig(L)[0]

print('operator L =', oplambda)
print('matrix L = ', matlambda)
print('numerical matrix L =', numlambda)

y = np.zeros((n-1))

plt.figure(2)
plt.plot(oplambda,y,marker='o',label='operator L eigenvalue')
plt.plot(matlambda,y,marker='^',label='matrix L eigenvalue')
plt.plot(numlambda,y,marker='+',label='numerical matrix L eigenvalue')
plt.legend()

numv = np.linalg.eig(L)[1]
zeros = np.zeros((n-1))
numvBC = np.vstack((zeros,numv))
numvBC = np.vstack((numvBC,zeros))



opv = np.zeros((n-1,n))
for j in range(1,n):
    for k in range(1,n+1):
        opv[j-1,k-1] = np.sin(k*np.pi*j*h)

opvBC = np.insert(opv,0,0,axis=0)
opvBC = np.insert(opvBC,n,0,axis=0)

X = np.arange(0,1.0,0.001)

plt.figure(3)
for i in range(1,n):
    opeigenfn = np.sin(i*np.pi*X)
    plt.plot(X,opeigenfn)
    plt.plot(grid,opvBC[:,i-1],"+")

plt.figure(4)
opeigenfn = np.sin((n)*np.pi*X)
plt.plot(X,opeigenfn)
plt.plot(grid,opvBC[:,-1],"+")

plt.show()

