#!/bin/bash

import time
t1 = time.time()

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from scipy.sparse import diags, vstack
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve, eigs

from operators import *
from plots import *

# ===============================================================================================

mod1 = np.zeros((80, 80))


for i in range(80):
    for j in range(80):
#         mod1[i, j] = 10000 * np.random.uniform(low = .2, high = .7, size = (1))[-1]

          if i- 30 <= .3*j:
              mod1[i, j] = 1500
          if i- 20 > .3*j:
              mod1[i, j] = 1550

          if i- 60 > -.3*j:
              mod1[i,j] = 1600
          if i-40 < -.7*j:
              mod1[i,j] = 1450

# n1 = 3
# for i1 in range(0, 40, n1):
#     for i2 in range(0, 40, n1):
#         mod1[i1:i1+n1, i2:i2+n1] = np.mean(mod1[i1:i1+n1, i2:i2+n1])


data1 = mod1.copy()

mod1[60:70, 50:55] = 1400
mod1[10:20, 10:20] = 1650

data = mod1

nz, nx = np.shape(data)


dz, dx = 20, 40

n_pml = 10


nz, nx = nz + n_pml, nx + 2*n_pml

z = np.arange(nz) * dz
x = np.arange(nx) * dx

vel = np.zeros((nz, nx))
vel[:-n_pml, n_pml:-n_pml] = data

vel[:-n_pml, :n_pml] = np.repeat(np.vstack(data[:, 0]), n_pml, axis = 1)
vel[:-n_pml, -n_pml:] = np.repeat(np.vstack(data[:, -1]), n_pml, axis = 1)

vel1 = vel[-n_pml-1, :]
vel1.shape = (1, vel1.size)

vel[-n_pml:, :] = np.repeat(vel1, n_pml, axis = 0)

v = vel.copy()

vi = v.copy()
vi[:-n_pml, n_pml:-n_pml] = data1

# ============================================================================================

d = np.ones((nz, nx)) * 2500

d1 = np.ones((nz+4, nx+4)) * 2500

b = 1/d1 

bi = np.zeros((nz+4, nx+3))
bj = np.zeros((nz+3, nx+4))

for i in range(nz+3):
    bj[i, :] = 1/2 * (b[i+1, :] + b[i, :])

for i in range(nx+3):
    bi[:, i] = 1/2 * (b[:, i+1] + b[:, i])

bj = bj[:, 2:-2]
bi = bi[2:-2, :]

# ===========================================================================================

K = np.power(v, 2) * d

# ==========================================================================================

Fp = 2
F = np.array([.1, .14, .2, .28, .4, .56, .79, 1.12, 1.6, 2.3, 3.25, 4.6, 6.5])
nf = len(F)
src = (2/np.sqrt(np.pi))  *  (F**2/Fp**3)  *  np.exp(-((F**2/Fp**2)))

nf1 = len(F)

# ==========================================================================================

z_src =  5
x_src = np.arange(25, 3200, 25)

Ts = np.zeros((nz, nx))


for i in range(len(x_src)):
        si = x_src[i]//dx + n_pml
        sj = z_src//dz

        if x_src[i]%dx == 0 and z_src%dz == 0:
          Ts[sj, si] = 1
          Ts[sj+1, si] = 0
          Ts[sj, si+1] = 0
          Ts[sj+1, si+1] = 0

        if x_src[i]%dx == 0 and z_src%dz != 0:
          Ts[sj, si] = (dz-z_src%dz)/dz
          Ts[sj+1, si] = (z_src%dz)/dz
          Ts[sj, si+1] = 0
          Ts[sj+1, si+1] = 0

        if x_src[i]%dx != 0 and z_src%dz == 0:
          Ts[sj, si] = (dx-x_src[i]%dx)/dx
          Ts[sj+1, si] = 0
          Ts[sj, si+1] = (x_src[i]%dx)/dx
          Ts[sj+1, si+1] = 0

        if x_src[i]%dx != 0 and z_src%dz != 0:
          Ts[sj, si] =  ((dx - x_src[i]%dx)/dx + (dz - z_src%dz)/dz)/2
          Ts[sj, si+1] = ((x_src[i]%dx)/dx + (dz - z_src%dz)/dz)/2
          Ts[sj+1, si] =  ((dx - x_src[i]%dx)/dx + (z_src%dz)/dz)/2
          Ts[sj+1, si+1] =  ((x_src[i]%dx)/dx + (z_src%dz)/dz)/2



z_rec = 10

rj = z_rec//dz

n_rec = 2 * (nx - 2*n_pml)

Tr = np.zeros((n_rec, nz*nx))


for i in range(nx-2*n_pml):
  Tr[2*i, i+n_pml] = 1
  Tr[2*i+1, i+n_pml] = .5
  Tr[2*i+1, i+n_pml+1] = .5


# ===============================================================================================


data = forward_solver(F, K, src, Ts, Tr, dx, dz, n_pml, bi, bj)


# ==============================================================================================

n_par = (nx-2*n_pml) * (nz-n_pml)

Ki = vi**2 * d
Kii = Ki.copy()

K_bound = np.power(v, 2) * d
kmax = np.max(K_bound.flatten())

K_bound = np.power(v, 2) * d
kmin = np.min(K_bound.flatten())

res = []

alpha = .8

a = -.0
Wm = diags([a, a, 1, a, a], [-nx, -1, 0, 1, nx], shape = (n_par, n_par), format = 'csc')
Wd = diags([100/abs(np.repeat(data, 2*nf1, 0)).T], [0], shape = (2*nf1*n_rec, 2*nf1*n_rec), format = 'csc')

it = 0



while it <= 9:
    
    J, pred = forward_and_jacobian(F, Ki, Kii, src, Ts, Tr, dx, dz, n_pml, bi, bj)
    
    if it == 0:
        pred0 = pred.copy()
    
    dr = data - pred
    res.append((np.matmul(dr.T, dr) / len(dr))[0][0])
    
    lamda = np.max(np.sum((np.matmul(J.T, (Wd.T * (Wd * J)))), axis=1)) / (it+1) * 10**-6
    
    if it > 0 and res[-1] > res[-2]:
        lamda = lamda ** 10*20
    
    K_new = GN_solver(J, dr, Ki, Kii, Wd, Wm, lamda, alpha, n_pml)
    
    for i1 in range(nz):
        for i2 in range(nx):
            if K_new[i1, i2] < kmin:
                K_new[i1, i2] = kmin
            if K_new[i1, i2] > kmax:
                K_new[i1, i2] = kmax
                
    Ki = K_new
    it += 1



# ==============================================================================================================


V_init = vi
V_upd = np.sqrt(Ki/d)
V_tru = v

figsize=(18, 4)
extent = [0, (nx-2*n_pml)*dx, (nz-n_pml)*dz, 0]

vmax = 1650
vmin = 1400

plot_models(V_init, V_upd, V_tru, n_pml, extent, vmax, vmin, figsize, cmap = 'Spectral')


# ===============================================================================================================

xline, zline = 20, 65
X , Z= np.arange(nx - 2*n_pml)*dx, np.arange(nz-n_pml)*dz


plot_profile(V_init, V_upd, V_tru, n_pml, X, Z, xline, zline)

# ===============================================================================================================

plt.figure()
plt.plot(res, 'r^-')
plt.xlabel('iteration')
plt.ylabel('residual')
plt.savefig('./results/residual.png')

# ===============================================================================================================

plot_error(data, pred0, pred)

# ==============================================================================================================

print('')
print('')
print('')
print('')
print('')

print('*'*50+' E N D    O F    I N V E R S I O N '+'*'*50)
print('-'*115)

print(f'Grid size            : {nz - n_pml} x {nx - 2*n_pml}')
print(f'Grid discretization  : dz = {dz}, dx = {dx}')
print(f'PML boundary layers  : {n_pml}')
print(f'Frequencies used     : {F} Hz')

print('-'*115)

print(f'Total iterarions     : {it - 1}')
print(f'Initial residual     : {res[1]}')
print(f'Minimised residual   : {res[-1]}')

print('-'*115)

t2 = time.time()
print(f'The inversion was completed in {(int(t2-t1)//60)} min, {int((t2-t1)%60)} sec')
