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

data = np.loadtxt('./marmousi.dat')
data = np.flip(data, axis = [0])

data = data[:80, 100:300]

data1 = data.copy()
data2 = data.copy()

nz, nx = np.shape(data)

z_off, x_off = data1.shape
z_cell = 13  #37
x_cell = 13  #45
checker = np.ones((data1.shape))

for i1 in range(0, z_off - 2, z_cell+1):
  for i2 in range(0, x_off - 2, x_cell+1):
    if (i1 + i2)%4 == 0:
      checker[i1:i1+z_cell, i2:i2+x_cell] = -1

data = data1.copy()

data += checker * 200


dz, dx = 10, 100	

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
# vii = v.copy()

vi[:-n_pml, n_pml:-n_pml] = data1
# vii[:-n_pml, n_pml:-n_pml] = data2


# ============================================================================================

d = np.power(v, .25) * 310

d1 = np.ones((nz+4, nx+4))
d1[:-4, 2:-2] = d

d2 = d1[-5, 2:-2]
d2.shape = (1, d2.size)

d1[-4:, 2:-2] = np.repeat(d2, 4, axis = 0)
d1[:, :2] = np.repeat(np.vstack(d1[:, 3]), 2, axis = 1)
d1[:, -2:] = np.repeat(np.vstack(d1[:, -3]), 2, axis = 1)

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
F = np.array([.1, .2, .4, .9, 2, 4.5])
nf = len(F)
src = (2/np.sqrt(np.pi))  *  (F**2/Fp**3)  *  np.exp(-((F**2/Fp**2)))

nf1 = len(F)

# ==========================================================================================

z_src =  5
x_src = np.arange(50, 20000, 50)

x_rec = np.arange(12, 20000, 12)
z_rec = 10
n_rec = len(x_rec)

Ts, Tr = src_rec(x_src, z_src, x_rec, z_rec, dx, dz, nx, nz, n_pml)

# ===============================================================================================


data = forward_solver(F, K, src, Ts, Tr, dx, dz, n_pml, bi, bj)


# ==============================================================================================

n_par = (nx-2*n_pml) * (nz-n_pml)

Ki = np.power(vi, 2) * d
# Kii = vii**2 * d * 0		# K.copy()

vii = vi.copy() 
v69 = vi.copy()

"""
K_bound = np.power(v, 2) * d
kmax = np.max(K_bound.flatten())

K_bound = np.power(v, 2) * d
kmin = np.min(K_bound.flatten())
"""

vmax, vmin = v.max()+500, v.min()-500

res = []

alpha = 1

a = -.0
Wm = diags([a, a, 1, a, a], [-nx, -1, 0, 1, nx], shape = (n_par, n_par), format = 'csc')
Wd = diags([100/abs(np.repeat(data, 2*nf1, 0)).T], [0], shape = (2*nf1*n_rec, 2*nf1*n_rec), format = 'csc')

it = 0



while it <= 10:
    
    J, pred = forward_and_jacobian(F, vi, d,  Ki, src, Ts, Tr, dx, dz, n_pml, bi, bj)
    
    if it == 0:
        pred0 = pred.copy()
    
    dr = data - pred
    res.append((np.matmul(dr.T, dr) / len(dr))[0][0])
    
    lamda = np.max(np.sum(abs(np.matmul(J.T, (Wd.T * (Wd * J)))), axis=1)) / (6**it)
    
    if it > 0 and res[-1] > res[-2]:
        lamda = lamda ** 10*20
    
    v_new = GN_solver(J, dr, vi, vii, Wd, Wm, lamda, alpha, n_pml)
    
    for i1 in range(nz):
        for i2 in range(nx):
            if v_new[i1, i2] < vmin:
                v_new[i1, i2] = vmin
            if v_new[i1, i2] > vmax:
                v_new[i1, i2] = vmax
                
    vi = v_new
    Ki = np.power(vi, 2) * d
    it += 1



# ==============================================================================================================


V_init = v69
V_upd = vi
V_tru = v

figsize=(18, 3)
extent = [0, (nx-2*n_pml)*dx, (nz-n_pml)*dz, 0]
vmax, vmin = v.max(), v.min()


plot_models(V_init, V_upd, V_tru, n_pml, extent, vmax, vmin, figsize, aspect = 'auto', cmap = 'Spectral')


# ===============================================================================================================

xline, zline = 80, 20
X , Z= np.arange(nx - 2*n_pml)*dx, np.arange(nz-n_pml)*dz

# ==============================================================================================================

plot_profile(V_init, V_upd, V_tru, n_pml, X, Z, xline, zline)

# ===============================================================================================================

plt.figure()
plt.plot(res, color = 'red')
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
