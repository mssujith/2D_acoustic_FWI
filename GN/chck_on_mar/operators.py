import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import diags, vstack
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve, eigs


"""
  ================================================================================== FORWARD WAVE PROPAGATION (FDFD) ==============================================================================================================
"""

def forward_solver(F, K, src, Ts, Tr, dx, dz, n_pml, bi, bj):
    
    nz, nx = K.shape
    
    P = np.empty((nz, nx))
    P = np.atleast_3d(P)
    data = []
    
    for k in range(len(F)):
        w = (2 * np.pi * F[k]) 

        z1 = np.zeros(nz+2-n_pml)
        z1 = np.append(z1, np.arange(n_pml-1, -3, -1)) * dz
        x1 = np.arange(-2, n_pml)
        x1 = np.append(x1, np.zeros(nx-2*n_pml))
        x1 = np.append(x1, np.arange(n_pml-1, -3, -1)) * dx

        c_pml_x = np.zeros(nx+4)
        c_pml_z = np.zeros(nz+4)

        c_pml1 = 20

        c_pml_x[:n_pml+2] = c_pml1
        c_pml_x[-n_pml-2:] = c_pml1
        c_pml_z[-n_pml+2:] = c_pml1


        Lx = n_pml * dx
        Lz = n_pml * dz

        gx = 1 + (1j * c_pml_x * np.cos(np.pi/2 * x1/Lx) / w)
        gz = 1 + (1j * c_pml_z * np.cos(np.pi/2 * z1/Lz) / w)
        gz.shape = (nz+4, 1)


        gxi = 1/2 * (gx[1:] + gx[:-1])
        gzj = 1/2 * (gz[1:] + gz[:-1])
        
        C1 = (w**2/K) - (1/(gx[2:-2] * dx**2)) * ((1/24 * 1/24 * (bi[:, :-3]/gxi[:-3] + bi[:, 3:]/gxi[3:])) + (9/8 * 9/8 * (bi[:, 1:-2]/gxi[1:-2] + bi[:, 2:-1]/gxi[2:-1]))) -\
              (1/(gz[2:-2] * dz**2)) * ((1/24 * 1/24 * (bj[:-3, :]/gzj[:-3] + bj[3:, :]/gzj[3:])) + (9/8 * 9/8 * (bj[1:-2, :]/gzj[1:-2] + bj[2:-1, :]/gzj[2:-1])))

        C2 = (1/(gx[2:-2] * dx**2)) * (1/24 * 1/24 * bi[:, :-3]/gxi[:-3])
        C3 = (-1/(gx[2:-2] * dx**2)) * (9/8 * 1/24 * (bi[:, 1:-2]/gxi[1:-2] + bi[:, :-3]/gxi[:-3]))
        C4 = (1/(gx[2:-2] * dx**2)) * ((9/8 * 1/24 * (bi[:, 2:-1]/gxi[2:-1] + bi[:, :-3]/gxi[:-3])) + (9/8 * 9/8 * bi[:, 1:-2]/gxi[1:-2]))
        C5 = (1/(gx[2:-2] * dx**2)) * ((9/8 * 1/24 * (bi[:, 1:-2]/gxi[1:-2] + bi[:, 3:]/gxi[3:])) + (9/8 * 9/8 * bi[:, 2:-1]/gxi[2:-1]))
        C6 = (-1/(gx[2:-2] * dx**2)) * (9/8 * 1/24 * (bi[:, 2:-1]/gxi[2:-1] + bi[:, 3:]/gxi[3:]))
        C7 = (1/(gx[2:-2] * dx**2)) * (1/24 * 1/24 * bi[:, 3:]/gxi[3:])
        C8 = (1/(gz[2:-2] * dz**2)) * (1/24 * 1/24 * bj[:-3, :]/gzj[:-3])
        C9 = (-1/(gz[2:-2] * dz**2)) * (9/8 * 1/24 * (bj[1:-2, :]/gzj[1:-2] + bj[:-3, :]/gzj[:-3]))
        C10 = (1/(gz[2:-2] * dz**2)) * ((9/8 * 1/24 * (bj[2:-1, :]/gzj[2:-1] + bj[:-3, :]/gzj[:-3])) + (9/8 * 9/8 * bj[1:-2, :]/gzj[1:-2]))
        C11 = (1/(gz[2:-2] * dz**2)) * ((9/8 * 1/24 * (bj[1:-2, :]/gzj[1:-2] + bj[3:, :]/gzj[3:])) + (9/8 * 9/8 * bj[2:-1, :]/gzj[2:-1]))
        C12 = (-1/(gz[2:-2] * dz**2)) * (9/8 * 1/24 * (bj[2:-1, :]/gzj[2:-1] + bj[3:, :]/gzj[3:]))
        C13 = (1/(gz[2:-2] * dz**2)) * (1/24 * 1/24 * bj[3:, :]/gzj[3:])

        C1 = C1.flatten()   # (i, j)
        C2 = C2.flatten()   # (i-3, j)
        C3 = C3.flatten()   # (i-2, j)
        C4 = C4.flatten()   # (i-1, j)
        C5 = C5.flatten()   # (i+1, j)
        C6 = C6.flatten()   # (i+2, j)
        C7 = C7.flatten()   # (i+3, j)
        C8 = C8.flatten()   # (i, j-3)
        C9 = C9.flatten()   # (i, j-2)
        C10 = C10.flatten() # (i, j-1)
        C11 = C11.flatten() # (i, j+1)
        C12 = C12.flatten() # (i, j+2)
        C13 = C13.flatten() # (i, 2j+3)


        M = diags([C8[3*nx:], C9[2*nx:], C10[nx:], C2[3:], C3[2:], C4[1:], C1, C5, C6, C7, C11, C12, C13], [-3*nx, -2*nx, -nx, -3, -2, -1, 0, 1, 2, 3, nx, 2*nx, 3*nx], shape = (nx*nz, nx*nz), format = 'csc')


        s = Ts * src[k]
        s = s.flatten()

        # solving the matrix equation

        p1 = spsolve(M, s)
        p2 = p1.copy()
        p2.shape = (nx*nz, 1)
        
        n_rec, temp = Tr.shape

        data1 = np.matmul(Tr, p1)
        data = np.append(data, data1, axis = 0)

        p = p1.reshape(nz, nx)
        P = np.append(P, np.atleast_3d(p), axis = 2)

    P = np.delete(P, 0, 2)
    
    data = np.append(data.real, data.imag, axis = 0)
    data.shape = (n_rec * 2*len(F), 1)
    return data


"""
 =================================================================================== FIRST ORDER DERIVATIVE AND MODELED DATA =====================================================================================================
"""



def forward_and_jacobian(F, V, d, K, src, Ts, Tr, dx, dz, n_pml, bi, bj):
    
    nz, nx = K.shape
    n_par = (nx-2*n_pml) * (nz-n_pml)
    n_rec, temp = Tr.shape
    
    G = np.empty((1, n_par))
    R1 = np.empty((1, n_rec))
    
    V1 = V[:-n_pml, n_pml:-n_pml].flatten()
    d1 = d[:-n_pml, n_pml:-n_pml].flatten()

    pred = []
    
    for k in range(len(F)):
        w = (2 * np.pi * F[k]) 

        z1 = np.zeros(nz+2-n_pml)
        z1 = np.append(z1, np.arange(n_pml-1, -3, -1)) * dz
        x1 = np.arange(-2, n_pml)
        x1 = np.append(x1, np.zeros(nx-2*n_pml))
        x1 = np.append(x1, np.arange(n_pml-1, -3, -1)) * dx

        c_pml_x = np.zeros(nx+4)
        c_pml_z = np.zeros(nz+4)

        c_pml1 = 20

        c_pml_x[:n_pml+2] = c_pml1
        c_pml_x[-n_pml-2:] = c_pml1
        c_pml_z[-n_pml+2:] = c_pml1


        Lx = n_pml * dx
        Lz = n_pml * dz

        gx = 1 + (1j * c_pml_x * np.cos(np.pi/2 * x1/Lx) / w)
        gz = 1 + (1j * c_pml_z * np.cos(np.pi/2 * z1/Lz) / w)
        gz.shape = (nz+4, 1)


        gxi = 1/2 * (gx[1:] + gx[:-1])
        gzj = 1/2 * (gz[1:] + gz[:-1])


        C1 = (w**2/K) - (1/(gx[2:-2] * dx**2)) * ((1/24 * 1/24 * (bi[:, :-3]/gxi[:-3] + bi[:, 3:]/gxi[3:])) + (9/8 * 9/8 * (bi[:, 1:-2]/gxi[1:-2] + bi[:, 2:-1]/gxi[2:-1]))) -\
              (1/(gz[2:-2] * dz**2)) * ((1/24 * 1/24 * (bj[:-3, :]/gzj[:-3] + bj[3:, :]/gzj[3:])) + (9/8 * 9/8 * (bj[1:-2, :]/gzj[1:-2] + bj[2:-1, :]/gzj[2:-1])))

        C2 = (1/(gx[2:-2] * dx**2)) * (1/24 * 1/24 * bi[:, :-3]/gxi[:-3])
        C3 = (-1/(gx[2:-2] * dx**2)) * (9/8 * 1/24 * (bi[:, 1:-2]/gxi[1:-2] + bi[:, :-3]/gxi[:-3]))
        C4 = (1/(gx[2:-2] * dx**2)) * ((9/8 * 1/24 * (bi[:, 2:-1]/gxi[2:-1] + bi[:, :-3]/gxi[:-3])) + (9/8 * 9/8 * bi[:, 1:-2]/gxi[1:-2]))
        C5 = (1/(gx[2:-2] * dx**2)) * ((9/8 * 1/24 * (bi[:, 1:-2]/gxi[1:-2] + bi[:, 3:]/gxi[3:])) + (9/8 * 9/8 * bi[:, 2:-1]/gxi[2:-1]))
        C6 = (-1/(gx[2:-2] * dx**2)) * (9/8 * 1/24 * (bi[:, 2:-1]/gxi[2:-1] + bi[:, 3:]/gxi[3:]))
        C7 = (1/(gx[2:-2] * dx**2)) * (1/24 * 1/24 * bi[:, 3:]/gxi[3:])
        C8 = (1/(gz[2:-2] * dz**2)) * (1/24 * 1/24 * bj[:-3, :]/gzj[:-3])
        C9 = (-1/(gz[2:-2] * dz**2)) * (9/8 * 1/24 * (bj[1:-2, :]/gzj[1:-2] + bj[:-3, :]/gzj[:-3]))
        C10 = (1/(gz[2:-2] * dz**2)) * ((9/8 * 1/24 * (bj[2:-1, :]/gzj[2:-1] + bj[:-3, :]/gzj[:-3])) + (9/8 * 9/8 * bj[1:-2, :]/gzj[1:-2]))
        C11 = (1/(gz[2:-2] * dz**2)) * ((9/8 * 1/24 * (bj[1:-2, :]/gzj[1:-2] + bj[3:, :]/gzj[3:])) + (9/8 * 9/8 * bj[2:-1, :]/gzj[2:-1]))
        C12 = (-1/(gz[2:-2] * dz**2)) * (9/8 * 1/24 * (bj[2:-1, :]/gzj[2:-1] + bj[3:, :]/gzj[3:]))
        C13 = (1/(gz[2:-2] * dz**2)) * (1/24 * 1/24 * bj[3:, :]/gzj[3:])

        C1 = C1.flatten()   # (i, j)
        C2 = C2.flatten()   # (i-3, j)
        C3 = C3.flatten()   # (i-2, j)
        C4 = C4.flatten()   # (i-1, j)
        C5 = C5.flatten()   # (i+1, j)
        C6 = C6.flatten()   # (i+2, j)
        C7 = C7.flatten()   # (i+3, j)
        C8 = C8.flatten()   # (i, j-3)
        C9 = C9.flatten()   # (i, j-2)
        C10 = C10.flatten() # (i, j-1)
        C11 = C11.flatten() # (i, j+1)
        C12 = C12.flatten() # (i, j+2)
        C13 = C13.flatten() # (i, 2j+3)


        M = diags([C8[3*nx:], C9[2*nx:], C10[nx:], C2[3:], C3[2:], C4[1:], C1, C5, C6, C7, C11, C12, C13], [-3*nx, -2*nx, -nx, -3, -2, -1, 0, 1, 2, 3, nx, 2*nx, 3*nx], shape = (nx*nz, nx*nz), format = 'csc')

        
        R = spsolve(M.T, Tr.T)

        R1 = np.append(R1, R, axis= 0)

        s = Ts * src[k]
        s = s.flatten()
        p1 = spsolve(M, s)
        p1.shape = (nx*nz, 1)

        pred1 = np.matmul(Tr, p1)
        pred = np.append(pred, pred1.flatten(), axis = 0)

        dMdk = np.zeros((nx*nz, n_par))

        indx1 = np.arange(nx*nz)
        indx1 = indx1.reshape(nz, nx)
        indx1 = indx1[:-n_pml, n_pml:-n_pml].flatten()
        k1 = 0

        for indx2 in indx1:
            dMdk[indx2, k1] = -2 * w**2 / ((V1[k1]**3) * d1[k1])
            k1 += 1

        G = np.append(G, dMdk * p1, axis = 0)


    G  = np.delete(G, 0, 0)
    R1 = np.delete(R1, 0, 0)

    pred = np.append(pred.real, pred.imag, axis = 0)
    pred.shape = (n_rec * 2*len(F), 1)
    
    nf1 = len(F)
    
    R0 = np.zeros((nf1 * nz * nx, nf1 * n_rec))
    for i1 in range(nf1):
        R0[i1 * nx * nz:(i1+1) * nx * nz, i1 * n_rec:(i1+1) * n_rec] = R1[i1 * nx * nz:(i1+1) * nx * nz, :].real

    R2 = np.zeros((nf1 * nz * nx, nf1 * n_rec))
    for i1 in range(nf1):
        R2[i1 * nx * nz:(i1+1) * nx * nz, i1 * n_rec:(i1+1) * n_rec] = R1[i1 * nx * nz:(i1+1) * nx * nz, :].imag

    R3 = R0 + 1j * R2

    Jt = -np.matmul(G.T, R3)

    J = np.append(Jt.T.real, Jt.T.imag, axis = 0)
    
    return J, pred

"""
 =================================================================================== GAUSS-NEWTON METHOD =========================================================================================================================
"""

def GN_solver(J, dr, K, K_ref, Wd, Wm, lamda, alpha, n_pml):
    
    nz, nx = K.shape
    
    Ki1 = K_ref[:-n_pml, n_pml:-n_pml].flatten()
    Ki1.shape = (Ki1.size, 1)

    Ki2 = K[:-n_pml, n_pml:-n_pml].flatten()
    Ki2.shape = (Ki2.size, 1)
    
    A = np.matmul(J.T, (Wd.T * (Wd * J)))
    B = lamda * (Wm.T * Wm)
    C = np.matmul(J.T, (Wd.T * (Wd * (dr + np.matmul(J, Ki2)))))
    D = lamda * Wm.T * (Wm * Ki1)
    
    E = A + B
    F = C + D
    
    K_new = np.linalg.solve(E, F)
    
    K[:-n_pml, n_pml:-n_pml] = alpha * K_new.reshape(nz-n_pml, nx-2*n_pml) + (1-alpha) * K[:-n_pml, n_pml:-n_pml]
    
    return K


"""
 =================================================================================== DATA & MODEL NORM  =========================================================================================================================
"""


def norm0(dr, Wd, Wm, K, K_ref, n_pml):
    
    dr.shape = (dr.size, 1)
    
    K1 = K[:-n_pml, n_pml:-n_pml].flatten()
    Ki1 = K_ref[:-n_pml, n_pml:-n_pml].flatten()
    K1.shape = (K1.size, 1)
    Ki1.shape = (Ki1.size, 1)
    
    data_norm = np.matmul(dr.T, (Wd.T * (Wd * dr)))
    modl_norm = np.matmul((K1 - Ki1).T, (Wm.T *(Wm * (K1 - Ki1))))
    
    return data_norm, modl_norm


"""
 ========================================================================================== GRADIENT  =============================================================================================================================
"""

def gradient(J, dr, Wd, Wm, K, K_ref, lamda, n_pml):
    
    N_d, N_p = J.shape
    
    K1 = K[:-n_pml, n_pml:-n_pml].flatten()
    Ki1 = K_ref[:-n_pml, n_pml:-n_pml].flatten()
    K1.shape = (K1.size, 1)
    Ki1.shape = (Ki1.size, 1)
    
    grad = - 2/N_d * np.matmul(J.T, (Wd.T * (Wd * dr))) + 2*lamda/N_p * ((K1 - Ki1).T *  (Wm.T * Wm)).T

#     grad = - 2/N_d * np.matmul(J.T, (Wd.T * (Wd * dr))) + 2*lamda/N_p * (Wm.T * (Wm * (K1 - Ki1)))
    
    return grad


# =======================================================================================================================================================================================================================



def src_rec(x_src, z_src, x_rec, z_rec, dx, dz, nx, nz, n_pml):
    
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
                
    n_rec = len(x_rec)
    Tr = np.zeros((n_rec, nz*nx))
    
    for i in range(len(x_rec)):
        ri = int(x_rec[i]//dx) + n_pml
        rj = z_src//dz
        
        Tr1 = np.zeros((nz, nx))

        if x_rec[i]%dx == 0 and z_rec%dz == 0:
            Tr1[rj, ri] = 1
            Tr1[rj+1, ri] = 0
            Tr1[rj, ri+1] = 0
            Tr1[rj+1, ri+1] = 0

        if x_rec[i]%dx == 0 and z_rec%dz != 0:
            Tr1[rj, ri] = (dz-z_rec%dz)/dz
            Tr1[rj+1, ri] = (z_rec%dz)/dz
            Tr1[rj, ri+1] = 0
            Tr1[rj+1, ri+1] = 0

        if x_rec[i]%dx != 0 and z_rec%dz == 0:
            Tr1[rj, ri] = (dx-x_rec[i]%dx)/dx
            Tr1[rj+1, ri] = 0
            Tr1[rj, ri+1] = (x_rec[i]%dx)/dx
            Tr1[rj+1, ri+1] = 0

        if x_rec[i]%dx != 0 and z_rec%dz != 0:
            Tr1[rj, ri] =  ((dx - x_rec[i]%dx)/dx + (dz - z_rec%dz)/dz)/2
            Tr1[rj, ri+1] = ((x_rec[i]%dx)/dx + (dz - z_rec%dz)/dz)/2
            Tr1[rj+1, ri] =  ((dx - x_rec[i]%dx)/dx + (z_rec%dz)/dz)/2
            Tr1[rj+1, ri+1] =  ((x_rec[i]%dx)/dx + (z_rec%dz)/dz)/2
            
        Tr2 = Tr1.flatten()
        Tr2.shape = (1, Tr2.size)
        
        Tr[i, :] = Tr2
                
    return Ts, Tr




