
import time as time
import math

import numpy as np
import matplotlib.pyplot as plt 
import scipy.linalg as sclin

from numba import njit
from numba.typed import Dict
from numba.core import types

from pocs.File_manage import read_write as rw 

from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

plot_dir    = "plots/"
data_dir    = "data/" 
# ## Chemical Master equation 
# P(m,p) = 
# +alphaR( P(m-1,p) - P(m,p) ) 
# +alphaP( P(m,p-1) - P(m,p) )
# +dR[ (m+1)P(m+1,p) - mP(m,p) ]
# +dP[ (p+1)P(m,p+1) - pP(m,p) ]
# 
# 
# 

@njit 
def delta(a, b): 
    if a == b:
        return 1 
    else : 
        return 0 
    #
# 


@njit
def element_fun(i, j, pmax, mmax, alpha, beta, gamma):
    
    m = i // (pmax + 1)
    p = i % (pmax + 1)
    
    mp = j // (pmax + 1)
    pp = j % (pmax + 1)
    
    Mij =   alpha * (      delta(m-1, mp)*delta(p, pp) - delta(m, mp)*delta(p, pp) ) 
    Mij +=  beta * gamma * m * (  delta(m, mp)*delta(p-1, pp) - delta(m, mp)*delta(p, pp) )
    Mij +=  gamma * (  (m+1)*delta(m+1, mp)*delta(p, pp) - m*delta(m,mp)*delta(p,pp) )    
    Mij +=  (p+1)*delta(m, mp)*delta(p+1, pp) - p*delta(m,mp)*delta(p,pp)


    return Mij 

@njit
def fill_matrix_FB(mat_len, pmax, mmax, mat, alpha, beta, gamma):
    for i in range(mat_len): 
        for j in range(mat_len):
            mat[i][j] = element_fun(i, j, pmax, mmax, alpha, beta, gamma)
        #
    #
    
    return mat 
#

@njit 
def matrixNormalization(mat_len, mat):
    for i in range(mat_len):
        mat[i][i] = mat[i][i] - sum(mat[:,i]) 
    #
    return mat
#

from functools import lru_cache
import numpy.linalg as nplin

def get_coef_vec(eig_vec, init_vec):
    X = nplin.inv(eig_vec).dot(init_vec)
    return X 
#

class Transition_Matrix():
    def __init__(self, t_mat):
        self.t_mat = t_mat
        return
    
    def get_coef_vec(self, init_vec):
        X = nplin.inv(self.eig_vec).dot(init_vec)
        return X 
    #
    
    @lru_cache(5)
    def update_eigens(self):
        self.eig_val, self.eig_vec = nplin.eig(self.t_mat)
    #
    
    def get_eigens(self): 
        return nplin.eig(self.t_mat)
    #
    
    def get_state_t(self, t, init_vec):
        self.update_eigens()
        coef_vec = self.get_coef_vec(init_vec)
        val = 0
        for idx, ci in enumerate(coef_vec): 
            val += ci * self.eig_vec[:,idx] * np.exp( self.eig_val[idx]*t )
        #
        return val 
    # 
# 

img = [] # some array of images
frames = [] # for storing the generated images
fig = plt.figure()

para_list = [20, 2.5, 10]

pmax = 199
mmax = 9
mat_len = (pmax+1) * (mmax + 1)
M = np.zeros((mat_len, mat_len), dtype=float)
I = np.zeros((mat_len, mat_len), dtype=float)
for i in range(mat_len):
    I[i][i] = 1 
# 


M = fill_matrix_FB(mat_len, pmax, mmax, M, *para_list)
M = matrixNormalization(mat_len, M)

mat = Transition_Matrix(M)

prob_eq = np.zeros(( mmax+1, pmax+1))

def update(frame):
    fig.clear()
    t = frame 
    print(t)
    max_vec = mat.get_state_t(t, [0.5,0.5] + [0.0 for _ in range(mat_len - 2)])
    
    vec_norm = sum(max_vec)
    for i in range(len(max_vec)): 
        m = i // (pmax + 1)
        p = i % (pmax + 1)
        
        prob_eq[m,p] = max_vec[i]/vec_norm
    # 
    plt.xlabel("A")
    plt.ylabel("B")
    
    
    plt.title(f"t= {round(frame,4)}, \n alpha {para_list[0]} beta {para_list[1]} gamma {para_list[2]}")
    plt.xlabel("Protein #")
    plt.ylabel("mRNA #")
    plt.imshow(prob_eq, aspect=10, origin="lower")
    plt.colorbar()
    
writer=FFMpegWriter(bitrate=500)
ani = FuncAnimation(fig, update, frames = np.linspace(0,1,81))
ani.save("Solution_slow.avi", dpi=300, writer=writer)

