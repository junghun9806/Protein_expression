from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt 

from numba import njit

from .pocs.File_manage import read_write as rw 
from . import general_func as gf 

@njit
def element_fun(i, j, pmax, mmax, alpha, beta, gamma):
    
    m = i // (pmax + 1)
    p = i % (pmax + 1)
    
    mp = j // (pmax + 1)
    pp = j % (pmax + 1)
    
    Mij =   alpha * (      gf.delta(m-1, mp)*gf.delta(p, pp) - gf.delta(m, mp)*gf.delta(p, pp) ) 
    Mij +=  beta * gamma * m * (  gf.delta(m, mp)*gf.delta(p-1, pp) - gf.delta(m, mp)*gf.delta(p, pp) )
    Mij +=  gamma * (  (m+1)*gf.delta(m+1, mp)*gf.delta(p, pp) - m*gf.delta(m,mp)*gf.delta(p,pp) )    
    Mij +=  (p+1)*gf.delta(m, mp)*gf.delta(p+1, pp) - p*gf.delta(m,mp)*gf.delta(p,pp)


    return Mij 
# 

@njit
def fill_matrix_FB(mat_len, pmax, mmax, mat, alpha, beta, gamma):
    for i in range(mat_len): 
        for j in range(mat_len):
            mat[i][j] = element_fun(i, j, pmax, mmax, alpha, beta, gamma)
        #
    #  
    return mat 
#

def solve_FB(pmax, mmax, para_list, file_name , data_dir= "data/"): 
    """
    Solve with Finite Buffer method with given pmax, mmax, parameter list. 
    returns eigenvalues, eigenvectors.
    """
    mat_len = (pmax+1) * (mmax + 1) 
    rw.make_dir(data_dir)
    
    M = np.zeros((mat_len, mat_len), dtype=float)
    
    M = fill_matrix_FB(mat_len, pmax, mmax, M, *para_list)
    M = gf.matrixNormalization(mat_len, M)
    
    eig_val, eig_vec = np.linalg.eig(M)
    
    if file_name: 
        rw.pickle_dump(data_dir, file_name + "_eig.bin", eig_val[0])
        rw.pickle_dump(data_dir, file_name + "_vec.bin", eig_val[1])
    #
    
    return eig_val, eig_vec
#

def generate_P_matrix_FB(pmax, mmax, eigen_rank, eig_val, eig_vec):
    arg_sort_eig = eig_val.argsort()
    arg_sort_eig = np.flip(arg_sort_eig)
    
    prob_eq = np.zeros(( mmax+1, pmax+1))
    
    vector_index = arg_sort_eig[eigen_rank]
    max_vec = eig_vec[:,vector_index]
    eig_value = eig_val[vector_index]
    
    vec_norm = sum(max_vec)
    
    for i in range(len(max_vec)): 
        m = i // (pmax + 1)
        p = i % (pmax + 1)
        
        prob_eq[m,p] = max_vec[i]/vec_norm
    #
    return prob_eq, eig_value
#

def draw_P_hmap_FB(Pmap, eig_value, title, para_list, plot_dir= "plots/"):
    rw.make_dir(plot_dir)
    
    plt.title(f"Lambda = {-round(eig_value,4)}, alpha {para_list[0]} beta {para_list[1]} gamma {para_list[2]}")
    plt.xlabel("Protein #")
    plt.ylabel("mRNA #")
    plt.imshow(Pmap, aspect=10, origin="lower")
    plt.colorbar()
    plt.savefig(plot_dir + title + ".png", dpi = 300)
    plt.clf()
    
    plt.plot(sum(Pmap.T), marker="o")
    plt.grid()
    plt.title(f"mRNA Lambda = {-round(eig_value,4)}, \n alpha {para_list[0]} beta {para_list[1]} gamma {para_list[2]}")
    plt.savefig(plot_dir + title + "mRNA.png", dpi = 300)
    plt.clf()
    
    plt.plot(sum(Pmap), marker="o")
    plt.grid()
    plt.title(f"Protien Lambda = {-round(eig_value,4)}, \n alpha {para_list[0]} beta {para_list[1]} gamma {para_list[2]}")
    plt.savefig(plot_dir + title + "Protein.png", dpi = 300)
    plt.clf()
    
    return 
#

def do_all_FB(pmax, mmax, para_list, max_rank, plot_dir, data_dir): 
    rw.make_dir(plot_dir)
    rw.make_dir(data_dir)
    file_prefix = f"FB_p_{pmax}_m_{mmax}_para_{para_list[0]}_{para_list[1]}_{para_list[2]}"
    print(plot_dir)
    eig_val_list, eig_vec_list = solve_FB(pmax, mmax, para_list, file_prefix, data_dir=data_dir)

    for rank in range(max_rank):
        if (not isinstance(eig_val_list, Iterable)) and rank != 0:
            continue
        #
        if rank >= len(eig_val_list) :
            continue
        # 
        
        prob_eq, eig_val = generate_P_matrix_FB(pmax, mmax, rank, eig_val_list, eig_vec_list) 
        title = file_prefix + f"_rank_{rank}"
        draw_P_hmap_FB(prob_eq, eig_val, title, para_list, plot_dir=plot_dir)
    #
    return eig_val_list, eig_vec_list
# 



if __name__ == "main": 
    plot_dir    = "plots/"
    data_dir    = "data/" 
    
    para_list = [20, 2.5, 10]

    pmax = 199
    mmax = 9
    max_rank = 5
    
    do_all_FB(pmax, mmax, para_list, max_rank, plot_dir, data_dir)

#