from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt

from numba import njit

from .pocs.File_manage import read_write as rw 
from . import general_func as gf 
from . import fb as fb 

@njit
def fill_matrix_FSP_Diag_M1(mat_len, pmax, mmax, mat, alpha, beta, gamma):
    # matrix elements for Finte Buffer model
    for i in range(mat_len-1): 
        for j in range(mat_len-1):
            mat[i][j] = fb.element_fun(i, j, pmax, mmax, alpha, beta, gamma)
        #
    #
    
    # terms for every boundary states influx from g. 
    j = mat_len - 1 
    for i in range(mat_len-1): 
        m = i // (pmax + 1)
        p = j % (pmax + 1)
        
        rate = 0 
        if m == mmax: 
            rate += gamma * float(m + 1) 
        if p == pmax : 
            rate += float(p + 1) 
        # 
        
        # Here should not be assignment
        # simply addition to original Finte Buffer Model
        mat[i][j] += rate 
    
    
    # terms for g influx
    i = mat_len - 1
    for j in range(mat_len - 1):
        mp = j // (pmax + 1)
        pp = j % (pmax + 1)
        
        rate = 0
        if mp == mmax : 
            rate += alpha 
        if pp == pmax :
            rate += mp*beta*gamma  
        # 
        
        mat[i][j] = rate 
    
    return mat 
#

def solve_FSP_diag_M1(pmax, mmax, para_list, file_name, data_dir= "data/"): 
    """
    Solve with Finite Buffer method with given pmax, mmax, parameter list. 
    returns eigenvalues, eigenvectors.
    """
    
    mat_len = (pmax+1) * (mmax + 1) + 1
    rw.make_dir(data_dir)
    
    M = np.zeros((mat_len, mat_len), dtype=float)
    
    M = fill_matrix_FSP_Diag_M1(mat_len, pmax, mmax, M, *para_list)
    M = gf.matrixNormalization(mat_len, M)
    
    eig_val, eig_vec = np.linalg.eig(M)
    
    if file_name: 
        rw.pickle_dump(data_dir, file_name + "_eig.bin", eig_val[0])
        rw.pickle_dump(data_dir, file_name + "_vec.bin", eig_val[1])
    #
    
    return eig_val, eig_vec
#

def generate_P_matrix_FSP_M1(pmax, mmax, eigen_rank, eig_val, eig_vec):
    arg_sort_eig = eig_val.argsort()
    arg_sort_eig = np.flip(arg_sort_eig)
    
    prob_eq = np.zeros((mmax+1, pmax+1))
    prob_g  = np.zeros((1, 1))
    
    vector_index = arg_sort_eig[eigen_rank]
    max_vec = eig_vec[:,vector_index]
    eig_value = eig_val[vector_index]
    
    vec_norm = sum(max_vec)
    
    for i in range(len(max_vec)-1): 
        m = i // (pmax + 1)
        p = i % (pmax + 1)
        
        prob_eq[m,p] = max_vec[i]/vec_norm
    #
    
    prob_g[0][0] = max_vec[-1]/vec_norm
    
    return prob_eq, prob_g, eig_value
#

def draw_P_hmap_FSP_M1(Pmap, Gmap, eig_value, title, para_list, plot_dir= "plots/"):
    rw.make_dir(plot_dir)
    
    min_p = np.min([np.min(Pmap)])
    max_p = np.max([np.max(Pmap)])
    
    fig, axes = plt.subplots(1,2, figsize=(16,9), gridspec_kw={"width_ratios" : [8,1]})
    fig.tight_layout()
    ax_prob = axes[0]
    ax_prob.set_title(f"Lambda = {-round(eig_value,4)}, alpha {para_list[0]} beta {para_list[1]} gamma {para_list[2]}")
    ax_prob.set_xlabel("Protein #")
    ax_prob.set_ylabel("mRNA #")
    im_prob = ax_prob.imshow(Pmap, aspect=10, origin="lower", vmin = min_p, vmax=max_p)
    plt.colorbar(im_prob, ax=ax_prob)
    
    Gp = "{:.2e}".format(Gmap[0][0])
    
    ax_g = axes[1]
    ax_g.set_title(f"probability \n g (sink state)  \n {Gp}")
    im_g = ax_g.imshow(Gmap, origin="lower")
    plt.colorbar(im_g, ax=ax_g)
    
    plt.savefig(plot_dir + title + ".png", dpi = 300)
    plt.clf()
    
    plt.plot(sum(Pmap.T), marker="o")
    plt.grid()
    plt.title(f"mRNA Lambda = {-round(eig_value,4)}, \n alpha {para_list[0]} beta {para_list[1]} gamma {para_list[2]}")
    plt.savefig(plot_dir + title + "_mRNA.png", dpi = 300)
    plt.clf()
    
    plt.plot(sum(Pmap), marker="o")
    plt.grid()
    plt.title(f"Protien Lambda = {-round(eig_value,4)}, \n alpha {para_list[0]} beta {para_list[1]} gamma {para_list[2]}")
    plt.savefig(plot_dir + title + "_Protein.png", dpi = 300)
    plt.clf()
    
    return 
#

def do_all_FSP_M1(pmax, mmax, para_list, max_rank, plot_dir, data_dir): 
    rw.make_dir(plot_dir)
    rw.make_dir(data_dir)
    
    file_prefix = f"FSP_Diag_M1_p_{pmax}_m_{mmax}_para_{para_list[0]}_{para_list[1]}_{para_list[2]}"
    
    eig_val_list, eig_vec_list = solve_FSP_diag_M1(pmax, mmax, para_list, file_prefix, data_dir=data_dir)
    for rank in range(max_rank):
        if (not isinstance(eig_val_list, Iterable)) and rank != 0:
            continue
        #
        if rank >= len(eig_val_list) :
            continue
        # 
        prob_eq, prob_g, eig_val = generate_P_matrix_FSP_M1(pmax, mmax, rank, eig_val_list, eig_vec_list) 
        title = file_prefix + f"_rank_{rank}"
        draw_P_hmap_FSP_M1(prob_eq, prob_g, eig_val, title, para_list, plot_dir=plot_dir)
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
    do_all_FSP_M1(pmax, mmax, para_list,  max_rank, plot_dir, data_dir)
#