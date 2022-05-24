from numba import njit

from pocs.Math.simple_func import delta 
from Models.element_fun import protein_exp

@njit
def fill_matrix_FB(mat_len, pmax, mmax, mat, alpha, beta, gamma):
    for i in range(mat_len): 
        for j in range(mat_len):
            mat[i][j] = protein_exp(i, j, pmax, mmax, alpha, beta, gamma)
        #
    #
    
    return mat 
#
