from numba import njit

@njit 
def delta(a, b): 
    if a == b:
        return 1 
    else : 
        return 0 
    #
# 

@njit 
def matrixNormalization(mat_len, mat):
    for i in range(mat_len):
        mat[i][i] = mat[i][i] - sum(mat[:,i]) 
    #
    return mat
#