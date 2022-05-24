import numpy.linalg as nplin
import numpy as np

from functools import lru_cache

class Transition_Matrix():
    def __init__(self, t_mat):
        self._t_mat = t_mat
        return
    
    def update_coef_vec(self, init_vec):
        self.coef_vec = nplin.inv(self.eig_vec).dot(init_vec)
    #

    @staticmethod
    def get_coef_vec(eig_vec, init_vec):
        X = nplin.inv(eig_vec).dot(init_vec)
        return X 
    #
    
    @lru_cache(5)
    def calculate_eigens(self, t_mat):
        return nplin.eig(t_mat)
    
    
    def update_eigens(self):
        self.eig_val, self.eig_vec = self.calculate_eigens(self._t_mat)
    #
    
    def get_eigens(self): 
        self.update_eigens()
        return self.eig_val, self.eig_vec
    #
    
    def get_state_t(self, t, init_vec):
        self.update_eigens()
        self.update_coef_vec(init_vec)
        val = 0
        for idx, ci in enumerate(self.coef_vec): 
            val += ci * self.eig_vec[:,idx] * np.exp( self.eig_val[idx]*t )
        #
        return val 
    # 
# 

class CME():
    def __init__(self, t_mat, x_len, y_len): 
        self.t_mat = Transition_Matrix(t_mat)
        self.x_len = x_len 
        self.y_len = y_len 
        self.prob_eq = np.zeros(shape=(x_len, y_len))
    # 
    
    @classmethod 
    def init_by_func(cls, fun, x_len, y_len, **args): 
        mat_len = x_len * y_len 
        mat = np.zeros(shape=(mat_len, mat_len))
        for i in range(mat_len):
            for j in range(mat_len): 
                mat[i][j] = fun(i, j, x_len, y_len, **args)
            #
        #
        return cls(mat, x_len, y_len)
    # 
    
    def __update_eigens(self): 
        self.t_mat.update_eigens() 
    # 
    
    def get_eigens(self):
        self.__update_eigens()
        return self.t_mat.get_eigens() 
    # 
    
    def __update_prob_eq(self): 
        self.
        for i in range(len(max_vec)): 
            m = i // (pmax + 1)
            p = i % (pmax + 1)
            
            prob_eq[m,p] = max_vec[i]/vec_norm
# 
    
    def get_prob_eq(self): 
        
    