
import os
import sys
import multiprocessing as mp 

import numpy as  np

from src import fb 
from src import fsp_diag as fsp 
from src import fsp_diag_m1 as fspm1 

from src.pocs.File_manage import read_write as rw

sys.path.append(os.getcwd())


def fsp_comparison(pmax, mmax, para_list):
    proj_dir = "FSP_comparison_02/" 

    rw.make_dir(proj_dir)

    case_dir = proj_dir + f"p_{pmax}_m_{mmax}/"
    data_dir = case_dir + "data/"
    plot_dir = case_dir + "plot/"
    rw.make_dir(case_dir)
    rw.make_dir(data_dir)
    rw.make_dir(plot_dir)
    
    file_prefix = f"p_{pmax}_m_{mmax}_para_{para_list[0]}_{para_list[1]}_{para_list[2]}"
    
    file_prefix_fb      = f"FB_" + file_prefix
    file_prefix_fsp     = f"FSP_" + file_prefix
    file_prefix_fspm1   = f"FSP_M1_" + file_prefix
    
    max_rank = 5
    
    fb.do_all_FB(pmax, mmax, para_list, max_rank, plot_dir, data_dir)
    fsp.do_all_FSP(pmax, mmax, para_list, max_rank, plot_dir, data_dir)
    fspm1.do_all_FSP_M1(pmax, mmax, para_list, max_rank, plot_dir, data_dir)
#


pmax_list = np.arange(1,199,2) 
mmax_list = np.arange(1,19,2)

para_list = [20, 2.5, 10]

work_list= [] 
for mmax in mmax_list: 
    for pmax in pmax_list:
        work_list.append((pmax, mmax, para_list))
    # 
#

pool = mp.Pool(1) 
pool.starmap(fsp_comparison, work_list)
