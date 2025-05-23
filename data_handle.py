import numpy as np
import abc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import osqp
from math import ceil
import scipy.sparse as sp
import os
import pandas as pd
import re
from scipy.sparse.linalg import eigsh
from scipy.sparse import block_diag, hstack, vstack, csc_matrix
from copy import deepcopy







def read_data(data_dir):
    """
    Reads CSV data from the given directory and returns DataFrames plus metadata.
    Returns:
        gen_df, load_df, bat_df, line_df: pd.DataFrame
        slack_bus_idx: int
        T: int (number of timesteps)
    """
    # --- parameters ---
    slack_bus_idx = 5  # TODO: make configurable or read from file

    # --- load CSVs ---
    gen_df  = pd.read_csv(os.path.join(data_dir, 'gen.csv'),  sep=';')
    load_df = pd.read_csv(os.path.join(data_dir, 'load.csv'), sep=';')
    line_df = pd.read_csv(os.path.join(data_dir, 'line.csv'), sep=';')
    bat_df  = pd.read_csv(os.path.join(data_dir, 'bat.csv'),  sep=';')

    # strip stray spaces
    for df in (gen_df, load_df, line_df, bat_df):
        df.columns = df.columns.str.strip()

    # rename battery E_t0 → E_init if needed
    if 'E_t0' in bat_df.columns:
        bat_df.rename(columns={'E_t0': 'E_init'}, inplace=True)

    # infer T from time-series columns in generator data
    time_min_cols = [c for c in gen_df.columns if c.startswith('P_min^t_')]
    T = len(time_min_cols)

    return gen_df, load_df, bat_df, line_df, slack_bus_idx, T



def main():


    """ # --- parameters ---
    t_final = 1
    H = 30
    max_iter = 1000
    duplicate_network = 2
    duplicate_timesteps= 3
    
    data_dir = os.path.join(os.getcwd(), 'PSCC2025', 'MP_DC_ESS', 'bus30')
    gen_df, load_df, bat_df, line_df, slack, T = read_data(data_dir)
    net = construct_network(gen_df, load_df, bat_df, line_df, H, slack)
    sim = construct_sim(net, T, H, max_iter,
                        duplicate_network=duplicate_network,
                        duplicate_timesteps=duplicate_timesteps)

    sim2 = deepcopy(sim)
    sim3 = deepcopy(sim) """
    #final_X, final_theta, E_local = sim.centralized_coordinator(t_final, display=False)


    """ import time 

    start = time.perf_counter()
    X_centralized, theta_centralized, Eb_central, total_cost_centralized  = sim3.centralized_coordinator(t_final=t_final)
    dt3 = time.perf_counter() - start
    print(f"Centralized formulation took {dt3:.2f} s")
    print(f"total cost : {total_cost_centralized}")


    start = time.perf_counter()
    X_ADMM, theta_ADMM, final_lbda, SoC_network, total_cost_ADMM  = sim.ADMM_coordinator(t_final=t_final, display=False, display_dual=False, debug = False)
    dt1 = time.perf_counter() - start
    print(f"Threaded ADMM took {dt1:.2f} s")
    print(f"total cost : {total_cost_ADMM}") """

    """ start = time.perf_counter()
    final_X, final_theta, final_lbda, SoC_network, total_cost  = sim2.ADMM_coordinator_centralized_resolution(t_final=t_final, display=False, display_dual=False, debug = False)
    dt2 = time.perf_counter() - start
    print(f"Centralized solve took {dt2:.2f} s")

    print(f"Speed-up: {dt2/dt1:.2f}×") """

    
    
    """ print("\n\n")
    print(X_centralized - X_ADMM)
    print("\n", X_centralized, "\n", X_ADMM)
    print("\n\n")
    print(theta_centralized - theta_ADMM)
    print("\n", theta_centralized, "\n", theta_ADMM)
    print("\n\n")
    print(total_cost_centralized, total_cost_ADMM)
    print("\n")
    print(f" t_centralise : {dt3}; t_decentralise: {dt1}; Speed-up: {dt3/dt1:.2f}")


    plt.rcParams['axes.formatter.useoffset'] = False
    plt.rcParams['axes.formatter.use_locale'] = False
 """

    """ plt.figure()
    time_axis = np.arange(0, len(SoC_network[0]), 1)
    print(time_axis)
    print(SoC_network)
    for SoC in SoC_network:
        plt.plot(time_axis, SoC, 'r+-')
    plt.title(f"Battery state{H}") """
    
    """ print(total_cost)
    print(np.sum(total_cost))

    plt.figure()
    plt.plot(range(len(total_cost)), total_cost, linewidth=2)
    plt.xlabel('Time step')
    plt.ylabel('Total network cost (ADMM)')
    plt.title('ADMM: Sum of All Buses’ Costs')
    plt.grid(True)
    plt.tight_layout()
    plt.show() """
        



if __name__ == "__main__":

    main()