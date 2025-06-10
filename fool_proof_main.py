import abc
import numpy as np
from data_handle import read_data
import scipy.sparse as sp
import osqp

#TODO: remove later
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import copy

import logging
import time
import csv
import pickle
import re

from math import ceil




class Line():
    def __init__(self, line_idx, from_bus, to_bus, x, P_max = 1e6):
        self.line_idx = line_idx
        self.from_bus = from_bus
        self.to_bus = to_bus
        self.x = x
        self.P_max = P_max

class Bus():
    def __init__(self, idx):
        self.idx = idx
        self.devices = []
        self.device_slices = {}
        # OSQP warm start
        self.x_shape = 0
        self.x0 = 0
        self.y0 = 0
    
    def __repr__(self):
        if not self.devices:
            return f"bus {self.idx} without any device"
        devs = ', '.join(str(d) for d in self.devices)
        return f"bus {self.idx} with devices: {devs}"

    def add_device(self, device):
        self.devices.append(device)

    def compute_problem(self, H, t0):

        # handle no devices case
        if not self.devices:
            P_bus = sp.csc_matrix((0, 0))
            q_bus = np.zeros(0)
            A_bus = sp.csc_matrix((0, 0))
            self.l = np.zeros(0)
            self.u = np.zeros(0)
            return P_bus, q_bus, A_bus
        
        # initialize lists for stacking
        P_blocks, q_blocks = [], []
        A_blocks, l_blocks, u_blocks = [], [], []

        # reset slice mapping
        self.device_slices.clear()
        ptr = 0

        # loop through devices and collect subproblems
        for device in self.devices:
            P_d, q_d, A_d, l_d, u_d = device.build_problem(H, t0)

            # record slices depending on device type
            name = device.__class__.__name__.lower()
            if name == 'battery':
                # battery: [Pc, Pd, E]
                pc_slice = slice(ptr, ptr + H)
                self.device_slices[f'P_c_{device.idx}'] = pc_slice
                ptr += H
                pd_slice = slice(ptr, ptr + H)
                self.device_slices[f'P_d_{device.idx}'] = pd_slice
                ptr += H
                # skip energy states
                e_slice = slice(ptr, ptr + H + 1)
                self.device_slices[f'E_{device.idx}'] = e_slice
                ptr += H + 1
            else:
                # single power variable generator, load, slack
                p_slice = slice(ptr, ptr + H)
                self.device_slices[f'P_{name}_{device.idx}'] = p_slice
                ptr += H

            # stack blocks
            P_blocks.append(P_d)
            q_blocks.append(q_d)
            A_blocks.append(A_d)
            l_blocks.append(l_d)
            u_blocks.append(u_d)

        # block diagonal P, vertical stack A
        self.P_bus = sp.block_diag(P_blocks, format='csc')
        self.q_bus = np.concatenate(q_blocks)
        self.A_bus = sp.block_diag(A_blocks, format='csc')
        
        # Store them internally since they never change
        self.l = np.concatenate(l_blocks)
        self.u = np.concatenate(u_blocks)

        return self.P_bus, self.q_bus, self.A_bus
    
    def compute_problem_update(self, H, X_c, lam, rho):

        # make copies of the original cost
        P_aug = self.P_bus.tolil().tocsc()
        q_aug = self.q_bus.copy()
        n = P_aug.shape[0]

        # build coefficient vector: coeffs[i] = +1 for P, -1 for P_c, +1 for P_d, 0 otherwise

        for t in range(H):
            z_t = np.zeros(n)
            for key, sl in self.device_slices.items():
                
                if key.startswith('P_c_'):
                    z_t[sl.start + t ] = -1
                elif key.startswith('P_d_'):
                    z_t[sl.start + t ] = 1
                elif key.startswith('P_generator_') or key.startswith('P_slack_'):
                    z_t[sl.start + t ] = 1
                elif key.startswith('P_load_'):
                    z_t[sl.start + t ] = -1
                else:
                    continue
                
            P_aug += rho * sp.csc_matrix(z_t[:, None] * z_t[None, :])
            q_aug += z_t * (lam[t] - rho * X_c[t])

        return P_aug, q_aug
    



    def build_problem(self, H, t0):
        solver = osqp.OSQP()
        P, q, A = self.compute_problem(H, t0)
        if P.shape[0] == 0:
            self.solver = None
            self.is_empty = True
            return
        

        # Add dummy coefficient to ensure properly update in the admm loop
        n      = self.P_bus.shape[0]
        P_pattern = self.P_bus.copy()
        for t in range(H):
            z_t = np.zeros(n)
            for key, sl in self.device_slices.items():
                if   key.startswith('P_c_'):    
                    z_t[sl.start + t] = -1
                elif key.startswith('P_d_'):
                    z_t[sl.start + t] = +1
                elif key.startswith('P_generator_') or key.startswith('P_slack_'):
                    z_t[sl.start + t] = +1
                elif key.startswith('P_load_'):
                    z_t[sl.start + t] = -1
                else:
                    continue
            P_pattern += 1 * sp.csc_matrix(z_t[:, None] * z_t[None, :]) # rho but who cares


        # 2) form a sparse rank‐one matrix of all the off‐diagonals you’ll need
        #    (and a 1 on the diag so you get the full pattern, but we’ll overwrite later)
        P_pattern = sp.triu(P_pattern, format='csc')

        # 3) keep only the upper‐triangle (OSQP wants that)
        P_pattern = sp.triu(P_pattern, format='csc')

        self.is_empty = False
        solver.setup(P=P_pattern, q=q, A=A, l=self.l, u=self.u, eps_abs=1e-3, eps_rel=1e-3, max_iter=2000000, polish=False, verbose=False)
        self.solver = solver

    def solve_problem(self, it):
        if it > 0:
            self.solver.warm_start(x=self.x0, y=self.y0)

        res = self.solver.solve()
        if res.info.status not in ('solved','solved_inaccurate'):
            raise RuntimeError(f"Bus {self.idx} local QP status: {res.info.status}")
        self.x0 = res.x.copy()
        self.y0 = res.y.copy()

        return res

    def parse_solution(self, x):
        result = {}
        # net injection time-series
        net = np.zeros_like(x[self.device_slices[next(iter(self.device_slices))]])
        for name, sl in self.device_slices.items():
            result[name] = x[sl]
            if name.startswith('P_c_'):
                net -= x[sl]                  # charge withdraws from bus
            elif name.startswith('P_d_'):
                net += x[sl]                  # discharge injects into bus
            elif name.startswith('P_generator_') or name.startswith('P_slack_'):
                net += x[sl]                  # gens & slack inject
            elif name.startswith('P_load_'):
                net -= x[sl]                  # loads withdraw
        result[f'P_bus_{self.idx}'] = net

        return result

    def compute_cost(self, x):

        # first parse out each device's timeseries
        if not self.devices:
            return 0.0

        # else parse the solution as before
        sol = self.parse_solution(x)

        total_cost = 0.0
        for dev in self.devices:
            cls = dev.__class__.__name__.lower()
            idx = dev.idx

            if cls == 'generator':
                P = sol[f'P_generator_{idx}']
                total_cost += dev.compute_cost(P)

            elif cls == 'load':
                P = sol[f'P_load_{idx}']
                total_cost += dev.compute_cost(P)

            elif cls == 'battery':
                P_c = sol[f'P_c_{idx}']
                P_d = sol[f'P_d_{idx}']
                total_cost += dev.compute_cost(P_c, P_d)

            elif cls == 'slack':
                P = sol[f'P_slack_{idx}']
                total_cost += dev.compute_cost(P)

        return total_cost



class Device(abc.ABC):

    def __init__(self, idx, bus_idx, P_bounds):
        """
        idx:    unique device index
        H:      horizon length (number of time steps) <= T 
        P_bounds: 2xT array, [lb; ub] for each time step
        """
        self.idx = idx
        self.bus_idx = bus_idx
        self.lb = P_bounds[0, :]
        self.ub = P_bounds[1, :]

    @abc.abstractmethod
    def clone(self):
        ...

    @abc.abstractmethod
    def build_problem(self, H, t0):
        ...
    
    def __repr__(self):
        name = self.__class__.__name__.lower()
        return f"{name} {self.idx}"

class Generator(Device):
    def __init__(self, idx, bus_idx, P_bounds, poly_cost):
        super().__init__(idx, bus_idx, P_bounds)
        self.c2, self.c1, self.c0 = poly_cost

    def build_problem(self, H: int, t0: float):
        diag_vals = 2*self.c2 * np.ones(H)
        P = sp.diags(diag_vals, 0, format='csc')
        q = self.c1 * np.ones(H)
        A = sp.eye(H, format='csc')
        l = self.lb[t0:t0+H]
        u = self.ub[t0:t0+H]
        return P, q, A, l, u

    def clone(self, idx_offset):
        return Generator(idx = self.idx+idx_offset, bus_idx=self.bus_idx + idx_offset, P_bounds = np.vstack([self.lb, self.ub]), poly_cost = np.array([self.c2, self.c1, self.c0]))
    
    def compute_cost(self, P):
        return np.sum(self.c2 * np.square(P) + self.c1 * P + self.c0)

class Load(Device):
    def __init__(self, idx, bus_idx, P_bounds):
        super().__init__(idx, bus_idx, P_bounds)
    
    def build_problem(self, H, t0: float):
        P = sp.csc_matrix((H, H))
        q = np.zeros(H)
        A = sp.eye(H, format='csc')
        l, u = self.lb[t0:t0+H], self.ub[t0:t0+H]        
        return P, q, A, l, u

    def clone(self, idx_offset):        
        return Load(idx = self.idx+idx_offset, bus_idx=self.bus_idx+idx_offset ,  P_bounds = np.vstack([self.lb, self.ub]))

    def compute_cost(self, P):
        return 0

       
class Battery(Device):
    def __init__(self, idx, bus_idx, P_bounds, bat_cost, E_init, E_bounds, dT, eta_c, eta_d):
        super().__init__(idx, bus_idx, P_bounds)
        self.E_init = E_init
        self.E_min = E_bounds[0]
        self.E_max = E_bounds[1]
        self.dT = dT
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.c1_d, self.c1_c = bat_cost
        self.new_soc = 0

    def build_problem(self, H, t0: float):
        # x = [P_c(0:H-1), P_d(0:H-1), E(0:H)]
        n = 2*H + (H+1)

        # 1) cost
        # 1a) quadratic cost
        P = sp.csc_matrix((n, n))

        # 1b) linear cost q
        q = np.hstack([
                        self.c1_c*np.ones(H),
                        self.c1_d*np.ones(H),
                        np.zeros(H+1)
                        ])

        # 2) bounds on P_c, P_d, E[1..H]

        # inequality blocks (P_c, P_d bounds; only E[1..H] has E_min/E_max)
        mats, l_ineq, u_ineq = [], [], []

        I_H = sp.eye(H,   format='csc')
        Z_H = sp.csc_matrix((H,   H))     # zeros for P_c / P_d slots
        Z_E = sp.csc_matrix((H, H+1))     # zeros for E[0..H] slots

        # selector for E[1..H] out of E[0..H]
        Esel = sp.eye(H+1, format='csc')[1:,:]  # shape H×(H+1)
        
        P_c_max = -self.lb[t0:t0+H]
        P_d_max =  self.ub[t0:t0+H]

        # 2a) 0 ≤ P_c ≤ P_c_max
        mats.append(
        sp.hstack([ I_H,   Z_H,   Z_E ], format='csc')
        )
        l_ineq.extend([0.0]*H)
        u_ineq.extend(P_c_max)

        # 2b) 0 ≤ P_d ≤ P_d_max
        mats.append(
        sp.hstack([ Z_H,   I_H,   Z_E ], format='csc')
        )
        l_ineq.extend([0.0]*H)
        u_ineq.extend(P_d_max)

        # 2c) E_min ≤ E[1..H] ≤ E_max
        mats.append(sp.hstack([ Z_H, Z_H, Esel ], format='csc'))
        l_ineq.extend([self.E_min]*H)
        u_ineq.extend([self.E_max]*H)


        A_ineq = sp.vstack(mats, format='csc')
        l_ineq = np.array(l_ineq)
        u_ineq = np.array(u_ineq)

        # 3) dynamics + initial SOC
        data, rows, cols, b_eq = [], [], [], []

        # 3a) pin E[0] = E_init
        rows.append(0); cols.append(2*H + 0); data.append(1.0)
        b_eq.append(self.E_init)

        # 3b) for h=1..H:
        #   E[h] - E[h-1] - dT*(ηc P_c[h-1] - P_d[h-1]/ηd) = 0
        for h in range(1, H+1):
            row = h
            # E[h]
            rows.append(row); cols.append(2*H + h);     data.append( 1.0)
            # E[h-1]
            rows.append(row); cols.append(2*H + h - 1); data.append(-1.0)
            # P_c[h-1]
            rows.append(row); cols.append(h - 1);       data.append(-self.dT*self.eta_c)
            # P_d[h-1]
            rows.append(row); cols.append(H + h - 1);   data.append(+self.dT*(1.0/self.eta_d))
            b_eq.append(0.0)

        A_dyn = sp.coo_matrix((data, (rows, cols)),
                              shape=(H+1, n)).tocsc()
        l_dyn = np.array(b_eq)
        u_dyn = l_dyn.copy()

        # 4) stack
        A = sp.vstack([A_ineq, A_dyn], format='csc')
        l = np.hstack([l_ineq, l_dyn])
        u = np.hstack([u_ineq, u_dyn])

        return P, q, A, l, u
    
    
    def clone(self, idx_offset: int):
        return Battery( idx       = self.idx + idx_offset,
                        bus_idx=self.bus_idx + idx_offset,
                        P_bounds  = np.vstack([self.lb, self.ub]),
                        bat_cost  = np.array([self.c1_d, self.c1_c]),
                        E_init    = self.E_init,
                        E_bounds  = np.array([self.E_min, self.E_max]),
                        dT        = self.dT,
                        eta_c     = self.eta_c,
                        eta_d     = self.eta_d
                    )
    def compute_cost(self, P_c, P_d):

        return np.sum(self.c1_c * P_c + self.c1_d * P_d)
    
class Slack(Device):
    def __init__(self, idx, bus_idx, P_bounds, poly_cost):
        super().__init__(idx, bus_idx, P_bounds)
        self.c2, self.c1, self.c0 = poly_cost

    def build_problem(self, H: int, t0: float):
        diag_vals = 2*self.c2 * np.ones(H)
        P = sp.diags(diag_vals, 0, format='csc')
        q = self.c1 * np.ones(H)
        A = sp.eye(H, format='csc')
        l = self.lb[t0:t0+H]
        u = self.ub[t0:t0+H]
        return P, q, A, l, u
    
    def clone(self, offset_bus):
        # don't clone the slack
        pass

    def compute_cost(self, p):
        return np.sum(self.c2 * p**2 + self.c1 * p + self.c0)

class NetworkDCOPF:
    def __init__(self, lines, buses, slack_idx):
        self.lines = lines
        self.buses = buses
        self.nb    = len(buses)
        self.slack = slack_idx
        B = np.zeros((self.nb, self.nb))
        self.flows = []
        for ln in lines:
            b = 1.0/ln.x
            i,j = ln.from_bus, ln.to_bus
            B[i,i] += b; B[j,j] += b
            B[i,j] -= b; B[j,i] -= b
            self.flows.append((i,j,b, ln.P_max))
        self.B    = sp.csc_matrix(B)
        self.Bbus = B

    def build_base(self, rho_bus, eps_abs=1e-3, eps_rel=1e-3, max_iter=2000000):
        n = self.nb
        rows, cols, data, l, u = [], [], [], [], []
        r = 0
        # slack angle
        rows.append(r); cols.append(self.slack); data.append(1.0)
        l.append(0.0); u.append(0.0); r+=1
        # line limits
        for i,j,b,Pmax in self.flows:
            rows += [r,r]; cols += [i,j]; data += [ b, -b ]
            
            l.append(-Pmax); u.append(Pmax)
            r+=1
        A = sp.csc_matrix((data,(rows,cols)), shape=(r,n))

        self.P_net = self.B.T @ sp.diags(rho_bus) @ self.B
        self.q_net = np.zeros(n)

        self.osqp = osqp.OSQP()
        self.osqp.setup(P=self.P_net, q=self.q_net, A=A, l=np.array(l), u=np.array(u),
                        eps_abs=eps_abs, eps_rel=eps_rel,
                        max_iter=max_iter, polish=False, verbose=False)

        self.x0 = np.zeros(n)
        self.y0 = np.zeros(r)

        P_base = self.B.T @ self.B            # dense, but fixed pattern
        P_base_triu = sp.triu(P_base, format='csc')
        self._P_pattern = P_base_triu.data.copy()


    def update_problem(self, X_m, lam, rho_bus):
        P_new = (self.B.T @ sp.diags(rho_bus) @ self.B).tocsc()
        P_triu = sp.triu(P_new, format='csc')
        self.q_net = - (self.B.T @ lam   +   self.B.T @ (rho_bus * X_m))
        self.osqp.update(Px = P_triu.data, q=self.q_net)

    def solve(self):
        self.osqp.warm_start(x=self.x0, y=self.y0)
        res = self.osqp.solve()
        if res.info.status not in ('solved','solved_inaccurate'):
            raise RuntimeError(f"Network QP failed: {res.info.status}")
        self.x0 = res.x; self.y0 = res.y
        theta = res.x
        P_inj = self.B.dot(theta)
        return P_inj, theta


def dc_opf_admm(network,
                H,
                rho_bus,
                max_iter=2000,
                tol=1e-3,
                debug=True,
                display=True,
                display_dual=True):
    nb = network.nb
    buses = network.buses

    # Histories: iteration × buses × H
    if display:
        P_bus_hist = np.zeros((max_iter+1, nb, H))
        P_coor_hist   = np.zeros((max_iter+1, nb, H))
        lbda_hist   = np.zeros((max_iter+1, nb, H))
        P_inj_hist = np.zeros((max_iter+1, nb))
    else:
        P_bus_hist = None
        P_coor_hist   = None
        lbda_hist   = None
        P_inj_hist = None

    # device power histories, keyed by sol‐keys (only length H)
    device_hist = {}

    # initial consensus & dual
    P_coor = np.zeros((nb, H))
    lbda = np.zeros((nb, H))

    # build subproblems
    for bus in buses:
        bus.build_problem(H, t0=0)
    network.build_base(rho_bus)

    # record iteration 0
    if display:
        P_bus_hist[0] = 0
        P_coor_hist[0]   = P_coor
        lbda_hist[0]   = lbda
        P_inj_hist[0] = 0
    P_bus = np.zeros((nb, H))

    P_coor_prev = P_coor.copy()

    for it in range(max_iter):
        if it %100 == 0:
            print(f"\n=== ADMM Iter {it} ===")
            pass
        for bus in buses:
            if bus.is_empty:
                sol = {f'P_bus_{bus.idx}': np.zeros(H)}
            else:
                # ADMM update & re‐solve
                # 1) build & apply augmented QP
                P_aug, q_aug = bus.compute_problem_update(H, P_coor[bus.idx], lbda[bus.idx], rho_bus[bus.idx])
                P_aug_triu = sp.triu(P_aug, format='csc')
               
                bus.solver.update(Px=P_aug_triu.data.copy(), q=q_aug)

                res = bus.solve_problem(it)
                sol = bus.parse_solution(res.x)

                P_bus[bus.idx] = sol[f'P_bus_{bus.idx}']             
                

            # initialize device_hist on first iteration
            if display:
                if it == 0:
                    for name, arr in sol.items():
                        # only keep those of length H (i.e. power variables)
                        if arr.shape == (H,):
                            device_hist[name] = np.zeros((max_iter+1, nb, H))

                # store only the length‐H series
                for name, arr in sol.items():
                    if name in device_hist:
                        device_hist[name][it, bus.idx] = arr


        # --- network coordination step ---
        for t in range(H):
            p_bus_vec = P_bus[:, t]
            lam_vec   = lbda[:,    t]
            network.update_problem(p_bus_vec, lam_vec, rho_bus)
            P_inj_t, theta_t = network.solve()
            P_coor[:, t] = P_inj_t

        # --- update dual ---
        lbda += rho_bus[:, None] * (P_bus - P_coor)

        # record histories at it+1
        if display:
            P_bus_hist[it] = P_bus
            P_coor_hist[it]   = P_coor
            lbda_hist[it]   = lbda
            P_inj_hist[it] = P_inj_t

        r_norm = np.linalg.norm(P_bus - P_coor) 
        s_norm = np.linalg.norm(rho_bus[:,None] * (P_coor - P_coor_prev))

        # save for next dual‐residual
        P_coor_prev = P_coor.copy()

        # only break once we’ve done ≥1 iteration AND both are below tol
        if it >= 1 and r_norm < tol and s_norm < tol:
            print(f"Converged at iter {it}")
            last = it
            break

        else:
            last = max_iter

    # --- PLOTTING ---
    if display:
        its   = np.arange(last)
        nbus  = nb
        ncols = 4
        nrows = int(np.ceil(nbus / ncols))
        fig, axs = plt.subplots(nrows, ncols,
                                figsize=(4*ncols, 2.5*nrows),
                                sharex=True)
        axs = axs.flatten()

        # pick up Tableau palette
        palette = list(mcolors.TABLEAU_COLORS.values())
        # exactly the same key order as your second code:
        color_key = ['P_bus','P_gen','P_load','P_bat','P_slack','P_coor','dual']
        color_dict = {
            name: palette[i % len(palette)]
            for i, name in enumerate(color_key)
        }

        for i, bus in enumerate(buses):
            ax = axs[i]
            has_bat = False
            """ # plot bus active power
            ax.plot(its,
                    P_bus_hist[:last+1, i, 0],
                    color=color_dict['P_bus'],
                    linewidth=2,
                    label='P_bus') """

            # plot coordinator
            ax.plot(its,
                    P_coor_hist[:last, i, 0],
                    color=color_dict['P_coor'],
                    linewidth=2,
                    linestyle='--',
                    label='P_coor')

            # plot each device (skip charge/discharge keys)
            for key, hist in device_hist.items():
                
                series = hist[ :last, i, 0]
                if np.any(series):           # skip entirely-zero keys
                    if key.startswith('P_c_'):
                        series_P_c = series
                        has_bat = True
                    elif key.startswith('P_d_'):
                        series_P_d = series
                    else:
                        if key.startswith('P_c_'):
                            color_dict_key = "P_c"
                        elif key.startswith('P_generator_'):
                            color_dict_key = "P_gen"
                        elif key.startswith('P_slack_'):
                            color_dict_key = "P_slack"
                        elif key.startswith('P_load_'):
                            color_dict_key = "P_load"
                            series = - series
                        elif key.startswith('P_bus_'):
                            color_dict_key = "P_bus"
                        elif key.startswith('P_c_'):
                            color_dict_key = "P_c"
                        elif key.startswith('P_d_'):
                            color_dict_key = "P_d"

                        ax.plot(its,
                                series,
                                color = color_dict[color_dict_key],
                                label=key,
                                linewidth=2)
            if has_bat:
                ax.plot(its,
                        series_P_c-series_P_d,
                        color = color_dict['P_bat'],
                        label="P_bat",
                        linewidth=2)

                
        

            

            # dual λ
            if display_dual:
                ax.plot(its,
                        lbda_hist[:last, i, 0],
                        color=color_dict['dual'],
                        linewidth=2,
                        linestyle=':',
                        label='dual λ')

            # annotate devices on this bus
            devs = [f"{type(d).__name__.lower()} {d.idx}"
                    for d in buses[i].devices]
            dev_str = '\n'.join(devs) if devs else 'no devices'
            ax.set_ylabel(f'Bus {i}\n{dev_str}', fontsize=9)

            ax.grid(True)
            ax.legend(fontsize=6)

        # turn off any unused axes
        for j in range(nbus, len(axs)):
            axs[j].axis('off')

        axs[-1].set_xlabel('ADMM iteration', fontsize=12)
        plt.tight_layout()
        plt.show()

    return P_bus, P_inj_t, theta_t, last





def dc_opf_admm_threaded(network,
                H,
                rho_bus,
                max_iter=2000,
                tol=1e-3,
                debug=True,
                display=True,
                display_dual=True):
    nb = network.nb
    buses = network.buses

    # 1) Build each bus's OSQP solver once
    for bus in buses:
        bus.build_problem(H, t0=0)
    network.build_base(rho_bus)

    # 2) Prepare histories if requested
    if display:
        P_bus_hist  = np.zeros((max_iter+1, nb, H))
        P_coor_hist = np.zeros((max_iter+1, nb, H))
        lbda_hist   = np.zeros((max_iter+1, nb, H))
        P_inj_hist  = np.zeros((max_iter+1, nb))
        device_hist = {}
    else:
        P_bus_hist = P_coor_hist = lbda_hist = P_inj_hist = device_hist = None

    # 3) Initialize ADMM variables
    P_bus  = np.zeros((nb, H))
    P_coor = np.zeros((nb, H))
    P_coor_prev = P_coor.copy()
    lbda   = np.zeros((nb, H))
    last   = max_iter
    P_inj_t = np.zeros(nb)

    # 4) Thread pool size
    max_workers = max(1, os.cpu_count() - 1)

    # 5) Solver subroutine for one bus
    def solve_one(idx, it):
        bus = buses[idx]
        if bus.is_empty:
            return idx, {f'P_bus_{idx}': np.zeros(H)}

        # build augmented QP
        P_aug, q_aug = bus.compute_problem_update(
            H,
            P_coor[idx],
            lbda[idx],
            rho_bus[idx]
        )
        P_aug_triu = sp.triu(P_aug, format='csc')
        bus.solver.update(Px=P_aug_triu.data.copy(), q=q_aug)

        # solve: skip warm_start on first iteration
        if it == 1:
            res = bus.solver.solve()
        else:
            res = bus.solve_problem(it)

        if res.info.status not in ('solved', 'solved_inaccurate'):
            raise RuntimeError(f"Bus {idx} QP failed: {res.info.status}")

        # save warm-start for next iteration
        bus.x0, bus.y0 = res.x.copy(), res.y.copy()

        sol = bus.parse_solution(res.x)
        return idx, sol

    # 6) Main ADMM loop
    for it in range(1, max_iter+1):
        if it % 100 == 0 and debug:
            print(f"\n=== ADMM Iter {it} ===")

        # local solves in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = [exe.submit(solve_one, i, it) for i in range(nb)]
            for future in as_completed(futures):
                idx, sol = future.result()
                P_bus[idx] = sol.get(f'P_bus_{idx}', np.zeros(H))

                if display:
                    if it == 1:
                        for name, arr in sol.items():
                            if arr.shape == (H,):
                                device_hist[name] = np.zeros((max_iter+1, nb, H))
                    for name, arr in sol.items():
                        if name in device_hist:
                            device_hist[name][it, idx] = arr

        # network coordination
        for t in range(H):
            network.update_problem(P_bus[:, t], lbda[:, t], rho_bus)
            P_inj_t, theta_t = network.solve()
            P_coor[:, t] = P_inj_t

        # dual update
        lbda += rho_bus[:, None] * (P_bus - P_coor)

        # record histories
        if display:
            P_bus_hist[it]  = P_bus.copy()
            P_coor_hist[it] = P_coor.copy()
            lbda_hist[it]   = lbda.copy()
            P_inj_hist[it]  = P_inj_t.copy()

        # convergence check
        r_norm = np.linalg.norm(P_bus - P_coor)
        s_norm = np.linalg.norm(rho_bus[:, None] * (P_coor - P_coor_prev))
        P_coor_prev = P_coor.copy()
        if r_norm < tol and s_norm < tol:
            if debug:
                print(f"Converged at iter {it}")
            break
  
    # --- PLOTTING ---
    if display:
        its   = np.arange(last)
        nbus  = nb
        ncols = 4
        nrows = int(np.ceil(nbus / ncols))
        fig, axs = plt.subplots(nrows, ncols,
                                figsize=(4*ncols, 2.5*nrows),
                                sharex=True)
        axs = axs.flatten()

        # pick up Tableau palette
        palette = list(mcolors.TABLEAU_COLORS.values())
        # exactly the same key order as your second code:
        color_key = ['P_bus','P_gen','P_load','P_bat','P_slack','P_coor','dual']
        color_dict = {
            name: palette[i % len(palette)]
            for i, name in enumerate(color_key)
        }

        for i, bus in enumerate(buses):
            ax = axs[i]
            has_bat = False
            """ # plot bus active power
            ax.plot(its,
                    P_bus_hist[:last+1, i, 0],
                    color=color_dict['P_bus'],
                    linewidth=2,
                    label='P_bus') """

            # plot coordinator
            ax.plot(its,
                    P_coor_hist[:last, i, 0],
                    color=color_dict['P_coor'],
                    linewidth=2,
                    linestyle='--',
                    label='P_coor')

            # plot each device (skip charge/discharge keys)
            for key, hist in device_hist.items():
                
                series = hist[ :last, i, 0]
                if np.any(series):           # skip entirely-zero keys
                    if key.startswith('P_c_'):
                        series_P_c = series
                        has_bat = True
                    elif key.startswith('P_d_'):
                        series_P_d = series
                    else:
                        if key.startswith('P_c_'):
                            color_dict_key = "P_c"
                        elif key.startswith('P_generator_'):
                            color_dict_key = "P_gen"
                        elif key.startswith('P_slack_'):
                            color_dict_key = "P_slack"
                        elif key.startswith('P_load_'):
                            color_dict_key = "P_load"
                            series = - series
                        elif key.startswith('P_bus_'):
                            color_dict_key = "P_bus"
                        elif key.startswith('P_c_'):
                            color_dict_key = "P_c"
                        elif key.startswith('P_d_'):
                            color_dict_key = "P_d"

                        ax.plot(its,
                                series,
                                color = color_dict[color_dict_key],
                                label=key,
                                linewidth=2)
            if has_bat:
                ax.plot(its,
                        series_P_c-series_P_d,
                        color = color_dict['P_bat'],
                        label="P_bat",
                        linewidth=2)

                
        

            

            # dual λ
            if display_dual:
                ax.plot(its,
                        lbda_hist[:last, i, 0],
                        color=color_dict['dual'],
                        linewidth=2,
                        linestyle=':',
                        label='dual λ')

            # annotate devices on this bus
            devs = [f"{type(d).__name__.lower()} {d.idx}"
                    for d in buses[i].devices]
            dev_str = '\n'.join(devs) if devs else 'no devices'
            ax.set_ylabel(f'Bus {i}\n{dev_str}', fontsize=9)

            ax.grid(True)
            ax.legend(fontsize=6)

        # turn off any unused axes
        for j in range(nbus, len(axs)):
            axs[j].axis('off')

        axs[-1].set_xlabel('ADMM iteration', fontsize=12)
        plt.tight_layout()
        plt.show()

    return P_bus, P_inj_t, theta_t



def centralized_dc_opf(net, H, t0=0):
    buses = net.buses
    lines = net.lines
    slack_idx = net.slack

    nb = len(buses)

    # --- 1) Build and collect all per-bus local problem blocks ---
    bus_offsets = []
    P_blocks, q_blocks = [], []
    A_blocks, l_blocks, u_blocks = [], [], []
    offset = 0
    for bus in buses:
        # compute and store each bus's local QP data
        P_bus, q_bus, A_bus = bus.compute_problem(H, t0)
        bus_offsets.append(offset)
        P_blocks.append(P_bus)
        q_blocks.append(q_bus)
        A_blocks.append(A_bus)
        l_blocks.append(bus.l)
        u_blocks.append(bus.u)
        offset += P_bus.shape[0]
    n_dev = offset

    # stack device-level matrices/vectors
    P_dev = sp.block_diag(P_blocks, format='csc')
    q_dev = np.concatenate(q_blocks)
    A_dev = sp.block_diag(A_blocks, format='csc')
    l_dev = np.concatenate(l_blocks)
    u_dev = np.concatenate(u_blocks)

    # --- 2) Add global bus-angle variables ---
    # one angle per bus per time-step
    n_theta = nb * H
    theta_offset = n_dev

    # build full objective matrix
    P_theta = sp.csc_matrix((n_theta, n_theta))
    P = sp.block_diag([P_dev, P_theta], format='csc')
    q = np.hstack([q_dev, np.zeros(n_theta)])

    # --- 3) Build global network constraints ---
    rows, cols, data = [], [], []
    l_net, u_net = [], []
    row = 0

    # precompute Bbus susceptance Laplacian
    Bbus = build_Bbus(nb, lines)

    # a) power-injection balance at each bus, each t
    for k, bus in enumerate(buses):
        for t in range(H):
            # sum of device injections
            for name, sl in bus.device_slices.items():
                if   name.startswith('P_c_'):
                    sign = -1
                elif name.startswith('P_d_'):
                    sign = +1
                elif name.startswith('P_generator_') or name.startswith('P_slack_'):
                    sign = +1
                elif name.startswith('P_load_'):
                    sign = -1
                else:
                    # *don’t* include E-slices (or anything else)
                    continue

                for i_local in range(sl.start, sl.stop):
                    rows.append(row)
                    cols.append(bus_offsets[k] + i_local)
                    data.append(sign)
            # network contribution: -Bbus * theta
            for j in range(nb):
                coeff = -Bbus[k, j]
                if coeff != 0:
                    idx = theta_offset + t*nb + j
                    rows.append(row)
                    cols.append(idx)
                    data.append(coeff)
            l_net.append(0.0)
            u_net.append(0.0)
            row += 1

    # b) fix slack-bus angle = 0 for each t
    for t in range(H):
        rows.append(row)
        cols.append(theta_offset + t*nb + slack_idx)
        data.append(1.0)
        l_net.append(0.0)
        u_net.append(0.0)
        row += 1

    # c) line flow limits for each line, each t
    for ln in lines:
        i, j = ln.from_bus, ln.to_bus
        b = 1.0 / ln.x
        Pmax = ln.P_max
        for t in range(H):
            rows += [row, row]
            cols += [theta_offset + t*nb + i, theta_offset + t*nb + j]
            data += [b, -b]
            l_net.append(-Pmax)
            u_net.append(Pmax)
            row += 1

    A_net = sp.csc_matrix((data, (rows, cols)), shape=(row, n_dev + n_theta))
    l_global = np.hstack([l_dev, l_net])
    u_global = np.hstack([u_dev, u_net])

    # --- 4) Solve with OSQP ---
    # pad A_dev with zeros for theta columns
    zero_pad = sp.csc_matrix((A_dev.shape[0], n_theta))
    A_dev_full = sp.hstack([A_dev, zero_pad], format='csc')

    # stack global constraints
    A = sp.vstack([A_dev_full, A_net], format='csc')

    # OSQP expects upper-triangular P
    P_triu = sp.triu(P, format='csc')
    solver = osqp.OSQP()
    solver.setup(P=P_triu, q=q,
                 A=A,
                 l=np.hstack([l_dev, l_net]),
                 u=np.hstack([u_dev, u_net]),
                 eps_abs=1e-3, eps_rel=1e-3,
                 polish = False,
                 verbose=True,
                 max_iter=2000000)
    print("starting solving")
    res = solver.solve()
    if res.info.status not in ('solved', 'solved_inaccurate'):
        raise RuntimeError(f"Central DC-OPF failed: {res.info.status}")

    # --- 5) Parse results ---
    x = res.x
    device_sols = {}
    for k, bus in enumerate(buses):
        lo = bus_offsets[k]; hi = lo + P_blocks[k].shape[0]
        if lo<hi:
            device_sols[k] = bus.parse_solution(x[lo:hi])
        else:
            # no devices: zero net injection
            device_sols[k] = {f'P_bus_{k}': np.zeros(H)}

    theta = x[theta_offset:].reshape(H,nb).T if n_theta>0 else np.zeros((nb,H))


    return device_sols, theta


def build_Bbus(nb, lines):
    B = np.zeros((nb, nb))
    for ln in lines:
        b = 1.0 / ln.x
        i, j = ln.from_bus, ln.to_bus
        B[i, i] += b; B[j, j] += b
        B[i, j] -= b; B[j, i] -= b
    return B





def duplicate_network(net, dup_net=1, dup_time=1, link_bus=14, link_x=0.0375):
    

    network = copy.deepcopy(net)
    # 1) extract originals
    orig_buses = list(network.buses)
    orig_lines = list(network.lines)
    orig_nb    = len(orig_buses)

    # 2) duplicate topology
    for r in range(1, dup_net):
        offset = r * orig_nb
        # copy each bus
        for bus in orig_buses:
            new_bus = Bus(bus.idx + offset)
            # clone & re-attach each device
            for dev in bus.devices:
                new_bus.add_device(dev.clone(offset))
            network.buses.append(new_bus)
        # copy each line
        for ln in orig_lines:
            network.lines.append(
                Line(
                    line_idx = ln.line_idx + offset,
                    from_bus = ln.from_bus + offset,
                    to_bus   = ln.to_bus   + offset,
                    x        = ln.x,
                    P_max    = ln.P_max
                )
            )

    # 3) tile all device time-series if requested
    if dup_time > 1:
        for bus in network.buses:
            for dev in bus.devices:
                if hasattr(dev, 'lb') and hasattr(dev, 'ub'):
                    # lb/ub are arrays of length T
                    dev.lb = np.tile(dev.lb, dup_time)
                    dev.ub = np.tile(dev.ub, dup_time)

    # 4) rebuild network susceptance & line-limit data
    nb_new = len(network.buses)
    B = np.zeros((nb_new, nb_new))
    network.flows = []
    for ln in network.lines:
        b = 1.0 / ln.x
        i,j = ln.from_bus, ln.to_bus
        B[i,i] += b;  B[j,j] += b
        B[i,j] -= b;  B[j,i] -= b
        network.flows.append((i, j, b, ln.P_max))

    network.nb   = nb_new
    network.B    = sp.csc_matrix(B)
    network.Bbus = B

    return network

def create_net(gen_df, load_df, bat_df, line_df, slack_bus_idx, T):
    # helper to sort time-series columns
    def sort_ts(cols, prefix):
        return sorted(
            [c for c in cols if c.startswith(prefix)],
            key=lambda c: int(re.search(r'_(\d+)$', c).group(1))
        )

    # time-series bounds columns
    Pmin_g = sort_ts(gen_df.columns,  'P_min^t_')
    Pmax_g = sort_ts(gen_df.columns,  'P_max^t_')
    Pmin_l = sort_ts(load_df.columns, 'P_min^t_')
    Pmax_l = sort_ts(load_df.columns, 'P_max^t_')

    # 1) build your bus objects
    buses_idx = set(gen_df['bus']).union(load_df['bus'], bat_df['bus'],
                                        line_df['from_bus'], line_df['to_bus'])
    # convert to zero-based and create Bus instances
    buses = {int(b)-1: Bus(int(b)-1) for b in buses_idx}

    # 2) add generators
    for _, row in gen_df.iterrows():
        bus_idx = int(row['bus']) - 1
        gen_idx = int(row['id']) - 1
        lb = row[Pmin_g].astype(float).values
        ub = row[Pmax_g].astype(float).values
        P_bounds = np.vstack([lb, ub])
        cost = [float(row['c2']), float(row['c1']), float(row['c0'])]
        gen = Generator(idx=gen_idx, bus_idx=bus_idx,
                        P_bounds=P_bounds, poly_cost=cost)
        buses[bus_idx].add_device(gen)

    # 3) add loads
    for _, row in load_df.iterrows():
        bus_idx = int(row['bus']) - 1
        load_idx = int(row['id']) - 1
        lb = row[Pmin_l].astype(float).values
        ub = row[Pmax_l].astype(float).values
        P_bounds = np.vstack([lb, ub])
        load = Load(idx=load_idx, bus_idx=bus_idx, P_bounds=P_bounds)
        buses[bus_idx].add_device(load)

     # 4) add batteries
    T = len(Pmin_g)
    for _, row in bat_df.iterrows():
        bus_idx = int(row['bus']) - 1
        bat_idx = int(row['id']) - 1
        lb = float(row['P_min'])
        ub = float(row['P_max'])
        P_bounds = np.vstack([[lb]*T, [ub]*T])
        battery = Battery(
            idx=bat_idx,
            bus_idx=bus_idx,
            P_bounds=P_bounds,
            bat_cost=[float(row['c1_d']), float(row['c1_c'])],
            E_init=float(row['E_init']),
            E_bounds=(float(row['E_min']), float(row['E_max'])),
            dT=5,
            eta_c=float(row['eta_c']),
            eta_d=float(row['eta_d'])
        )
        buses[bus_idx].add_device(battery)

    slack_idx = 2
    P_max = 1e6
    P_bounds = np.vstack([[-P_max]*T, [P_max]*T])
    slack = Slack(idx=0,
                  bus_idx=slack_idx,
                  P_bounds = P_bounds,
                  poly_cost=[10, 1, 1]
                  )

    #buses[slack_idx].add_device(slack)    

    # 5) now you have a dict of Bus objects, each with its devices attached
    all_buses = list(buses.values())
    
    # create line objects
    lines = []
    for _, row in line_df.iterrows():
        fb = int(row['from_bus']) - 1
        tb = int(row['to_bus'])   - 1
        x  = float(row['x'])
        lines.append(Line(line_idx=int(row['id']),
                          from_bus=fb,
                          to_bus=tb,
                          x=x))
    




    

    # --- 1) Network dupe ---
    net = NetworkDCOPF(lines, all_buses, slack_idx)

    return net

def run_experiment(H, M, net, csv_fname='summary.csv'):
    admm_time = None
    centralized_time = None
    n_iter = None
    status = []

    logging.info(f"\n=== RUN START: H={H}, M={M} ===")

    # ---- ADMM phase ----
    try:
        logging.info("ADMM start")
        t0 = time.time()
        rho = 3.25
        # your call that might throw…
        _, _, _, n_iter = dc_opf_admm(
            net, H,
            rho_bus=rho * np.ones(net.nb),
            max_iter=30000,
            display_dual=False,
            display=False
        )
        admm_time = time.time() - t0
        logging.info(f"ADMM converged after {n_iter} iter (t={admm_time:.3f}s)")
        status.append("ADMM OK")
    except Exception:
        # logs full stack trace to both file & console
        logging.exception("ADMM phase crashed")
        status.append("ADMM CRASH")

    # ---- Centralized phase ----
    try:
        logging.info("Centralized start")
        t1 = time.time()
        _, _ = centralized_dc_opf(net, H)
        centralized_time = time.time() - t1
        logging.info(f"Centralized converged (t={centralized_time:.3f}s)")
        status.append("CENTRALIZED OK")
    except Exception:
        logging.exception("Centralized phase crashed")
        status.append("CENTRALIZED CRASH")

    # --- append summary to CSV ---
    # header: H,M,admm_iters,admm_time,centralized_time,status
    write_header = not os.path.exists(csv_fname)
    with open(csv_fname, 'a', newline='') as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=['H','M','admm_iters','admm_time','centralized_time','status']
        )
        if write_header:
            writer.writeheader()
        writer.writerow({
            'H': H,
            'M': M,
            'admm_iters': n_iter,
            'admm_time': admm_time,
            'centralized_time': centralized_time,
            # join statuses so you can see e.g. "ADMM OK | CENTRALIZED CRASH"
            'status': " | ".join(status)
        })

    logging.info(f"=== RUN END: H={H}, M={M}, status={' | '.join(status)} ===")
    return status


def setup_logging(log_fname='results.log'):
    # Create a logger that writes both to file and to console
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter with timestamp, level and message
    fmt = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    # File handler—always append, so you build up history
    fh = logging.FileHandler(log_fname, mode='a')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler—so you still see things on screen as they run
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logging.info(f"--- Logging started, output -> {os.path.abspath(log_fname)} ---")




def main():
    

    setup_logging('results.log')

    data_dir = os.path.join(os.getcwd(), 'bus30')
    #data_dir = os.path.join(os.getcwd(), 'PSCC2025', 'MP_DC_ESS', 'IWANT2DIE')
    gen_df, load_df, bat_df, line_df, slack_bus_idx, T = read_data(data_dir)
    net = create_net(gen_df, load_df, bat_df, line_df, slack_bus_idx, T)

    # 1) SET UP
    LOG_FILENAME = 'results.log'

    # Configure the root logger to write INFO-level lines to a file, flushed on every write
    logging.basicConfig(
        filename=LOG_FILENAME,
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # ensure each log record is flushed
    for handler in logging.getLogger().handlers:
        handler.flush = handler.stream.flush

    # Optionally, maintain a CSV summary as well
    CSV_FILENAME = 'summary.csv'
    if not os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                'H','M',
                'admm_iters','admm_time',
                'centralized_time',
                'status'
            ])
            writer.writeheader()

    for repeat_simu in range(10):
        H_vals = [5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1]
        M_vals = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        for H in H_vals:
            for M in M_vals:
                dup_time = ceil(H / 24)
                dup_net = duplicate_network(
                    net, dup_net=M,
                    dup_time=dup_time,
                    link_bus=14,
                    link_x=0.0375
                )
                run_experiment(H, M, dup_net, csv_fname='summary.csv')
    

    


    cost_admm = sum(bus.compute_cost(bus.x0) for bus in net.buses)


    
    """ cost_central = 0.0
    for k, sol in device_sols_central.items():
        for dev in all_buses[k].devices:
            cls = dev.__class__.__name__.lower()
            idx = dev.idx
            if cls == 'generator':
                P = sol[f'P_generator_{idx}']
                cost_central += dev.compute_cost(P)
            elif cls == 'load':
                # loads have zero cost
                continue
            elif cls == 'battery':
                P_c = sol[f'P_c_{idx}']
                P_d = sol[f'P_d_{idx}']
                cost_central += dev.compute_cost(P_c, P_d)
            elif cls == 'slack':
                P = sol[f'P_slack_{idx}']
                cost_central += dev.compute_cost(P) """


    # --- 3) Comparison ---
    """ print("\n===== Comparison of ADMM vs Centralized =====\n")

    print(f"\nCost comparison:\n")
    print(f"  ADMM  = {cost_admm:.4f}")
    #print(f"  Central = {cost_central:.4f}")
    #print(f"  Δ       = {cost_admm - cost_central:.4e}\n")
    print(f"\nTime comparison:\n")
    print(f"  nb bus  = {len(duplicated_net.buses)}")
    print(f"  H = {H}")
    print(f"  ADMM  = {dt_admm}")
    print(f"  ADMM threaded  = {dt_admm_threaded}")
    print(f"  centralisé      = {dt_centr}\n") """




    """ print("\nBus active power:\n")
    for k in range(len(all_buses)):
        Pbus_admm = P_bus_admm[k]
        Pbus_cent = device_sols_central[k][f'P_bus_{k}']
        diff = Pbus_admm - Pbus_cent
        print(f"Bus {k}: ADMM P = {Pbus_admm}, Central P = {Pbus_cent}, Diff = {diff}")

    print("\nBus angles:\n")
    for k in range(len(all_buses)):
        ang_adm = theta_admm[k]
        ang_cent = theta_central[k]
        print(f"Bus {k}: ADMM θ = {ang_adm}, Central θ = {ang_cent}, Δθ = {ang_adm - ang_cent}") """
    
    """ print("\nBus and their devices:\n")
    for bus in all_buses:
        print(bus) """

if __name__ == "__main__":
    main()
    
