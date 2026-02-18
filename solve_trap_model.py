#
# TITLE: CTN trap model
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE:
#


import sys, time, copy
import numpy as np
import sympy as sp
import scipy as sc
import matplotlib.pyplot as plt


#
# CLASS: GRID
#

class GRID:

    # fundamental constants
    q = 1.60217663e-19      # electron charge, [C]
    kb = 1.380649e-23       # Boltzmann constant, [m]^2 [kg] [K]^-1 [s]^-2 
    ep0 = 8.854187e-12      # elelctric permittivity of free space, [F] [m]^-1

    def __init__(self):
        # grids
        self.G  = {}    # grid information
        self.R  = []    # radius (node)
        self.M  = []    # material name (element)
        self.EP = []    # electric permittivity (element)
        self.Q  = []    # charging flag (node)
        self.T  = []    # charging density (node)
        self.CB = []    # conduction band barrier height (element)
        self.VB = []    # valence band barrier height (element)
        self.CE = []    # conduction band effective mass (element)
        self.VE = []    # valence band effective mass (element)
        self.PM = []    # poission matrix
        # solutions
        self.V  = []    # electric potential (node)
        self.E  = []    # electric field (element)

    def add_material_parameters(self, mat_para_dic):
        # user input
        self.MAT = mat_para_dic

    def set_plug_cd(self, plug_cd):
        # user input
        self.GEO_CD = plug_cd               # in angstrom

    def set_channel_wordline_para(self, ch_e_affinity, wl_workfunction):
        # user input
        self.CH_E_AFF = ch_e_affinity       # c-silicon channel electron affinity = 4.05 eV
        self.WL_WF = wl_workfunction        # ALD TiN workfunction = 4.65 eV

    def add_dielectric_layer(self, mat_name, thk, dt, trap):
        # user input
        if self.G == {}:
            # starting from wordline side
            self.G['WL']  = self.GEO_CD / 2.0   # R @WL start
            self.G['MAT'] = [mat_name]          # material name
            self.G['T']   = [thk]               # thickness in angstrom
            self.G['dT']  = [dt]                # resolution in angstrom
            self.G['Q']   = [trap]              # trapping flag
        else:
            #
            self.G['MAT'].append(mat_name)
            self.G['T'].append(thk)
            self.G['dT'].append(dt)
            self.G['Q'].append(trap)

    def make_unit_cell_structure(self):
        # starting from channel side
        uc_t   = self.G['T'][::-1]
        uc_dt  = self.G['dT'][::-1]
        uc_mat = self.G['MAT'][::-1]
        uc_q   = self.G['Q'][::-1]
        
        # making R, Q, T (node)
        for index, each_t in enumerate(uc_t):
            #
            start_r = self.G['WL'] - np.sum(uc_t[index:])
            elmts   = int(uc_t[index] / uc_dt[index])
            end_r   = start_r + elmts * uc_dt[index]
            r_array = list(np.arange(start_r, end_r, uc_dt[index]))
            #
            if uc_q[index] == True:
                q_array = [uc_q[index]] * (elmts + 1)
                t_array = [1.0] * (elmts + 1)
            else:
                q_array = [uc_q[index]] * elmts
                t_array = [0.0] * elmts
            #
            self.R += r_array
            self.Q += q_array
            self.T += t_array
        # 
        self.R += [self.G['WL']]
        #
        self.R = np.array(self.R)
        
        # making M (element)
        for index, each_mat in enumerate(uc_mat):
            #
            elmts     = uc_t[index] / uc_dt[index]
            mat_array = [each_mat] * int(elmts)
            #
            self.M += mat_array

        # making EP, CB, VB, CE, VE (element)
        for each_mat in self.M:
            #
            self.EP.append(self.MAT[each_mat]['k'])
            self.CB.append(self.MAT[each_mat]['cb_bh'])
            self.VB.append(self.MAT[each_mat]['vb_bh'])
            self.CE.append(self.MAT[each_mat]['cb_meff'])
            self.VE.append(self.MAT[each_mat]['vb_meff'])

    def make_poission_matrix(self):
        #
        self.R_len = len(self.R)
        # external bias vector
        self.EB = np.zeros(self.R_len)
        # poisson matrix
        self.PM = sc.sparse.dok_matrix((self.R_len, self.R_len))
        #
        for index in range(self.R_len):
            # MIM model, channel electrode
            if index == 0:
                self.PM[index, index] = 1.0
            # MIM model, wordline electrode
            elif index == (self.R_len-1):
                self.PM[index, index] = 1.0
            # MIM model, dielectrics
            else:
                # geometry factor
                R        = self.R[index+0] * 1e-10          # in SI, m
                R_inner  = self.R[index-1] * 1e-10          # in SI, m
                R_outer  = self.R[index+1] * 1e-10          # in SI, m
                dR_inner = (R - R_inner)
                dR_outer = (R_outer - R)
                # electric perimittivity
                EP_inner = self.EP[index-1] * self.ep0      # in SI
                EP_outer = self.EP[index+0] * self.ep0      # in SI
                # electric displacement
                D_inner = EP_inner / dR_inner * ( R - dR_inner / 2.0 ) / R
                D_outer = EP_outer / dR_outer * ( R + dR_outer / 2.0 ) / R
                # poission equation
                self.PM[index, index-1] =             -D_inner / (dR_inner * 0.5 + dR_outer * 0.5)
                self.PM[index, index+0] = +(D_inner + D_outer) / (dR_inner * 0.5 + dR_outer * 0.5)
                self.PM[index, index+1] =             -D_outer / (dR_inner * 0.5 + dR_outer * 0.5)
        # 
        self.PMcsr = self.PM.tocsr()
                
    def set_trap_density(self, trap_density):
        #
        for index, q_flag in enumerate(self.Q):
            # charging region
            if self.Q[index] == True:
                # change trap density
                self.T[index] = trap_density
        #
        self.T = np.array(self.T)

    def display_grid_info(self):
        # grid information
        for each_key in self.G.keys():
            #
            print(each_key, self.G[each_key])

    def display_band_diagram(self):
        # visualization
        fig, ax = plt.subplots(2, 2, figsize=(14,8))
        ax[0,0].plot(self.R, self.V, 'o-')
        ax[0,0].grid(ls=':')
        ax[0,0].set_title('electric potential [V]')
        ax[0,1].plot(self.R[:-1], +np.array(self.CB)-self.V[:-1], 'o-')
        ax[0,1].plot(self.R[:-1], -np.array(self.VB)-1.1-self.V[:-1], 'o-')
        ax[0,1].set_title('band diagram [eV]')
        ax[0,1].grid(ls=':')
        ax[1,0].plot(self.R[:-1], self.E/1e8, 'o-')
        ax[1,0].set_title('electric field [MV/cm]')
        ax[1,0].grid(ls=':')
        ax[1,1].plot(self.R, self.T/1e6, 'o-')
        ax[1,1].set_title('charge density [1/cm^3]')
        ax[1,1].grid(ls=':')
        plt.show()


#
# CLASS: SOLVER
#

class SOLVER(GRID):

    def solve_poission_equation(self, ch_bias, wl_bias):
        # updating external bias vector
        self.EB[0]  = ch_bias
        self.EB[-1] = -wl_bias + (self.CH_E_AFF - self.WL_WF)            # Fermi line align
        # solve poisson equation
        self.V = sc.sparse.linalg.spsolve(self.PMcsr, self.EB + self.q * self.T)
        # electric field
        self.E = -(self.V[1:] - self.V[:-1]) / (self.R[1:] * 1e-10 - self.R[:-1] * 1e-10)




#
# MAIN
#

mat_para_dic = {}
mat_para_dic['SIO2']  = {'no':10, 'k':3.9,  'eg':8.9, 'cb_bh':3.0, 'vb_bh':4.4, 'cb_meff':0.50, 'vb_meff':0.60}
mat_para_dic['SI3N4'] = {'no':11, 'k':7.5,  'eg':5.0, 'cb_bh':2.0, 'vb_bh':1.5, 'cb_meff':0.24, 'vb_meff':0.30} 
mat_para_dic['AL2O3'] = {'no':12, 'k':9.0,  'eg':8.7, 'cb_bh':2.7, 'vb_bh':4.8, 'cb_meff':0.35, 'vb_meff':0.45}
mat_para_dic['HFO2']  = {'no':13, 'k':23.0, 'eg':4.5, 'cb_bh':1.5, 'vb_bh':1.9, 'cb_meff':0.17, 'vb_meff':0.22}
mat_para_dic['ZRO2']  = {'no':14, 'k':18.0, 'eg':5.0, 'cb_bh':1.4, 'vb_bh':2.2, 'cb_meff':0.15, 'vb_meff':0.25}

grid_solver = SOLVER()
grid_solver.add_material_parameters(mat_para_dic=mat_para_dic)
grid_solver.set_plug_cd(plug_cd=1200)                                               # in angstrom
grid_solver.set_channel_wordline_para(ch_e_affinity=4.05, wl_workfunction=4.65)     # in eV
grid_solver.add_dielectric_layer(mat_name='AL2O3', thk=25.0, dt=1.0, trap=False)    # in angstrom, BOX
grid_solver.add_dielectric_layer(mat_name='SIO2',  thk=68.0, dt=1.0, trap=False)    # in angstrom, BOX
grid_solver.add_dielectric_layer(mat_name='SI3N4', thk=44.0, dt=1.0, trap=True)     # in angstrom, CTN
grid_solver.add_dielectric_layer(mat_name='SIO2',  thk=49.0, dt=1.0, trap=False)    # in angstrom, TOX
grid_solver.make_unit_cell_structure()
grid_solver.make_poission_matrix()
grid_solver.set_trap_density(trap_density=-1.0e25)
grid_solver.solve_poission_equation(ch_bias=0.0, wl_bias=0.0)
grid_solver.display_band_diagram()
grid_solver.display_grid_info()


