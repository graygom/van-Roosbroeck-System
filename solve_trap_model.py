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
    me = 9.1093837e-31      # electron mass, [kg]
    h = 6.62607015e-34      # Planck constant, [m]^2 [kg] [s]^-1
    hbar = h / (2.0*np.pi)

    def __init__(self):
        # grids
        self.G  = {}    # grid information
        self.R  = []    # radius (node)
        self.M  = []    # material name (element)
        self.K  = []    # dielectric constant (element)
        self.Q  = []    # charging flag (node)
        self.T  = []    # charging density (node)
        self.TT = []    # TOX tunneling flag (node)
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

    def add_dielectric_layer(self, mat_name, thk, dt, trap, tox):
        # user input
        if self.G == {}:
            # starting from wordline side
            self.G['WL']  = self.GEO_CD / 2.0   # R @WL start
            self.G['MAT'] = [mat_name]          # material name
            self.G['T']   = [thk]               # thickness in angstrom
            self.G['dT']  = [dt]                # resolution in angstrom
            self.G['Q']   = [trap]              # trapping flag
            self.G['TT']  = [tox]               # TOX tunneling flag
        else:
            #
            self.G['MAT'].append(mat_name)
            self.G['T'].append(thk)
            self.G['dT'].append(dt)
            self.G['Q'].append(trap)
            self.G['TT'].append(tox)

    def make_unit_cell_structure(self, tox_n_profile):
        # starting from channel side
        uc_t   = self.G['T'][::-1]
        uc_dt  = self.G['dT'][::-1]
        uc_mat = self.G['MAT'][::-1]
        uc_q   = self.G['Q'][::-1]
        uc_tt  = self.G['TT'][::-1]
        
        # making R, Q, T, TT (node)
        for index, each_t in enumerate(uc_t):
            #
            start_r = self.G['WL'] - np.sum(uc_t[index:])
            elmts   = int(uc_t[index] / uc_dt[index])
            end_r   = start_r + elmts * uc_dt[index]
            r_array = list(np.arange(start_r, end_r, uc_dt[index]))
            #
            if uc_q[index] == True:
                q_array  = [uc_q[index]] * (elmts + 1)
                t_array  = [1.0] * (elmts + 1)
                tt_array = [uc_tt[index]] * elmts
            elif uc_tt[index] == True:
                q_array  = [uc_q[index]] * elmts
                t_array  = [0.0] * elmts
                tt_array = [uc_tt[index]] * (elmts + 1)
            else:
                q_array = [uc_q[index]] * elmts
                t_array = [0.0] * elmts
                tt_array = [uc_tt[index]] * elmts
            #
            self.R  += r_array
            self.Q  += q_array
            self.T  += t_array
            self.TT += tt_array
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

        # making K, CB, VB, CE, VE (element)
        for each_mat in self.M:
            #
            self.K.append(self.MAT[each_mat]['k'])
            self.CB.append(self.MAT[each_mat]['cb_bh'])
            self.VB.append(self.MAT[each_mat]['vb_bh'])
            self.CE.append(self.MAT[each_mat]['cb_meff'])
            self.VE.append(self.MAT[each_mat]['vb_meff'])

        # TOX N profile (band engineering, linear interpolation)
        uc_tox_n_peak_pos  = int(uc_t[0]*tox_n_profile['tox_peak']['pos'])
        uc_tox_n_ch_side   = tox_n_profile['ch_side']['n']
        uc_tox_n_peak      = tox_n_profile['tox_peak']['n']
        uc_tox_n_ctn_side  = tox_n_profile['ctn_side']['n']
        uc_tox_n_diff_1    = uc_tox_n_peak - uc_tox_n_ch_side
        uc_tox_n_diff_2    = uc_tox_n_ctn_side - uc_tox_n_peak
        uc_tox_pos = np.array( list( range( int(uc_t[0]/uc_dt[0]) ) ), dtype=int )
        uc_tox_n   = np.zeros( int(uc_t[0]/uc_dt[0]) )
        uc_tox_n   = np.where( uc_tox_pos <= uc_tox_n_peak_pos, \
                               uc_tox_n_ch_side + uc_tox_n_diff_1 * uc_tox_pos / uc_tox_n_peak_pos, 0.0 )
        uc_tox_n  += np.where( uc_tox_pos > uc_tox_n_peak_pos, \
                               uc_tox_n_peak + uc_tox_n_diff_2 * (uc_tox_pos - uc_tox_n_peak_pos) / (uc_tox_pos[-1] - uc_tox_n_peak_pos), 0.0 ) 

        for index in range( int(uc_t[0]/uc_dt[0]) ):
            #
            self.K[index]  = self.MAT['SIO2']['k']       * (1.0 - uc_tox_n[index]) + self.MAT['SI3N4']['k']       * uc_tox_n[index]
            self.CB[index] = self.MAT['SIO2']['cb_bh']   * (1.0 - uc_tox_n[index]) + self.MAT['SI3N4']['cb_bh']   * uc_tox_n[index]
            self.VB[index] = self.MAT['SIO2']['vb_bh']   * (1.0 - uc_tox_n[index]) + self.MAT['SI3N4']['vb_bh']   * uc_tox_n[index]
            self.CE[index] = self.MAT['SIO2']['cb_meff'] * (1.0 - uc_tox_n[index]) + self.MAT['SI3N4']['cb_meff'] * uc_tox_n[index]
            self.VE[index] = self.MAT['SIO2']['vb_meff'] * (1.0 - uc_tox_n[index]) + self.MAT['SI3N4']['vb_meff'] * uc_tox_n[index]

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
                EP_inner = self.K[index-1] * self.ep0       # in SI
                EP_outer = self.K[index+0] * self.ep0       # in SI
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

    def display_band_diagram(self, output_filename):
        # preprocessing 1
        r_node = self.R
        v_node = self.V
        t_node = self.T
        # preprocessing 2
        r_elmt = self.R[:-1]
        v_elmt = self.V[:-1]
        e_elmt = self.E
        k_elmt = self.K
        # preprocessing 3
        cb_elmt = +np.array(self.CB) - v_elmt
        vb_elmt = -np.array(self.VB) - v_elmt - 1.1
        # preprocessing 4
        dr_ch = r_node[1]  - r_node[0]
        dr_wl = r_node[-1] - r_node[-2]
        r_elmt_ext  = list(np.arange(r_node[0]-3*dr_ch, r_node[0], dr_ch)) + list(r_elmt) + \
                      list(np.arange(r_node[-1]+dr_wl, r_node[-1]+4*dr_wl, dr_wl))
        cb_elmt_ext = list([-self.EB[0]]*3) + list(cb_elmt) + list([-self.EB[-1]+(self.CH_E_AFF-self.WL_WF)]*3)
        vb_elmt_ext = list([-self.EB[0]-1.1]*3) + list(vb_elmt) + list([-self.EB[-1]+(self.CH_E_AFF-self.WL_WF)]*3)
        # preprocessing 5
        r_elmt_ext2  = [r_elmt_ext[0], r_elmt_ext[-1]]
        cb_tunneling = [0.0, 0.0]
        vb_tunneling = [-1.1, -1.1]
        # visualization
        fig, ax = plt.subplots(2, 3, figsize=(18,7))
        ax[0,0].plot(r_node, v_node, 'o-')
        ax[0,0].grid(ls=':')
        ax[0,0].set_ylabel('electric potential [V]')
        ax[1,0].plot(r_elmt, e_elmt/1e8, 'o-')
        ax[1,0].set_ylabel('electric field [MV/cm]')
        ax[1,0].set_xlabel('radius [Angstrom]')
        ax[1,0].grid(ls=':')
        ax[0,1].plot(r_elmt_ext, cb_elmt_ext, 'o-')
        ax[0,1].plot(r_elmt_ext2, cb_tunneling, ':k')
        ax[0,1].set_ylabel('conduction band diagram [eV]')
        ax[0,1].grid(ls=':')
        ax[1,1].plot(r_elmt_ext, vb_elmt_ext, 'o-')
        ax[1,1].plot(r_elmt_ext2, vb_tunneling, ':k')
        ax[1,1].set_ylabel('valence band diagram [eV]')
        ax[1,1].set_xlabel('radius [Angstrom]')
        ax[1,1].grid(ls=':')
        ax[0,2].plot(r_elmt, k_elmt, 'o-')
        ax[0,2].set_ylabel('dielectric constant')
        ax[0,2].grid(ls=':')
        ax[1,2].plot(r_node, t_node/1e6, 'o-')
        ax[1,2].set_ylabel('charge density [1/cm^3]')
        ax[1,2].set_xlabel('radius [Angstrom]')
        ax[1,2].grid(ls=':')
        plt.savefig(output_filename)
        plt.show()
        plt.close()


#
# CLASS: SOLVER
#

class SOLVER(GRID):

    def calculate_vt_shift(self):
        # starting from neutral state
        self.VTS = 0.0
        # check trap layer
        for each_index in range(len(self.Q)):
            # trap layer
            if self.Q[each_index] == True:
                # geometry 1
                each_r = self.R[each_index] * 1e-10
                each_dr = self.R[each_index+1] * 1e-10 - self.R[each_index] * 1e-10
                # geometry 2
                each_area = 2.0 * np.pi * each_r
                each_volume = each_area * each_dr
                # net charge
                each_charge = self.q * self.T[each_index] * each_volume
                # from net charge to wordline capacitance inverse
                k_array  = np.array(self.K[each_index:])
                r_array  = np.array(self.R[each_index:-1])
                dr_array = np.array(self.R[(each_index+1):]) - np.array(self.R[each_index:-1])
                c_inv_array = dr_array / (self.ep0 * k_array) / (2.0 * np.pi * r_array)
                C_inv = np.sum(c_inv_array)
                # accumulated charge on channel: dV = dQ * C inverse
                self.VTS += -each_charge * C_inv
        #
        return self.VTS

    def solve_poission_equation(self, ch_bias, wl_bias):
        # updating external bias vector
        self.EB[0]  = ch_bias
        self.EB[-1] = wl_bias + (self.CH_E_AFF - self.WL_WF)            # Fermi line align
        # solve poisson equation
        self.V = sc.sparse.linalg.spsolve(self.PMcsr, self.EB + self.q * self.T)
        # electric field
        self.E = -(self.V[1:] - self.V[:-1]) / (self.R[1:] * 1e-10 - self.R[:-1] * 1e-10)

    def calculate_tox_tunneling_probability(self):
        # preprocessing 1
        r_node = self.R
        v_node = self.V
        # preprocessing 2
        r_elmt = self.R[:-1]
        v_elmt = self.V[:-1]
        # preprocessing 3
        cb_elmt = np.array(self.CB) - v_elmt
        vb_elmt = np.array(self.VB) + v_elmt
        # 
        self.TOX_TNL_PROB_CB = 0.0
        self.TOX_TNL_PROB_CB_index = 0.0
        self.TOX_TNL_PROB_VB = 0.0
        self.TOX_TNL_PROB_VB_index = 0.0
        # WKB approximation
        wkb_approx_const = 2.0 / self.hbar
        #
        for index in range(len(self.TT)):
            # TOX only
            if (self.TT[index] == True) and (self.Q[index] == False):
                # distance
                dr = ( self.R[index+1] - self.R[index] ) * 1e-10
                # conduction band electron Tunneling only
                if cb_elmt[index] > 0.0:
                    self.TOX_TNL_PROB_CB += np.sqrt( 2.0 * self.CE[index] * self.me * self.q * cb_elmt[index] ) * dr
                    self.TOX_TNL_PROB_CB_index = index
                # valence band electron Tunneling only
                if vb_elmt[index] > 0.0:
                    self.TOX_TNL_PROB_VB += np.sqrt( 2.0 * self.VE[index] * self.me * self.q * vb_elmt[index] ) * dr
                    self.TOX_TNL_PROB_VB_index = index
        #
        self.TOX_TNL_PROB_CB = np.exp( -2.0 / self.hbar * self.TOX_TNL_PROB_CB)
        self.TOX_TNL_PROB_VB = np.exp( -2.0 / self.hbar * self.TOX_TNL_PROB_VB)
                    
        # return
        return [self.TOX_TNL_PROB_CB, self.TOX_TNL_PROB_CB_index, self.TOX_TNL_PROB_VB, self.TOX_TNL_PROB_VB_index]



#
# MAIN
#

mat_para_dic = {}
mat_para_dic['SIO2']  = {'no':10, 'k':3.9,  'eg':8.9, 'cb_bh':3.0, 'vb_bh':4.4, 'cb_meff':0.50, 'vb_meff':0.60}
mat_para_dic['SI3N4'] = {'no':11, 'k':7.5,  'eg':5.0, 'cb_bh':2.0, 'vb_bh':1.5, 'cb_meff':0.24, 'vb_meff':0.30} 
mat_para_dic['AL2O3'] = {'no':12, 'k':9.0,  'eg':8.7, 'cb_bh':2.7, 'vb_bh':4.8, 'cb_meff':0.40, 'vb_meff':0.45}
mat_para_dic['HFO2']  = {'no':13, 'k':23.0, 'eg':4.5, 'cb_bh':1.5, 'vb_bh':1.9, 'cb_meff':0.17, 'vb_meff':0.22}
mat_para_dic['ZRO2']  = {'no':14, 'k':18.0, 'eg':5.0, 'cb_bh':1.4, 'vb_bh':2.2, 'cb_meff':0.15, 'vb_meff':0.25}

tox_n_profile = {}
tox_n_profile['ch_side']  = {'n':0.05, 'pos':0.00}      # portion
tox_n_profile['tox_peak'] = {'n':0.25, 'pos':0.50}      # portion
tox_n_profile['ctn_side'] = {'n':0.15, 'pos':1.00}      # portion

grid_solver = SOLVER()
grid_solver.add_material_parameters(mat_para_dic=mat_para_dic)
grid_solver.set_plug_cd(plug_cd=1200)                                                           # in angstrom
grid_solver.set_channel_wordline_para(ch_e_affinity=4.05, wl_workfunction=4.65)                 # in eV
grid_solver.add_dielectric_layer(mat_name='AL2O3', thk=25.0, dt=1.0, trap=False, tox=False)     # in angstrom, BOX
grid_solver.add_dielectric_layer(mat_name='SIO2',  thk=68.0, dt=1.0, trap=False, tox=False)     # in angstrom, BOX
grid_solver.add_dielectric_layer(mat_name='SI3N4', thk=44.0, dt=1.0, trap=True,  tox=False)     # in angstrom, CTN
grid_solver.add_dielectric_layer(mat_name='SIO2',  thk=49.0, dt=1.0, trap=False, tox=True)      # in angstrom, TOX
grid_solver.make_unit_cell_structure(tox_n_profile=tox_n_profile)
grid_solver.display_grid_info()

grid_solver.make_poission_matrix()
grid_solver.set_trap_density(trap_density=+0.0e25)
vt_shift = grid_solver.calculate_vt_shift()

vg_bias_array = np.arange(5.0, 18.1, 0.5)
tox_cb_tunl_prob = []
tox_cb_tunl_prob_index = []
tox_vb_tunl_prob = []
tox_vb_tunl_prob_index = []

for each_vg_bias in vg_bias_array:
    grid_solver.solve_poission_equation(ch_bias=0.0, wl_bias=each_vg_bias)
    cb_tunl_prob, cb_tunl_index, vb_tunl_prob, vb_tunl_index = grid_solver.calculate_tox_tunneling_probability()
    tox_cb_tunl_prob.append(cb_tunl_prob)
    tox_vb_tunl_prob.append(vb_tunl_prob)
    tox_cb_tunl_prob_index.append(cb_tunl_index)
    tox_vb_tunl_prob_index.append(vb_tunl_index)
    print(each_vg_bias, cb_tunl_prob, vb_tunl_prob, cb_tunl_index, vb_tunl_index)

plt.semilogy(vg_bias_array, tox_cb_tunl_prob, 'o-')
plt.grid(ls=':')
plt.show()

#grid_solver.display_band_diagram(output_filename='ctn_trap_model_test.pdf')







