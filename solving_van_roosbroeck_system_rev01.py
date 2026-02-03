#
# TITLE: solving the van Roosbroeck system numerically
# AUTHOR: Hyunseung Yoo
# PURPOSE: 
# REVISION: 
# REFERENCE: a numerical study of the van Roosbroeck system for semiconductor (SJSU, 2018) 
#

import sys, time, copy
import numpy as np
import scipy as sc
import sympy as sp
import matplotlib.pyplot as plt

#
# CLASS: GRID (finite difference method)
#

class GRID:

    # fundamental constants
    q = 1.60217663e-19      # electron charge, [C]
    kb = 1.380649e-23       # Boltzmann constant, [m]^2 [kg] [K]^-1 [s]^-2 
    ep0 = 8.854187e-12      # elelctric permittivity of free space, [F] [m]^-1
    

    # ===== constructor =====
    def __init__(self):
        # material information
        self.MAT_info = {}
        # radial structure information
        self.R_info = {}
        # vertical structure information
        self.Z_coord = 0.0
        self.Z_info = {}


    # ===== adding material information =====
    def add_material_parameters(self, name, name_no, category, category_no, wf=0.0, k=0.0, n_int=0.0, mu_n=0.0, mu_p=0.0, tau_n=0.0, tau_p=0.0):
        # making name
        if name not in self.MAT_info.keys():
            self.MAT_info[name] = {}
            self.MAT_info[name]['no'] = name_no

        # adding category
        self.MAT_info[name]['category'] = category          # M, I, S
        self.MAT_info[name]['category_no'] = category_no    # float
        # adding wf
        self.MAT_info[name]['wf'] = wf                  # [eV]
        # adding k, ep
        self.MAT_info[name]['k'] = k                    # dimension-less
        self.MAT_info[name]['ep'] = k * self.ep0        # [s]^4 [A]^2 [m]^-3 [kg]^-1
        # adding n_int, mu_n, mu_p, tau_n, tau_p
        self.MAT_info[name]['n_int'] = n_int            # [m]^-3
        self.MAT_info[name]['mu_n'] = mu_n              # [m]^2 [V]^-1 [s]^-1 
        self.MAT_info[name]['mu_p'] = mu_p              # [m]^2 [V]^-1 [s]^-1 
        self.MAT_info[name]['tau_n'] = tau_n            # [s]
        self.MAT_info[name]['tau_p'] = tau_p            # [s]


    # ===== setting spatial resolution =====
    def set_spatial_resolution(self, dr=1.0, dz=1.0):
        # spatial resolution
        self.dR = dr            # [angstrom]
        self.dZ = dz            # [angstrom]


    # ===== adding radial information =====
    def add_radial_composition(self, name, cd, inside, outside):
        # check name
        if name not in self.R_info.keys():
            self.R_info[name] = {}

        # cd -> radius
        self.R_info[name]['CD/2'] = cd / 2.0       # [angstrom]

        # r coordinate : material name (dictionary, elements)
        self.R_info[name]['R:M'] = {}

        # inside, inward
        self.R_info[name]['inward'] = {}
        r_end = self.R_info[name]['CD/2']       # start
        # check user input
        for each_mat_name in inside.keys():
            # check thickness
            each_thk = inside[each_mat_name]    # [angstrom]
            # calculate r_start
            if each_thk != -1:
                r_start = r_end - each_thk      # inward
            else:
                r_start = 0.0                   # center
            # update r_start & r_end
            self.R_info[name]['inward'][each_mat_name] = [r_start, r_end]
            # r coordinate : material name (dictionary)
            for each_r_coordinate in np.arange(r_start, r_end, self.dR):
                if each_r_coordinate not in self.R_info[name]['R:M'].keys():
                    self.R_info[name]['R:M'][each_r_coordinate] = each_mat_name
            # preparing for next mat_name
            r_end = r_start                     # inward

        # outside, outward
        self.R_info[name]['outward'] = {}
        r_start = self.R_info[name]['CD/2']     # start
        # check user input
        for each_mat_name in outside.keys():
            # check thickness
            each_thk = outside[each_mat_name]   # [angstrom]
            # calculate r_end
            r_end = r_start + each_thk          # outward
            # update r_start & r_end
            self.R_info[name]['outward'][each_mat_name] = [r_start, r_end]
            # r coordinate : material name (dictionary)
            for each_r_coordinate in np.arange(r_start, r_end, self.dR):
                if each_r_coordinate not in self.R_info[name]['R:M'].keys():
                    self.R_info[name]['R:M'][each_r_coordinate] = each_mat_name
            # preparing for next mat_name
            r_start = r_end                     # outward

        # r coordinate (elements, float)
        self.R_info[name]['R'] = list(self.R_info[name]['R:M'].keys())
        self.R_info[name]['R'].sort()           # starting from the center

        # material name (elements, string)
        self.R_info[name]['M'] = []     #
        for each_r_coord in self.R_info[name]['R']:
            self.R_info[name]['M'].append(self.R_info[name]['R:M'][each_r_coord])       # user defined string

        # material category (elements, string)
        self.R_info[name]['C'] = []     # elements
        for each_r_mat_name in self.R_info[name]['M']:
            self.R_info[name]['C'].append(self.MAT_info[each_r_mat_name]['category'])       # 'M', 'I', 'S'

        # dielectric constant, electric permittivity (elements, float)
        self.R_info[name]['EP_K'] = []
        self.R_info[name]['EP'] = []
        for each_r_mat_name in self.R_info[name]['M']:
            self.R_info[name]['EP_K'].append(self.MAT_info[each_r_mat_name]['k'])       # float
            self.R_info[name]['EP'].append(self.MAT_info[each_r_mat_name]['ep'])        # float

        # semiconductor parameters (elements, float)
        self.R_info[name]['N_INT'] = []                 # intrinsic concentration
        self.R_info[name]['MU_N'] = []                  # electron mobility
        self.R_info[name]['MU_P'] = []                  # hole mobility
        self.R_info[name]['TAU_N'] = []                 # electron life time
        self.R_info[name]['TAU_P'] = []                 # hole life time
        for each_r_mat_name in self.R_info[name]['M']:
            self.R_info[name]['N_INT'].append(self.MAT_info[each_r_mat_name]['n_int'])      # float
            self.R_info[name]['MU_N'].append(self.MAT_info[each_r_mat_name]['mu_n'])        # float
            self.R_info[name]['MU_P'].append(self.MAT_info[each_r_mat_name]['mu_p'])        # float
            self.R_info[name]['TAU_N'].append(self.MAT_info[each_r_mat_name]['tau_n'])      # float
            self.R_info[name]['TAU_P'].append(self.MAT_info[each_r_mat_name]['tau_p'])      # float
        

    # ===== adding vertical information =====
    def add_vertical_composition(self, z_info_name, height, electrodes, r_info_name):
        # check z name
        if z_info_name not in self.Z_info.keys():
            self.Z_info[z_info_name] = {}

        # r coordinate : material name (dictionary, elements)
        self.Z_info[z_info_name]['Z:M'] = {}

        # check height
        for z_coord in np.arange(self.Z_coord, self.Z_coord+height, self.dZ):
                self.Z_info[z_info_name]['Z:M'][z_coord] = [electrodes, r_info_name]

        # preparing for next z coordinate
        self.Z_coord += height


    # ===== arranging radial & vertical information =====
    def arrange_radial_vertical_composition(self):
        # z coordinate (elements, float, angstrom)
        self.Z_info['Z'] = []
        # r info name (elements, string), ex) 'SP', 'WL_C', WL_E''
        self.Z_info['M'] = []
        # electrode number (elements, string) 
        self.Z_info['E'] = []
        
        # collecting z coordinates
        for each_z_info_name in self.Z_info.keys():
            # except 'Z', 'M', 'E' dictionaries
            if (each_z_info_name != 'Z') and (each_z_info_name != 'M') and (each_z_info_name != 'E'):
                # sweep 'Z:M' dictionary keys
                for each_z_coord in list( self.Z_info[each_z_info_name]['Z:M'].keys() ):
                    # adding z coordinate (elements, float, angstrom)
                    self.Z_info['Z'].append( each_z_coord )
                    # adding r info name (elements, string)
                    self.Z_info['M'].append( self.Z_info[each_z_info_name]['Z:M'][each_z_coord][1] )
                    # adding electrode number (elements, string)
                    self.Z_info['E'].append( self.Z_info[each_z_info_name]['Z:M'][each_z_coord][0] )
        
        # check the size of 2D array (RZ coordinates, for finite difference method)
        self.Z_elmts = self.Z_info['Z']
        self.Z_nodes = self.Z_elmts + [ self.Z_elmts[-1] + self.dZ ]
        self.Z_elmts_ea = len(self.Z_elmts)                             # total number of Z elements
        self.Z_nodes_ea = len(self.Z_nodes)                             # total number of Z nodes
        
        self.R_elmts = self.R_info[self.Z_info['M'][0]]['R']
        self.R_nodes = self.R_elmts + [ self.R_elmts[-1] + self.dR ]
        self.R_elmts_ea = len(self.R_elmts)                             # total number of R elements
        self.R_nodes_ea = len(self.R_nodes)                             # total number of R nodes

        self.RZ_elmts_ea = self.R_elmts_ea * self.Z_elmts_ea            # for sparse matrix
        self.RZ_nodes_ea = self.R_nodes_ea * self.Z_nodes_ea            # for sparse matrix

        # 2D array (RZ_R, RZ_Z, RZ_dR, RZ_dZ, RZ_Ravg, RZ_Zavg, RZ_dRavg, RZ_dZavg) for FDM
        self.RZ_R = np.zeros([self.R_nodes_ea, self.Z_nodes_ea])        # R coordinate (nodes, nodes)
        for each_z in range(self.Z_nodes_ea):
            self.RZ_R[:,each_z] = np.array(self.R_nodes) * 1e-10        # [angstrom] -> [m]

        self.RZ_Z = np.zeros([self.R_nodes_ea, self.Z_nodes_ea])        # Z coordinate (nodes, nodes)
        for each_r in range(self.R_nodes_ea):
            self.RZ_Z[each_r,:] = np.array(self.Z_nodes) * 1e-10        # [angstrom] -> [m]

        self.RZ_dR = self.RZ_R[1:,:] - self.RZ_R[:-1,:]                 # dR (elements, nodes)
        self.RZ_dZ = self.RZ_Z[:,1:] - self.RZ_Z[:,:-1]                 # dZ (nodes, elements)

        self.RZ_Ravg = (self.RZ_R[1:,:] + self.RZ_R[:-1,:]) / 2.0       # Ravg (elements, nodes)
        self.RZ_Zavg = (self.RZ_Z[:,1:] + self.RZ_Z[:,:-1]) / 2.0       # Zavg (nodes, elements)

        self.RZ_dRavg = (self.RZ_dR[:-1,:] + self.RZ_dR[1:,:]) / 2.0    # dRavg (elements-1, nodes)
        self.RZ_dZavg = (self.RZ_dZ[:,:-1] + self.RZ_dZ[:,1:]) / 2.0    # dZavg (nodes, elements-1)

        # 2D array (MAT_no, MIS_no, ELE_no) for FDM
        self.MAT_no = np.zeros([self.R_elmts_ea, self.Z_elmts_ea])      # material no (elements, elements)
        self.MIS_no = np.zeros([self.R_elmts_ea, self.Z_elmts_ea])      # category no (elements, elements)
        self.ELE_no = np.zeros([self.R_elmts_ea, self.Z_elmts_ea])      # electrode no (elements, elements)
        self.MAT_no_info = {}                                           # mat_no : mat_name , ex) {5: 'VOID', 4: 'LINER', 11: 'SI', 0: 'TOX', 1: 'CTN', 2: 'BOX_SIO2', 6: 'ON_SIO2', 3: 'BOX_AL2O3', 10: 'WL'}
        self.MIS_no_info = {}                                           # mis_no : mis_name , ex) {1: 'I', 2: 'S', 0: 'M'}
        
        for each_z_index in range(self.Z_elmts_ea):
            #
            radial_name = self.Z_info['M'][each_z_index]
            #
            for each_r_index in range(self.R_elmts_ea):
                #
                mat_name = self.R_info[radial_name]['M'][each_r_index]
                mat_no = self.MAT_info[mat_name]['no']
                self.MAT_no_info[mat_no] = mat_name
                #
                mis = self.MAT_info[mat_name]['category']
                mis_no = self.MAT_info[mat_name]['category_no']
                self.MIS_no_info[mis_no] = mis
                #
                self.MAT_no[each_r_index, each_z_index] = mat_no        # float
                self.MIS_no[each_r_index, each_z_index] = mis_no        # float

        # 2D array (SW_S, SW_M, ELE_no) for FDM
        self.SW_S = np.zeros([self.R_nodes_ea, self.Z_nodes_ea])        # switch for semiconductor (nodes, nodes)
        self.SW_M = np.zeros([self.R_nodes_ea, self.Z_nodes_ea])        # switch for metal (nodes, nodes)
        self.ELE_no = -np.ones([self.R_nodes_ea, self.Z_nodes_ea])      # electrode number (nodes, nodes)
        self.ELE_no_info = {}                                           # ele_no : ele_name , ex) {0: 'WL', 1: 'WL', 2: 'WL', 3: 'WL', 4: 'WL'...

        for each_z_index in range(self.Z_elmts_ea):
            for each_r_index in range(self.R_elmts_ea):
                #
                mis_no = self.MIS_no[each_r_index, each_z_index]
                mis_name = self.MIS_no_info[mis_no]
                ele_no = self.Z_info['E'][each_z_index]
                # collecting electrodes no & name
                if ele_no != 0:
                    for each_ele_name in ele_no.keys():
                        each_ele_no = ele_no[each_ele_name]
                        self.ELE_no_info[each_ele_no] = each_ele_name
                # 
                if mis_name == 'S':
                    self.SW_S[each_r_index+0, each_z_index+0] = 1.0     # semiconductor
                    self.SW_S[each_r_index+1, each_z_index+0] = 1.0     # semiconductor
                    self.SW_S[each_r_index+0, each_z_index+1] = 1.0     # semiconductor
                    self.SW_S[each_r_index+1, each_z_index+1] = 1.0     # semiconductor
                #
                if mis_name == 'M':
                    self.SW_M[each_r_index+0, each_z_index+0] = 1.0     # metal
                    self.SW_M[each_r_index+1, each_z_index+0] = 1.0     # metal
                    self.SW_M[each_r_index+0, each_z_index+1] = 1.0     # metal
                    self.SW_M[each_r_index+1, each_z_index+1] = 1.0     # metal
                    # check electrode number
                    for each_ele_name in ele_no.keys():
                        each_ele_no = ele_no[each_ele_name]
                        self.ELE_no[each_r_index+0, each_z_index+0] = each_ele_no     # electrode number
                        self.ELE_no[each_r_index+1, each_z_index+0] = each_ele_no     # electrode number
                        self.ELE_no[each_r_index+0, each_z_index+1] = each_ele_no     # electrode number
                        self.ELE_no[each_r_index+1, each_z_index+1] = each_ele_no     # electrode number
     
        # 2D array (EP, MU_N, MU_P) for FDM
        self.EP = np.zeros([self.R_elmts_ea, self.Z_elmts_ea])          # electric permittivity (elements, elements)
        self.MU_N = np.zeros([self.R_elmts_ea, self.Z_elmts_ea])        # electron mobility (elements, elements)
        self.MU_P = np.zeros([self.R_elmts_ea, self.Z_elmts_ea])        # hole mobility (elements, elements)

        for each_z_index in range(self.Z_elmts_ea):
            #
            radial_name = self.Z_info['M'][each_z_index]
            #
            for each_r_index in range(self.R_elmts_ea):
                #
                mat_name = self.R_info[radial_name]['M'][each_r_index]
                #
                mat_no = self.MAT_info[mat_name]['no']
                #
                mis = self.MAT_info[mat_name]['category']
                mis_no = self.MAT_info[mat_name]['category_no']
                self.MIS_no_info[mis_no] = mis
                #
                ep = self.MAT_info[mat_name]['ep']
                #
                mu_n = self.MAT_info[mat_name]['mu_n']
                mu_p = self.MAT_info[mat_name]['mu_p']
                #
                self.MAT_no[each_r_index, each_z_index] = mat_no        # float
                self.MIS_no[each_r_index, each_z_index] = mis_no        # float
                #
                self.EP[each_r_index, each_z_index] = ep                # float
                #
                self.MU_N[each_r_index, each_z_index] = mu_n            # float
                self.MU_P[each_r_index, each_z_index] = mu_p            # float

        self.EP_r_avg_on_Ez = (self.EP[:-1,:] + self.EP[1:,:]) / 2.0    # electric permittivity r avg @Ez (elements-1, elements)
        self.EP_z_avg_on_Er = (self.EP[:,:-1] + self.EP[:,1:]) / 2.0    # electric permittivity z avg @Er (elements, elements-1)

        # visualizations
        if False:
            print('ELE_no: ', set(list(self.ELE_no.reshape(-1))) )
            print('ELE_no_info:', self.ELE_no_info )
            
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(self.SW_S, origin='lower', aspect='equal')
            ax[1].imshow(self.ELE_no, origin='lower', aspect='equal')
            plt.show()


    # ===== setting semiconductor properties =====
    def set_semiconductor_properties(self, doping_density, dopant_type, op_temp, ohmic_contact):
        # semiconductor region (target nodes)
        self.S_TG = []
        r_array, z_array = np.where( self.SW_S == 1.0 )
        for each_point in range(len(r_array)):
            each_node = [ r_array[each_point], z_array[each_point] ]
            self.S_TG.append(each_node)

        # doping (free carrier provider) (nodes, nodes)
        if dopant_type == 'n':
            self.DP = np.where( self.SW_S==1.0, +doping_density, 0.0 )      # [m]^-3
        elif dopant_type == 'p':
            self.DP = np.where( self.SW_S==1.0, -doping_density, 0.0 )      # [m]^-3
        else:
            output_string = 'GRID > set_semiconductor_properties() > invalid dopant_type...'
            print(output_string)
            exit(1)

        # free carrier density (doping effect) (nodes, nodes)
        self.n = ( np.sqrt( self.DP**2 + 4.0 * self.MAT_info['SI']['n_int']**2 ) + self.DP ) / 2.0 * self.SW_S      # [m]^-3
        self.p = ( np.sqrt( self.DP**2 + 4.0 * self.MAT_info['SI']['n_int']**2 ) - self.DP ) / 2.0 * self.SW_S      # [m]^-3

        # thermal voltage (scalar)
        self.T = op_temp + 273.15               # celsius -> kelvin
        self.Vtm = self.kb * self.T / self.q    # [V]

        # built-in voltage (nodes, nodes)
        self.Vbi = self.Vtm * \
                   np.log( ( self.DP + np.sqrt( self.DP**2 + 4.0 * self.MAT_info['SI']['n_int']**2 ) ) / \
                           ( 2.0 * self.MAT_info['SI']['n_int'] ) )                                             # [V]

        # ohmic contact
        for each_ohmic_contact in ohmic_contact.keys():
            # details
            elec_no, elec_z_position = ohmic_contact[each_ohmic_contact]
            
            # updating ELE_no_info
            self.ELE_no_info[elec_no] = each_ohmic_contact

            # updating ELE_no
            if elec_z_position != 0.5:
                # MIS structure
                self.ELE_no[:, elec_z_position] = np.where( self.SW_S[:, elec_z_position] == 1.0, elec_no, -1.0)
            else:
                # MIM structure
                if each_ohmic_contact == 'BL':
                    r_array, z_array = np.where( self.SW_S[:, :int(self.Z_nodes_ea/2.0)] == 1.0 )
                    self.ELE_no[ (r_array, z_array) ] = elec_no
                if each_ohmic_contact == 'SL':
                    r_array, z_array = np.where( self.SW_S[:, int(self.Z_nodes_ea/2.0):] == 1.0 )
                    self.ELE_no[ (r_array, z_array+int(self.Z_nodes_ea/2.0)) ] = elec_no
                    

        # visualizations
        if False:
            print('ELE_no: ', set(list(self.ELE_no.reshape(-1))) )
            print('ELE_no_info:', self.ELE_no_info )
            
            fig, ax = plt.subplots(3, 1)
            ax[0].imshow(self.SW_M, origin='lower', aspect='equal')
            ax[1].imshow(self.SW_S, origin='lower', aspect='equal')
            ax[2].imshow(self.ELE_no, origin='lower', aspect='equal')
            plt.show()


    # ===== making poisson matrix =====
    def make_poission_matrix(self):
        # STEP 0: making dok sparse matrix
        self.PM = sc.sparse.dok_matrix( (self.RZ_nodes_ea, self.RZ_nodes_ea) )

        # STEP 1: processing dirichlet boundary conditions (electrodes)
        for each_elec_no in self.ELE_no_info.keys():
            # finding where elec no points are
            selected_rz_array = np.where( self.ELE_no == each_elec_no)
            selected_r_array, selected_z_array = selected_rz_array
            # each point
            for each_selected_point in range(len(selected_r_array)):
                # selected point
                selected_r = selected_r_array[each_selected_point]
                selected_z = selected_z_array[each_selected_point]
                # sparse matrix index (1D serialization)
                selected_r_z = self.R_nodes_ea * (selected_z) + selected_r
                # sparse matrix elements (1D serialization)
                self.PM[selected_r_z, selected_r_z] = +1.0

        # STEP 2-1: processing neumann boundary conditions (Z boundaries)
        for selected_z in [0, self.Z_nodes_ea-1]:
            for selected_r in range(self.R_nodes_ea):
                # sparse matrix index (1D serialization)
                selected_r_z = self.R_nodes_ea * (selected_z) + selected_r
                # except dirichlet boundary conditions
                if self.PM[selected_r_z, selected_r_z] != 1.0:
                    # bottom
                    if selected_z == 0:
                        # sparse matrix index (1D serialization)
                        selected_r_zp1 = self.R_nodes_ea * (selected_z+1) + selected_r
                        # sparse matrix elements (1D serialization)
                        self.PM[selected_r_z, selected_r_z  ] = +1.0
                        self.PM[selected_r_z, selected_r_zp1] = -1.0
                    # top
                    if selected_z == (self.Z_nodes_ea-1):
                        # sparse matrix index (1D serialization)
                        selected_r_zm1 = self.R_nodes_ea * (selected_z-1) + selected_r
                        # sparse matrix elements (1D serialization)
                        self.PM[selected_r_z, selected_r_z  ] = +1.0
                        self.PM[selected_r_z, selected_r_zm1] = -1.0

        # STEP 2-2: processing neumann boundary conditions (R boundaries)
        for selected_r in [0, self.R_nodes_ea-1]:
            for selected_z in range(self.Z_nodes_ea):
                # sparse matrix index (1D serialization)
                selected_r_z = self.R_nodes_ea * (selected_z) + selected_r
                # except dirichlet boundary conditions
                if self.PM[selected_r_z, selected_r_z] != 1.0:
                    # inside
                    if selected_r == 0:
                        # sparse matrix index (1D serialization)
                        selected_rp1_z = self.R_nodes_ea * (selected_z) + (selected_r+1)
                        # sparse matrix elements (1D serialization)
                        self.PM[selected_r_z, selected_r_z  ] = +1.0
                        self.PM[selected_r_z, selected_rp1_z] = -1.0
                    # outside
                    if selected_r == (self.R_nodes_ea-1):
                        # sparse matrix index (1D serialization)
                        selected_rm1_z = self.R_nodes_ea * (selected_z) + (selected_r-1)
                        # sparse matrix elements (1D serialization)
                        self.PM[selected_r_z, selected_r_z  ] = +1.0
                        self.PM[selected_r_z, selected_rm1_z] = -1.0

        # STEP 3: poisson finite difference equations
        for selected_r in range(1, self.R_nodes_ea-1):
            for selected_z in range(1, self.Z_nodes_ea-1):
                # sparse matrix index (1D serialization)
                selected_r_z   = self.R_nodes_ea * (selected_z+0) + (selected_r+0)
                selected_rp1_z = self.R_nodes_ea * (selected_z+0) + (selected_r+1)
                selected_rm1_z = self.R_nodes_ea * (selected_z+0) + (selected_r-1)
                selected_r_zp1 = self.R_nodes_ea * (selected_z+1) + (selected_r+0)
                selected_r_zm1 = self.R_nodes_ea * (selected_z-1) + (selected_r+0)
                # except dirichlet boundary conditions & neumann boundary conditions
                if self.PM[selected_r_z, selected_r_z] != 1.0:
                    # sparse matrix constants (r-1, z)
                    a_rm1_z  = ( self.RZ_Ravg[selected_r-1, selected_z+0] / self.RZ_R[selected_r, selected_z] )         # geometry factor
                    a_rm1_z *= ( self.EP_z_avg_on_Er[selected_r-1, selected_z-1] )                                      # electric permittivity
                    a_rm1_z /= ( self.RZ_dR[selected_r-1, selected_z+0] * self.RZ_dRavg[selected_r-1, selected_z])
                    # sparse matrix constants (r+1, z)
                    a_rp1_z  = ( self.RZ_Ravg[selected_r+0, selected_z+0] / self.RZ_R[selected_r, selected_z] )         # geometry factor
                    a_rp1_z *= ( self.EP_z_avg_on_Er[selected_r+0, selected_z-1] )                                      # electric permittivity
                    a_rp1_z /= ( self.RZ_dR[selected_r+0, selected_z+0] * self.RZ_dRavg[selected_r-1, selected_z])
                    # sparse matrix constants (r, z-1)
                    a_r_zm1  = ( self.EP_r_avg_on_Ez[selected_r-1, selected_z-1] )                                      # electric permittivity
                    a_r_zm1 /= ( self.RZ_dZ[selected_r+0, selected_z-1] * self.RZ_dZavg[selected_r+0, selected_z-1])
                    # sparse matrix constants (r, z+1)
                    a_r_zp1  = ( self.EP_r_avg_on_Ez[selected_r-1, selected_z+0] )                                      # electric permittivity
                    a_r_zp1 /= ( self.RZ_dZ[selected_r+0, selected_z+0] * self.RZ_dZavg[selected_r+0, selected_z-1])
                    # sparse matrix elements (1D serialization)
                    self.PM[selected_r_z, selected_r_z  ] = +a_rm1_z + a_rp1_z + a_r_zm1 + a_r_zp1
                    self.PM[selected_r_z, selected_rp1_z] = -a_rp1_z
                    self.PM[selected_r_z, selected_rm1_z] = -a_rm1_z
                    self.PM[selected_r_z, selected_r_zp1] = -a_r_zp1
                    self.PM[selected_r_z, selected_r_zm1] = -a_r_zm1
        
        # STEP 4: CSR format conversion
        self.PMcsr = self.PM.tocsr()
        

    # ===== debugging =====
    def debugging(self):
        # add_material_parameters()
        output_string = 'name/cat./wf/k/n_int/mu_n/mu_p/tau_n/tau_p'
        print(output_string)
        #
        for each_name in self.MAT_info.keys():
                #
                each_category = self.MAT_info[each_name]['category']
                each_wf = self.MAT_info[each_name]['wf']
                each_k = self.MAT_info[each_name]['k']
                each_n_int = self.MAT_info[each_name]['n_int']
                each_mu_n = self.MAT_info[each_name]['mu_n']
                each_mu_p = self.MAT_info[each_name]['mu_p']
                each_tau_n = self.MAT_info[each_name]['tau_n']
                each_tau_p = self.MAT_info[each_name]['tau_p']
                #
                output_string = ' = %s / %s / %.2f / %.2f /%.1e/%.1e/%.1e/%.1e/%.1e' % \
                                (each_category, each_name, each_wf, each_k, each_n_int, each_mu_n, each_mu_p, each_tau_n, each_tau_p)
                print(output_string)

                

#
# CLASS: SOLVER (using sparse matrix), inheritance from GRID class
#

class SOLVER(GRID):

    # ===== making external bias vector =====
    def make_external_bias_vector(self, external_bias_input):
        # external bias vector
        self.EB = np.zeros(self.RZ_nodes_ea)
        #
        for each_ele_no in external_bias_input.keys():
            # finding points
            r_array, z_array = np.where( self.ELE_no == each_ele_no )
            # sweep
            for each_point in range(len(r_array)):
                # each point
                each_r = r_array[each_point]
                each_z = z_array[each_point]
                # 1D serialization
                selected_r_z = self.R_nodes_ea * (each_z) + (each_r)
                # BL, SL contact (+ bulit-in potential)
                if (self.ELE_no_info[each_ele_no] == 'BL') or (self.ELE_no_info[each_ele_no] == 'SL'):
                    #                          built-in potential     +        external bias
                    self.EB[selected_r_z] =  self.Vbi[each_r, each_z] + external_bias_input[each_ele_no]
                elif (self.ELE_no_info[each_ele_no] == 'WL'):
                    self.EB[selected_r_z] =  external_bias_input[each_ele_no]

        # visulization
        if False:
            self.EB2 = np.resize( self.EB, (self.Z_nodes_ea, self.R_nodes_ea) ).T
            plt.imshow(self.EB2, origin='lower', aspect='equal')
            plt.show()
            

    # ===== solving poission equation =====
    def solve_poission_equation(self, fig_output=False):
        # scipy sparse matrix solver
        self.V = sc.sparse.linalg.spsolve(self.PMcsr, self.EB)

        # post analysis
        self.V2 = np.resize(self.V, (self.Z_nodes_ea, self.R_nodes_ea)).T
        self.Er = (self.V2[1:,:] - self.V2[:-1,:]) / self.RZ_dR
        self.Ez = (self.V2[:,1:] - self.V2[:,:-1]) / self.RZ_dZ
        self.E  = np.sqrt( self.Er[:,:-1]**2 + self.Ez[:-1,:]**2 )

        # visualization
        if fig_output != False:
            fig, ax = plt.subplots(3, 2, figsize=(16, 10))
            ax[0,0].imshow(self.MAT_no, origin='lower', aspect='equal')
            ax[0,0].set_title('geometry')
            ax[0,0].set_ylabel('R')
            ax[0,1].imshow(self.V2, origin='lower', aspect='equal')
            ax[0,1].set_title('V potential')
            ax[0,1].set_ylabel('R')
            ax[1,0].imshow(self.Er, origin='lower', aspect='equal')
            ax[1,0].set_title('Er field')
            ax[1,0].set_ylabel('R')
            ax[1,1].imshow(self.Ez, origin='lower', aspect='equal')
            ax[1,1].set_title('Ez field')
            ax[1,1].set_ylabel('R')
            ax[2,0].imshow(self.E, origin='lower', aspect='equal')
            ax[2,0].set_title('|E| field mag.')
            ax[2,0].set_ylabel('R')
            ax[2,0].set_xlabel('Z')
            ax[2,1].imshow(self.ELE_no, origin='lower', aspect='equal')
            ax[2,1].set_title('electrode no.')
            ax[2,1].set_ylabel('R')
            ax[2,1].set_xlabel('Z')
            plt.savefig(fig_output)
            plt.show()
            plt.close()



#
# MAIN
#

grid_solver = SOLVER()

# material parameters
grid_solver.add_material_parameters(name='WL', name_no=10, category='M', category_no=0, wf=4.5)
grid_solver.add_material_parameters(name='TOX', name_no=0, category='I', category_no=1, k=4.9)
grid_solver.add_material_parameters(name='CTN', name_no=1, category='I', category_no=1, k=7.5)
grid_solver.add_material_parameters(name='BOX_SIO2', name_no=2, category='I', category_no=1, k=5.0)
grid_solver.add_material_parameters(name='BOX_AL2O3', name_no=3, category='I', category_no=1, k=9.0)
grid_solver.add_material_parameters(name='LINER', name_no=4, category='I', category_no=1, k=3.9)
grid_solver.add_material_parameters(name='VOID', name_no=5, category='I', category_no=1, k=1.0)
grid_solver.add_material_parameters(name='ON_SIO2', name_no=6, category='I', category_no=1, k=3.9)
grid_solver.add_material_parameters(name='SI', name_no=11, category='S', category_no=2, k=11.7, n_int=1.5e16, mu_n=0.14, mu_p=0.045, tau_n=1e-6, tau_p=1e-5)

# spatial resolution
grid_solver.set_spatial_resolution(dr=5.0, dz=5.0)     # angstrom

# radial composition (from CD)
inward_thk  = {'BOX_SIO2':70.0, 'CTN':50.0, 'TOX':50.0, 'SI':70.0, 'LINER':70.0, 'VOID':-1}         # angstrom
outward_thk_wl_c = {'BOX_AL2O3':30.0, 'WL':100.0}                                                   # angstrom
outward_thk_wl_e = {'BOX_AL2O3':130.0}                                                              # angstrom
outward_thk_sp = {'ON_SIO2':130.0}                                                                  # angstrom
grid_solver.add_radial_composition(name='WL_C', cd=1300, inside=inward_thk, outside=outward_thk_wl_c)      # angstrom
grid_solver.add_radial_composition(name='WL_E', cd=1300, inside=inward_thk, outside=outward_thk_wl_e)      # angstrom
grid_solver.add_radial_composition(name='SP', cd=1300, inside=inward_thk, outside=outward_thk_sp)          # angstrom

# vertical composition (from bottom)
total_wl_number = 7

for each_wl_number in range(total_wl_number):
    each_wl_name = 'WL%03i' % each_wl_number

    grid_solver.add_vertical_composition(z_info_name=each_wl_name+'_SP_B', height=120.0, electrodes=0.0,                   r_info_name='SP')
    grid_solver.add_vertical_composition(z_info_name=each_wl_name+'_E_B',  height=30.0,  electrodes=0.0,                   r_info_name='WL_E')
    grid_solver.add_vertical_composition(z_info_name=each_wl_name+'_C',    height=120.0, electrodes={'WL':each_wl_number}, r_info_name='WL_C')
    grid_solver.add_vertical_composition(z_info_name=each_wl_name+'_E_T',  height=30.0,  electrodes=0.0,                   r_info_name='WL_E')
    grid_solver.add_vertical_composition(z_info_name=each_wl_name+'_SP_T', height=120.0, electrodes=0.0,                   r_info_name='SP')

# arrange radial and vertical composition
grid_solver.arrange_radial_vertical_composition()

# set semiconductor properties
bl_sl_contact = {'BL':[10001, 0],   'SL':[10002, -1] }     # MIS structure, elec_name : [elec_no, z_position]
bl_sl_contact = {'BL':[10001, 0.5], 'SL':[10002, 0.5]}     # MIM structure, elec_name : [elec_no, 0.5]
grid_solver.set_semiconductor_properties(doping_density=1e18, dopant_type='n', op_temp=25.0, ohmic_contact=bl_sl_contact)

# make poisson matrix
grid_solver.make_poission_matrix()

# make external bias vector
bl_sl_wl_ext_bias = {10001:0.0, 10002:0.0, 0:1.0, 1:1.0, 2:1.0, 3:1.0, 4:1.0, 5:1.0, 6:1.0}
grid_solver.make_external_bias_vector(external_bias_input=bl_sl_wl_ext_bias)

# solve poission equation
grid_solver.solve_poission_equation(fig_output='poisson_solver_output.png')

# debugging
#grid.debugging()
