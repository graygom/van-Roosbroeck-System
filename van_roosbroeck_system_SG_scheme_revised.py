#
# TITLE: solving the van Roosbroeck system numerically
# AUTHOR: Hyunseung Yoo
# PURPOSE: 
# REVISION: ohmic contacts & edge boundary conditions revision @Feb. 2026
# REFERENCE: a numerical study of the van Roosbroeck system for semiconductor (SJSU, 2018) 
#

import sys, time, copy, psutil, platform, cpuinfo
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
        # poisson equation (1813) solution (for SOLVER class)
        self.V1 = []        # electric potential 1D (sparse matrix solution)
        self.V2 = []        # electric potential 2D (visualization)
        self.E  = []        # electric field magnitude 2D (visualization)
        self.Er = []        # electric field r direction 2D (visualization)
        self.Ez = []        # electric field z direction 2D (visualization)
        self.EB = []        # external bias vector 
        self.FC = []        # fixed charge density vector
        
        # semiconductor continuity equation (1950) solution (for SOLVER class)
        self.n1 = []        # electron density 1D (sparse matrix solution)
        self.p1 = []        # hole density 1D (sparse matrix solution)
        self.n2 = []        # electron density 2D (visualization)
        self.p2 = []        # hole density 2D (visualization)
        self.Jn = []        # electron current density magnitude 2D (visualization)
        self.Jp = []        # hole current density magnitude 2D (visualization)
        self.Jn_r = []      # electron current density in r direction 2D (visualization)
        self.Jn_z = []      # electron current density in z direction 2D (visualization)
        self.Jp_r = []      # hole current density in r direction 2D (visualization)
        self.Jp_z = []      # hole current density in z direction 2D (visualization)

    # ===== adding material parameters =====
    def add_material_parameters(self, mat_para_dictionary):
        # CPU time
        start = time.time()
        
        # user input
        self.MAT = mat_para_dictionary

        # key: mis, value: mat. no. array
        self.MIS_MAT_no = {}
        for mat_name in self.MAT.keys():
            mat_mis = self.MAT[mat_name]['type']        # string, {'I', 'S', 'M'}
            mat_no  = self.MAT[mat_name]['mat_no']      # integer, unique identifier 
            # check dictionary string keys
            if mat_mis not in self.MIS_MAT_no.keys():
                self.MIS_MAT_no[mat_mis] = []
            # add integer identifier number
            self.MIS_MAT_no[mat_mis].append(mat_no)

        # CPU time
        end = time.time()

        # CPU time
        return end-start
    
    # ===== setting unit cell R direction grid (angstrom) =====
    def set_unit_cell_R_grid(self, inward_thk_dr, outward_thk_dr):
        # CPU time
        start = time.time()
        
        # user input
        self.R_inward  = inward_thk_dr
        self.R_outward = outward_thk_dr
        
        # STEP 0: r coordinate (angstrom)
        self.cd = 0.0
        self.R_IN = []
        # r material name & number
        self.R_IN_MAT_name = []
        self.R_IN_MAT_mis = []
        self.R_IN_MAT_ep = []
        self.R_IN_MAT_no = []
        
        # check inward_thk_dr dictionary
        for each_layer in self.R_inward.keys():
            # CD
            if each_layer == 'CD':
                self.cd = self.R_inward[each_layer]
                self.R_IN.append(self.cd/2.0)
                
            # each layer
            else:
                # each layer information
                mat_no = self.R_inward[each_layer]['mat_no']    # integer
                mis = self.MAT[each_layer]['type']              # string
                ep = self.ep0 * self.MAT[each_layer]['k']       # float
                thk = self.R_inward[each_layer]['thk']          # float
                dr = self.R_inward[each_layer]['dr']            # float
                
                # thickness, float (angstrom)
                if thk == -1:
                    r_array = list(np.arange(0.0, self.R_IN[0], dr))
                    self.R_IN = r_array + self.R_IN                         # forward adding
                else:
                    r_array = list(np.arange(self.R_IN[0]-thk, self.R_IN[0], dr))
                    self.R_IN = r_array + self.R_IN
                    
                # material name, string
                mat_name_array = [each_layer] * len(r_array)
                self.R_IN_MAT_name = mat_name_array + self.R_IN_MAT_name    # forward adding

                # material type, string
                mis_array = [mis] * len(r_array)
                self.R_IN_MAT_mis = mis_array + self.R_IN_MAT_mis           # forward adding
                
                # electric permittivity, float, in SI
                mat_ep_array = [ep] * len(r_array)
                self.R_IN_MAT_ep = mat_ep_array + self.R_IN_MAT_ep          # forward adding
                
                # material number, integer (identifier)
                mat_no_array = [mat_no] * len(r_array)
                self.R_IN_MAT_no = mat_no_array + self.R_IN_MAT_no          # forward adding
        
        # STEP 1: r coordinate (angstrom)
        self.R_OUT = {}
        
        # r material name & number
        self.R_OUT_MAT_name = {}
        self.R_OUT_MAT_mis = {}
        self.R_OUT_MAT_ep = {}
        self.R_OUT_MAT_no = {}

        # check outward_thk_dr dictionary
        for each_region in self.R_outward.keys():
            # each region
            self.R_OUT[each_region] = []
            self.R_OUT_MAT_name[each_region] = []
            self.R_OUT_MAT_mis[each_region] = []
            self.R_OUT_MAT_ep[each_region] = []
            self.R_OUT_MAT_no[each_region] = []
            
            # check layers
            for each_index, each_layer in enumerate(list(self.R_outward[each_region])):
                # each layer information
                mat_no = self.R_outward[each_region][each_layer]['mat_no']      # integer
                mis = self.MAT[each_layer]['type']                              # string
                ep = self.ep0 * self.MAT[each_layer]['k']                       # float
                thk = self.R_outward[each_region][each_layer]['thk']            # float
                dr = self.R_outward[each_region][each_layer]['dr']              # float
                
                # thickness, float (angstrom)
                if each_index == 0:
                    r_array = list(np.arange(self.cd/2.0+dr, self.cd/2.0+dr+thk, dr))
                else:
                    r_array = list(np.arange(self.R_OUT[each_region][-1]+dr, self.R_OUT[each_region][-1]+dr+thk, dr))
                self.R_OUT[each_region] = self.R_OUT[each_region] + r_array                             # backwrad adding
                
                # material name, string
                mat_name_array = [each_layer] * len(r_array)
                self.R_OUT_MAT_name[each_region] = self.R_OUT_MAT_name[each_region] + mat_name_array    # backwrad adding
                
                # material type, string
                mis_array = [mis] * len(r_array)
                self.R_OUT_MAT_mis[each_region] = self.R_OUT_MAT_mis[each_region] + mis_array           # backwrad adding
                
                # electric permittivity, float, in SI
                ep_array = [ep] * len(r_array)
                self.R_OUT_MAT_ep[each_region] = self.R_OUT_MAT_ep[each_region] + ep_array              # backwrad adding
                
                # material number, integer (identifier)
                mat_no_array = [mat_no] * len(r_array)
                self.R_OUT_MAT_no[each_region] = self.R_OUT_MAT_no[each_region] + mat_no_array          # backwrad adding

        # STEP 2: r coordinate (angstrom)
        self.R = {}
        # r material name & number
        self.R_MAT_name = {}
        self.R_MAT_mis = {}
        self.R_MAT_ep = {}
        self.R_MAT_no = {}

        # merge > inward_thk_dr + outward_thk_dr dictionary
        for each_region in self.R_OUT.keys():
            # merge
            self.R[each_region] = self.R_IN + self.R_OUT[each_region]                               # node
            self.R_MAT_name[each_region] = self.R_IN_MAT_name + self.R_OUT_MAT_name[each_region]    # element
            self.R_MAT_mis[each_region] = self.R_IN_MAT_mis + self.R_OUT_MAT_mis[each_region]       # element
            self.R_MAT_ep[each_region] = self.R_IN_MAT_ep + self.R_OUT_MAT_ep[each_region]          # element
            self.R_MAT_no[each_region] = self.R_IN_MAT_no + self.R_OUT_MAT_no[each_region]          # element

        # CPU time
        end = time.time()

        # CPU time
        return end-start

    # ===== setting unit cell Z direction grid (angstrom) =====
    def set_unit_cell_Z_grid(self, z_on_thk_dz, z_offset):
        # CPU time
        start = time.time()
        
        # user input
        self.Z_stack = z_on_thk_dz          # dictionary
        self.Z_start = z_offset             # angstrom
        
        # z coordinate (angstrom)
        self.Z = [self.Z_start]             # node
        # z region name & mis flag
        self.Z_REGION = []                  # element
        self.Z_MAT_mis = []                 # element
        self.Z_MAT_ep = []                  # element
        self.Z_MAT_no = []                  # element

        # check Z direction dictionary keys
        for index, each_region in enumerate(list(self.Z_stack.keys())):
            # each region information
            thk = self.Z_stack[each_region]['thk']      # angstrom
            dz = self.Z_stack[each_region]['dz']        # angstrom
            
            # for nodes
            # Z coordinate (angstrom)
            if index == 0:      # first layer
                z_array = list( np.arange(self.Z[index]+dz, self.Z[index]+dz+thk, dz) )
                self.Z = self.Z + z_array           # backward adding
            else:               # others
                z_array =list( np.arange(self.Z[-1]+dz, self.Z[-1]+dz+thk, dz) )
                self.Z = self.Z + z_array           # backward adding
                
            # for elements
            # material name (string), material type (string), electric permittivity (float), material number (integer)
            for each_index in range(len(z_array)):
                self.Z_REGION.append( each_region )                         # element
                self.Z_MAT_mis.append( self.R_MAT_mis[each_region] )        # element
                self.Z_MAT_ep.append( self.R_MAT_ep[each_region] )          # element
                self.Z_MAT_no.append( self.R_MAT_no[each_region] )          # element

        # CPU time
        end = time.time()

        # CPU time
        return end-start

    # ===== setting unit cell RZ 2D grid (angstrom) =====
    def set_unit_cell_RZ_grid(self):
        # CPU time
        start = time.time()
        
        # 1D array (angstrom), node
        self.R_nodes = copy.copy( self.R[ list(self.R.keys())[0] ] )    # one of region
        self.Z_nodes = copy.copy( self.Z )

        # 1D array length (used in array making)
        self.R_nodes_len = len(self.R_nodes)
        self.Z_nodes_len = len(self.Z_nodes)
        self.R_elmts_len = self.R_nodes_len - 1
        self.Z_elmts_len = self.Z_nodes_len - 1

        # sparse matrix size (used in array making)
        self.RZ_nodes_len = self.R_nodes_len * self.Z_nodes_len
        self.RZ_elmts_len = self.R_elmts_len * self.Z_elmts_len
        
        # 2D array R (angstrom -> m)
        self.RZ_R = np.zeros([self.R_nodes_len, self.Z_nodes_len])      # R nodes, Z nodes
        for each_z_node in range(self.Z_nodes_len):
            self.RZ_R[:,each_z_node] = copy.copy( self.R_nodes )        # stacking in Z direction
        self.RZ_R *= 1e-10                                              # angstrom -> m
        self.RZ_dR = self.RZ_R[1:,:] - self.RZ_R[:-1,:]                 # R elements, Z nodes
        
        # 2D array Z (angstrom -> m)
        self.RZ_Z = np.zeros([self.R_nodes_len, self.Z_nodes_len])      # R nodes, Z nodes
        for each_r_node in range(self.R_nodes_len):
            self.RZ_Z[each_r_node,:] = copy.copy( self.Z_nodes )        # stacking in Z direction
        self.RZ_Z *= 1e-10                                              # angstrom -> m
        self.RZ_dZ = self.RZ_Z[:,1:] - self.RZ_Z[:,:-1]                 # R nodes, Z elements

        # 2D array (electric perimittivity)
        self.RZ_EP = np.zeros([self.R_elmts_len, self.Z_elmts_len])                     # R elements, Z elements
        for each_z_elmt in range(self.Z_elmts_len):
            self.RZ_EP[:,each_z_elmt] = copy.copy( self.Z_MAT_ep[each_z_elmt] )         # stacking in Z direction

        # 2D array (material number)
        self.RZ_MATno = np.zeros([self.R_elmts_len, self.Z_elmts_len])                  # R elements, Z elements
        for each_z_elmt in range(self.Z_elmts_len):
            self.RZ_MATno[:,each_z_elmt] = copy.copy( self.Z_MAT_no[each_z_elmt] )      # stacking in Z direction

        # CPU time
        end = time.time()

        # CPU time
        return end-start

    # ===== setting metal-insulator-semiconductor region =====
    def set_unit_cell_RZ_mis_region(self):
        # CPU time
        start = time.time()
        
        # making dictionary
        self.RZ_MIS = {}

        # making edge points set (RZ bondaries)
        edge_points_set = set()
        for each_r in range(self.R_nodes_len):
            edge_points_set.add( (each_r, 0) )                      # Z =  0 boundary
            edge_points_set.add( (each_r, self.Z_nodes_len-1) )     # Z = -1 boundary
        for each_z in range(self.Z_nodes_len):
            edge_points_set.add( (0, each_z) )                      # R =  0 boundary
            edge_points_set.add( (self.R_nodes_len-1, each_z) )     # R = -1 boundary

        # sweep mat_mis (string) keys (first key)
        for each_mat_mis in self.MIS_MAT_no.keys():
            # check each mat_mis key
            if each_mat_mis not in self.RZ_MIS.keys():
                self.RZ_MIS[each_mat_mis] = {}
                
            # sweep mat_no (integer) list (second key)
            for each_mat_no in self.MIS_MAT_no[each_mat_mis]:
                # check each mat_no array
                if each_mat_no not in self.RZ_MIS[each_mat_mis].keys():
                    self.RZ_MIS[each_mat_mis][each_mat_no] = set()
                    
                # get specific R, Z coordinates array in self.RZ_MATno array (elements) having the same each mat_no
                r_index_array, z_index_array = np.where( self.RZ_MATno == each_mat_no )
                
                # mapping: 1 element -> 4 nodes (2D structure)
                for each_point in range(len(r_index_array)):
                    # nodes
                    self.RZ_MIS[each_mat_mis][each_mat_no].add( ( r_index_array[each_point]+0, z_index_array[each_point]+0 ) )
                    self.RZ_MIS[each_mat_mis][each_mat_no].add( ( r_index_array[each_point]+0, z_index_array[each_point]+1 ) )
                    self.RZ_MIS[each_mat_mis][each_mat_no].add( ( r_index_array[each_point]+1, z_index_array[each_point]+0 ) )
                    self.RZ_MIS[each_mat_mis][each_mat_no].add( ( r_index_array[each_point]+1, z_index_array[each_point]+1 ) )

        # set difference: I - edge points set (excluing RZ boundaries, neumann BC)
        for tg_mat_no in self.MIS_MAT_no['I']:
            self.RZ_MIS['I'][tg_mat_no] = self.RZ_MIS['I'][tg_mat_no].difference(edge_points_set)
        
        # set difference: I - M (electrodes, dirichlet BC)
        for tg_mat_no in self.MIS_MAT_no['I']:
            for diff_mat_no in self.MIS_MAT_no['M']:
                self.RZ_MIS['I'][tg_mat_no] = self.RZ_MIS['I'][tg_mat_no].difference(self.RZ_MIS['M'][diff_mat_no])
                
        # set difference: I - S (semiconductors, Scharfetter-Gummel scheme, continuity equations)
        for tg_mat_no in self.MIS_MAT_no['I']:
            for diff_mat_no in self.MIS_MAT_no['S']:
                self.RZ_MIS['I'][tg_mat_no] = self.RZ_MIS['I'][tg_mat_no].difference(self.RZ_MIS['S'][diff_mat_no])

        # CPU time
        end = time.time()

        # CPU time
        return end-start

    # ===== adding ohmic contact =====
    def add_ohmic_contact(self, si_mat_no, bl_mat_no, sl_mat_no):
        # CPU time
        start = time.time()
        
        # BL contact RZ index set
        bl_contact_set = set()
        bl_contact_z_index = 0                          # Z start index
        for each_r_index in range(self.R_nodes_len):
            check_point = (each_r_index, bl_contact_z_index)
            if check_point in self.RZ_MIS['S'][si_mat_no]:      # check mat no
                bl_contact_set.add( check_point )
                
        # SL contact RZ index set
        sl_contact_set = set()
        sl_contact_z_index = self.Z_nodes_len-1         # Z end index
        for each_r_index in range(self.R_nodes_len):
            check_point = (each_r_index, sl_contact_z_index)
            if check_point in self.RZ_MIS['S'][si_mat_no]:      # check mat no
                sl_contact_set.add( check_point )

        # ohmic contact
        self.RZ_MIS['O'] = {}
        self.RZ_MIS['O'][bl_mat_no] = bl_contact_set
        self.RZ_MIS['O'][sl_mat_no] = sl_contact_set

        # semiconductor - ohmic contact
        self.RZ_MIS['S'][si_mat_no] = self.RZ_MIS['S'][si_mat_no].difference( self.RZ_MIS['O'][bl_mat_no] )
        self.RZ_MIS['S'][si_mat_no] = self.RZ_MIS['S'][si_mat_no].difference( self.RZ_MIS['O'][sl_mat_no] )

        # CPU time
        end = time.time()

        # CPU time
        return end-start

    # ===== setting semiconductor parameters =====
    def set_semiconductor_parameters(self, op_temperature, si_mat_no, bl_mat_no, sl_mat_no, doping):
        # CPU time
        start = time.time()
        
        # operating temperature (initialization)
        self.TEMP = op_temperature + 273.15             # celsius -> kelvin
        self.Vtm = self.kb * self.TEMP / self.q

        # semicondutor region RZ index (initialization)
        tg_points  = list(self.RZ_MIS['S'][si_mat_no])      # semiconductor
        bl_points  = list(self.RZ_MIS['O'][bl_mat_no])      # semiconductor + ohmic contact
        sl_points  = list(self.RZ_MIS['O'][sl_mat_no])      # semiconductor + ohmic contact

        # doping profile (initialization)
        self.DP = np.zeros(self.RZ_nodes_len)                           # 1D array
        dopant_type = doping[0]                                         # 'n' or 'p'
        dopant_density = doping[1]                                      # [m]^-3
        for each_point in (tg_points + bl_points + sl_points):
            r_node, z_node = each_point
            index_r_z = self.R_nodes_len * (z_node+0) + (r_node+0)      # 1D array index
            if dopant_type =='n':
                self.DP[index_r_z] = +dopant_density                    # w/ ionized polarity
            elif dopant_type =='p':
                self.DP[index_r_z] = -dopant_density                    # w/ ionized polarity
            else:
                print('set_semiconductor_parameters() > invalid dopant type')

        # intrinsic carrier density (initialization)
        self.N_INT = np.zeros(self.RZ_nodes_len)                        # 1D array
        for each_point in (tg_points + bl_points + sl_points):
            r_node, z_node = each_point
            index_r_z = self.R_nodes_len * (z_node+0) + (r_node+0)      # 1D index index
            #
            self.N_INT[index_r_z] = self.MAT['SI']['n_int']

        # free carrier density (initialization)
        self.n1 = ( np.sqrt( self.DP**2 + 4.0*self.N_INT**2 ) + self.DP ) / 2       # 1D array
        self.p1 = ( np.sqrt( self.DP**2 + 4.0*self.N_INT**2 ) - self.DP ) / 2       # 1D array

        # built-in potential (initialization)
        self.Vbi = np.where( self.N_INT != 0.0, \
                             self.Vtm * np.log( ( self.DP + np.sqrt( self.DP**2 + 4.0*self.N_INT**2 ) ) / ( 2.0*self.N_INT ) ), \
                             0.0 )      # 1D array

        # coefficient of continuity equation matrix (initialization)
        self.CM = {}            # semiconductor
        self.CM_B = {}          # ohmic contact
        self.CM_S = {}          # ohmic contact
        
        # STEP1: check neighbor points (semiconductor)
        for each_point in tg_points:
            # selected point in semiconductor region
            each_r_node, each_z_node = each_point
            # making key
            if each_point not in self.CM.keys():
                self.CM[each_point] = {}
                
            # check r-1, z (R direction)
            if (each_r_node-1, each_z_node) in tg_points:
                self.CM[each_point]['rm1_z'] = {}
                self.CM[each_point]['rm1_z']['index'] = self.R_nodes_len * (each_z_node+0) + (each_r_node-1)        # sparse matrix index
                self.CM[each_point]['rm1_z']['mu_n'] = self.MAT['SI']['mu_n']
                self.CM[each_point]['rm1_z']['mu_p'] = self.MAT['SI']['mu_p']
                R_r_z    = self.RZ_R[each_r_node+0, each_z_node+0]
                dR_rm1_z = self.RZ_R[each_r_node+0, each_z_node+0] - self.RZ_R[each_r_node-1, each_z_node+0]
                self.CM[each_point]['rm1_z']['geometry'] = ( R_r_z - dR_rm1_z/2.0 ) / R_r_z                         # divergence
                self.CM[each_point]['rm1_z']['dR'] = dR_rm1_z                                                       # electric field
                self.CM[each_point]['rm1_z']['dR2'] = dR_rm1_z/2.0                                                  # divergence
                
            # check r+1, z (R direction)
            if (each_r_node+1, each_z_node) in tg_points:
                self.CM[each_point]['rp1_z'] = {}
                self.CM[each_point]['rp1_z']['index'] = self.R_nodes_len * (each_z_node+0) + (each_r_node+1)        # sparse matrix index
                self.CM[each_point]['rp1_z']['mu_n'] = self.MAT['SI']['mu_n']
                self.CM[each_point]['rp1_z']['mu_p'] = self.MAT['SI']['mu_p']
                R_r_z    = self.RZ_R[each_r_node+0, each_z_node+0]
                dR_rp1_z = self.RZ_R[each_r_node+1, each_z_node+0] - self.RZ_R[each_r_node+0, each_z_node+0]
                self.CM[each_point]['rp1_z']['geometry'] = ( R_r_z + dR_rp1_z/2.0 ) / R_r_z                         # divergence
                self.CM[each_point]['rp1_z']['dR'] = dR_rp1_z                                                       # electric field
                self.CM[each_point]['rp1_z']['dR2'] = dR_rp1_z/2.0                                                  # divergence
                
            # check r, z-1 (Z direction)
            if (each_r_node, each_z_node-1) in (tg_points + bl_points):
                self.CM[each_point]['r_zm1'] = {}
                self.CM[each_point]['r_zm1']['index'] = self.R_nodes_len * (each_z_node-1) + (each_r_node+0)        # sparse matrix index
                self.CM[each_point]['r_zm1']['mu_n'] = self.MAT['SI']['mu_n']
                self.CM[each_point]['r_zm1']['mu_p'] = self.MAT['SI']['mu_p']
                Z_r_z    = self.RZ_Z[each_r_node+0, each_z_node+0]
                dZ_r_zm1 = self.RZ_Z[each_r_node+0, each_z_node+0] - self.RZ_Z[each_r_node+0, each_z_node-1]
                self.CM[each_point]['r_zm1']['geometry'] = 1.0                                                      # divergence
                self.CM[each_point]['r_zm1']['dZ'] = dZ_r_zm1                                                       # electric field
                self.CM[each_point]['r_zm1']['dZ2'] = dZ_r_zm1/2.0                                                  # divergence
                
            # check r, z+1 (Z direction)
            if (each_r_node, each_z_node+1) in (tg_points + sl_points):
                self.CM[each_point]['r_zp1'] = {}
                self.CM[each_point]['r_zp1']['index'] = self.R_nodes_len * (each_z_node+1) + (each_r_node+0)        # sparse matrix index
                self.CM[each_point]['r_zp1']['mu_n'] = self.MAT['SI']['mu_n']
                self.CM[each_point]['r_zp1']['mu_p'] = self.MAT['SI']['mu_p']
                Z_r_z    = self.RZ_Z[each_r_node+0, each_z_node+0]
                dZ_r_zp1 = self.RZ_Z[each_r_node+0, each_z_node+1] - self.RZ_Z[each_r_node+0, each_z_node+0]
                self.CM[each_point]['r_zp1']['geometry'] = 1.0                                                      # divergence
                self.CM[each_point]['r_zp1']['dZ'] = dZ_r_zp1                                                       # electric field
                self.CM[each_point]['r_zp1']['dZ2'] = dZ_r_zp1/2.0                                                  # divergence
  
        # STEP2: updating neighbor points (semiconductor)
        for each_point in tg_points:
            # selected point in semiconductor region
            each_r_node, each_z_node = each_point
            # read neighbor info
            each_point_neighbor = list(self.CM[each_point].keys())
            
            # updating dR2 (R direction)
            if ('rm1_z' in each_point_neighbor) and ('rp1_z' in each_point_neighbor):
                new_dR2 = self.CM[each_point]['rm1_z']['dR2'] + self.CM[each_point]['rp1_z']['dR2']     # calculate new value
                self.CM[each_point]['rm1_z']['dR2'] = new_dR2                                           # divergence
                self.CM[each_point]['rp1_z']['dR2'] = new_dR2                                           # divergence
                
            # updating dZ2 (Z direction)
            if ('r_zm1' in each_point_neighbor) and ('r_zp1' in each_point_neighbor):
                new_dZ2 = self.CM[each_point]['r_zm1']['dZ2'] + self.CM[each_point]['r_zp1']['dZ2']     # calculate new value
                self.CM[each_point]['r_zm1']['dZ2'] = new_dZ2                                           # divergence
                self.CM[each_point]['r_zp1']['dZ2'] = new_dZ2                                           # divergence
                
            # check r-1, z (R direction)
            if (each_r_node-1, each_z_node) in tg_points:
                self.CM[each_point]['rm1_z']['n_CM_coeff']  = self.Vtm * self.CM[each_point]['rm1_z']['mu_n'] * \
                                                              self.CM[each_point]['rm1_z']['geometry']
                self.CM[each_point]['rm1_z']['n_CM_coeff'] /= (self.CM[each_point]['rm1_z']['dR'] * self.CM[each_point]['rm1_z']['dR2'])
                self.CM[each_point]['rm1_z']['p_CM_coeff']  = self.Vtm * self.CM[each_point]['rm1_z']['mu_p'] * \
                                                              self.CM[each_point]['rm1_z']['geometry']
                self.CM[each_point]['rm1_z']['p_CM_coeff'] /= (self.CM[each_point]['rm1_z']['dR'] * self.CM[each_point]['rm1_z']['dR2'])

            # check r+1, z (R direction)
            if (each_r_node+1, each_z_node) in tg_points:
                self.CM[each_point]['rp1_z']['n_CM_coeff']  = self.Vtm * self.CM[each_point]['rp1_z']['mu_n'] * \
                                                              self.CM[each_point]['rp1_z']['geometry']
                self.CM[each_point]['rp1_z']['n_CM_coeff'] /= (self.CM[each_point]['rp1_z']['dR'] * self.CM[each_point]['rp1_z']['dR2'])
                self.CM[each_point]['rp1_z']['p_CM_coeff']  = self.Vtm * self.CM[each_point]['rp1_z']['mu_p'] * \
                                                              self.CM[each_point]['rp1_z']['geometry']
                self.CM[each_point]['rp1_z']['p_CM_coeff'] /= (self.CM[each_point]['rp1_z']['dR'] * self.CM[each_point]['rp1_z']['dR2'])

            # check r, z-1 (Z direction)
            if (each_r_node, each_z_node-1) in (tg_points + bl_points):
                self.CM[each_point]['r_zm1']['n_CM_coeff']  = self.Vtm * self.CM[each_point]['r_zm1']['mu_n'] * \
                                                              self.CM[each_point]['r_zm1']['geometry']
                self.CM[each_point]['r_zm1']['n_CM_coeff'] /= (self.CM[each_point]['r_zm1']['dZ'] * self.CM[each_point]['r_zm1']['dZ2'])
                self.CM[each_point]['r_zm1']['p_CM_coeff']  = self.Vtm * self.CM[each_point]['r_zm1']['mu_p'] * \
                                                              self.CM[each_point]['r_zm1']['geometry']
                self.CM[each_point]['r_zm1']['p_CM_coeff'] /= (self.CM[each_point]['r_zm1']['dZ'] * self.CM[each_point]['r_zm1']['dZ2'])

            # check r, z+1 (Z direction)
            if (each_r_node, each_z_node+1) in (tg_points + sl_points):
                self.CM[each_point]['r_zp1']['n_CM_coeff']  = self.Vtm * self.CM[each_point]['r_zp1']['mu_n'] * \
                                                              self.CM[each_point]['r_zp1']['geometry']
                self.CM[each_point]['r_zp1']['n_CM_coeff'] /= (self.CM[each_point]['r_zp1']['dZ'] * self.CM[each_point]['r_zp1']['dZ2'])
                self.CM[each_point]['r_zp1']['p_CM_coeff']  = self.Vtm * self.CM[each_point]['r_zp1']['mu_p'] * \
                                                              self.CM[each_point]['r_zp1']['geometry']
                self.CM[each_point]['r_zp1']['p_CM_coeff'] /= (self.CM[each_point]['r_zp1']['dZ'] * self.CM[each_point]['r_zp1']['dZ2'])

        # STEP3: check neighbor points (semiconductor + bl ohmic contact)
        for each_point in bl_points:
            # selected point in semiconductor region
            each_r_node, each_z_node = each_point
            # making key
            if each_point not in self.CM_B.keys():
                self.CM_B[each_point] = {}
                
            # check r-1, z (R direction)
            if (each_r_node-1, each_z_node) in bl_points:
                self.CM_B[each_point]['rm1_z'] = {}
                self.CM_B[each_point]['rm1_z']['index'] = self.R_nodes_len * (each_z_node+0) + (each_r_node-1)      # sparse matrix index
                self.CM_B[each_point]['rm1_z']['mu_n'] = self.MAT['SI']['mu_n']
                self.CM_B[each_point]['rm1_z']['mu_p'] = self.MAT['SI']['mu_p']
                R_r_z    = self.RZ_R[each_r_node+0, each_z_node+0]
                dR_rm1_z = self.RZ_R[each_r_node+0, each_z_node+0] - self.RZ_R[each_r_node-1, each_z_node+0]
                self.CM_B[each_point]['rm1_z']['geometry'] = ( R_r_z - dR_rm1_z/2.0 ) / R_r_z                       # divergence
                self.CM_B[each_point]['rm1_z']['dR'] = dR_rm1_z                                                     # electric field
                self.CM_B[each_point]['rm1_z']['dR2'] = dR_rm1_z/2.0                                                # divergence
                
            # check r+1, z (R direction)
            if (each_r_node+1, each_z_node) in bl_points:
                self.CM_B[each_point]['rp1_z'] = {}
                self.CM_B[each_point]['rp1_z']['index'] = self.R_nodes_len * (each_z_node+0) + (each_r_node+1)      # sparse matrix index
                self.CM_B[each_point]['rp1_z']['mu_n'] = self.MAT['SI']['mu_n']
                self.CM_B[each_point]['rp1_z']['mu_p'] = self.MAT['SI']['mu_p']
                R_r_z    = self.RZ_R[each_r_node+0, each_z_node+0]
                dR_rp1_z = self.RZ_R[each_r_node+1, each_z_node+0] - self.RZ_R[each_r_node+0, each_z_node+0]
                self.CM_B[each_point]['rp1_z']['geometry'] = ( R_r_z + dR_rp1_z/2.0 ) / R_r_z                       # divergence
                self.CM_B[each_point]['rp1_z']['dR'] = dR_rp1_z                                                     # electric field
                self.CM_B[each_point]['rp1_z']['dR2'] = dR_rp1_z/2.0                                                # divergence
                
            # check r, z+1 (Z direction)
            if (each_r_node, each_z_node+1) in tg_points:
                self.CM_B[each_point]['r_zp1'] = {}
                self.CM_B[each_point]['r_zp1']['index'] = self.R_nodes_len * (each_z_node+1) + (each_r_node+0)      # sparse matrix index
                self.CM_B[each_point]['r_zp1']['mu_n'] = self.MAT['SI']['mu_n']
                self.CM_B[each_point]['r_zp1']['mu_p'] = self.MAT['SI']['mu_p']
                Z_r_z    = self.RZ_Z[each_r_node+0, each_z_node+0]
                dZ_r_zp1 = self.RZ_Z[each_r_node+0, each_z_node+1] - self.RZ_Z[each_r_node+0, each_z_node+0]
                self.CM_B[each_point]['r_zp1']['geometry'] = 1.0                                                    # divergence
                self.CM_B[each_point]['r_zp1']['dZ'] = dZ_r_zp1                                                     # electric field
                self.CM_B[each_point]['r_zp1']['dZ2'] = dZ_r_zp1                                                    # divergence

        # STEP4: updating neighbor points (semiconductor + bl ohmic contact)
        for each_point in bl_points:
            # selected point in semiconductor region
            each_r_node, each_z_node = each_point
            # read neighbor info
            each_point_neighbor = list(self.CM_B[each_point].keys())
            
            # updating dR2 (R direction)
            if ('rm1_z' in each_point_neighbor) and ('rp1_z' in each_point_neighbor):
                new_dR2 = self.CM_B[each_point]['rm1_z']['dR2'] + self.CM_B[each_point]['rp1_z']['dR2']     # calculate new value
                self.CM_B[each_point]['rm1_z']['dR2'] = new_dR2                                             # divergence
                self.CM_B[each_point]['rp1_z']['dR2'] = new_dR2                                             # divergence
                
            # check r-1, z (R direction)
            if (each_r_node-1, each_z_node) in bl_points:
                self.CM_B[each_point]['rm1_z']['n_CM_coeff']  = self.Vtm * self.CM_B[each_point]['rm1_z']['mu_n'] * \
                                                                self.CM_B[each_point]['rm1_z']['geometry']
                self.CM_B[each_point]['rm1_z']['n_CM_coeff'] /= (self.CM_B[each_point]['rm1_z']['dR'] * self.CM_B[each_point]['rm1_z']['dR2'])
                self.CM_B[each_point]['rm1_z']['p_CM_coeff']  = self.Vtm * self.CM_B[each_point]['rm1_z']['mu_p'] * \
                                                                self.CM_B[each_point]['rm1_z']['geometry']
                self.CM_B[each_point]['rm1_z']['p_CM_coeff'] /= (self.CM_B[each_point]['rm1_z']['dR'] * self.CM_B[each_point]['rm1_z']['dR2'])

            # check r+1, z (R direction)
            if (each_r_node+1, each_z_node) in bl_points:
                self.CM_B[each_point]['rp1_z']['n_CM_coeff']  = self.Vtm * self.CM_B[each_point]['rp1_z']['mu_n'] * \
                                                                self.CM_B[each_point]['rp1_z']['geometry']
                self.CM_B[each_point]['rp1_z']['n_CM_coeff'] /= (self.CM_B[each_point]['rp1_z']['dR'] * self.CM_B[each_point]['rp1_z']['dR2'])
                self.CM_B[each_point]['rp1_z']['p_CM_coeff']  = self.Vtm * self.CM_B[each_point]['rp1_z']['mu_p'] * \
                                                                self.CM_B[each_point]['rp1_z']['geometry']
                self.CM_B[each_point]['rp1_z']['p_CM_coeff'] /= (self.CM_B[each_point]['rp1_z']['dR'] * self.CM_B[each_point]['rp1_z']['dR2'])

            # check r, z+1 (Z direction)
            if (each_r_node, each_z_node+1) in tg_points:
                self.CM_B[each_point]['r_zp1']['n_CM_coeff']  = self.Vtm * self.CM_B[each_point]['r_zp1']['mu_n'] * \
                                                                self.CM_B[each_point]['r_zp1']['geometry']
                self.CM_B[each_point]['r_zp1']['n_CM_coeff'] /= (self.CM_B[each_point]['r_zp1']['dZ'] * self.CM_B[each_point]['r_zp1']['dZ2'])
                self.CM_B[each_point]['r_zp1']['p_CM_coeff']  = self.Vtm * self.CM_B[each_point]['r_zp1']['mu_p'] * \
                                                                self.CM_B[each_point]['r_zp1']['geometry']
                self.CM_B[each_point]['r_zp1']['p_CM_coeff'] /= (self.CM_B[each_point]['r_zp1']['dZ'] * self.CM_B[each_point]['r_zp1']['dZ2'])

        # STEP5: check neighbor points (semiconductor + sl ohmic contact)
        for each_point in sl_points:
            # selected point in semiconductor region
            each_r_node, each_z_node = each_point
            # making key
            if each_point not in self.CM_S.keys():
                self.CM_S[each_point] = {}
                
            # check r-1, z (R direction)
            if (each_r_node-1, each_z_node) in sl_points:
                self.CM_S[each_point]['rm1_z'] = {}
                self.CM_S[each_point]['rm1_z']['index'] = self.R_nodes_len * (each_z_node+0) + (each_r_node-1)      # sparse matrix index
                self.CM_S[each_point]['rm1_z']['mu_n'] = self.MAT['SI']['mu_n']
                self.CM_S[each_point]['rm1_z']['mu_p'] = self.MAT['SI']['mu_p']
                R_r_z    = self.RZ_R[each_r_node+0, each_z_node+0]
                dR_rm1_z = self.RZ_R[each_r_node+0, each_z_node+0] - self.RZ_R[each_r_node-1, each_z_node+0]
                self.CM_S[each_point]['rm1_z']['geometry'] = ( R_r_z - dR_rm1_z/2.0 ) / R_r_z                       # divergence
                self.CM_S[each_point]['rm1_z']['dR'] = dR_rm1_z                                                     # electric field
                self.CM_S[each_point]['rm1_z']['dR2'] = dR_rm1_z/2.0                                                # divergence
                
            # check r+1, z (R direction)
            if (each_r_node+1, each_z_node) in sl_points:
                self.CM_S[each_point]['rp1_z'] = {}
                self.CM_S[each_point]['rp1_z']['index'] = self.R_nodes_len * (each_z_node+0) + (each_r_node+1)      # sparse matrix index
                self.CM_S[each_point]['rp1_z']['mu_n'] = self.MAT['SI']['mu_n']
                self.CM_S[each_point]['rp1_z']['mu_p'] = self.MAT['SI']['mu_p']
                R_r_z    = self.RZ_R[each_r_node+0, each_z_node+0]
                dR_rp1_z = self.RZ_R[each_r_node+1, each_z_node+0] - self.RZ_R[each_r_node+0, each_z_node+0]
                self.CM_S[each_point]['rp1_z']['geometry'] = ( R_r_z + dR_rp1_z/2.0 ) / R_r_z                       # divergence
                self.CM_S[each_point]['rp1_z']['dR'] = dR_rp1_z                                                     # electric field
                self.CM_S[each_point]['rp1_z']['dR2'] = dR_rp1_z/2.0                                                # divergence
                
            # check r, z-1 (Z direction)
            if (each_r_node, each_z_node-1) in tg_points:
                self.CM_S[each_point]['r_zm1'] = {}
                self.CM_S[each_point]['r_zm1']['index'] = self.R_nodes_len * (each_z_node-1) + (each_r_node+0)      # sparse matrix index
                self.CM_S[each_point]['r_zm1']['mu_n'] = self.MAT['SI']['mu_n']
                self.CM_S[each_point]['r_zm1']['mu_p'] = self.MAT['SI']['mu_p']
                Z_r_z    = self.RZ_Z[each_r_node+0, each_z_node+0]
                dZ_r_zm1 = self.RZ_Z[each_r_node+0, each_z_node+0] - self.RZ_Z[each_r_node+0, each_z_node-1]
                self.CM_S[each_point]['r_zm1']['geometry'] = 1.0                                                    # divergence
                self.CM_S[each_point]['r_zm1']['dZ'] = dZ_r_zm1                                                     # electric field
                self.CM_S[each_point]['r_zm1']['dZ2'] = dZ_r_zm1                                                    # divergence

        # STEP6: updating neighbor points (semiconductor + sl ohmic contact)
        for each_point in sl_points:
            # selected point in semiconductor region
            each_r_node, each_z_node = each_point
            # read neighbor info
            each_point_neighbor = list(self.CM_S[each_point].keys())
            
            # updating dR2 (R direction)
            if ('rm1_z' in each_point_neighbor) and ('rp1_z' in each_point_neighbor):
                new_dR2 = self.CM_S[each_point]['rm1_z']['dR2'] + self.CM_S[each_point]['rp1_z']['dR2']     # calculate new value
                self.CM_S[each_point]['rm1_z']['dR2'] = new_dR2                                             # divergence
                self.CM_S[each_point]['rp1_z']['dR2'] = new_dR2                                             # divergence
                
            # check r-1, z (R direction)
            if (each_r_node-1, each_z_node) in sl_points:
                self.CM_S[each_point]['rm1_z']['n_CM_coeff']  = self.Vtm * self.CM_S[each_point]['rm1_z']['mu_n'] * \
                                                                self.CM_S[each_point]['rm1_z']['geometry']
                self.CM_S[each_point]['rm1_z']['n_CM_coeff'] /= (self.CM_S[each_point]['rm1_z']['dR'] * self.CM_S[each_point]['rm1_z']['dR2'])
                self.CM_S[each_point]['rm1_z']['p_CM_coeff']  = self.Vtm * self.CM_S[each_point]['rm1_z']['mu_p'] * \
                                                                self.CM_S[each_point]['rm1_z']['geometry']
                self.CM_S[each_point]['rm1_z']['p_CM_coeff'] /= (self.CM_S[each_point]['rm1_z']['dR'] * self.CM_S[each_point]['rm1_z']['dR2'])

            # check r+1, z (R direction)
            if (each_r_node+1, each_z_node) in sl_points:
                self.CM_S[each_point]['rp1_z']['n_CM_coeff']  = self.Vtm * self.CM_S[each_point]['rp1_z']['mu_n'] * \
                                                                self.CM_S[each_point]['rp1_z']['geometry']
                self.CM_S[each_point]['rp1_z']['n_CM_coeff'] /= (self.CM_S[each_point]['rp1_z']['dR'] * self.CM_S[each_point]['rp1_z']['dR2'])
                self.CM_S[each_point]['rp1_z']['p_CM_coeff']  = self.Vtm * self.CM_S[each_point]['rp1_z']['mu_p'] * \
                                                                self.CM_S[each_point]['rp1_z']['geometry']
                self.CM_S[each_point]['rp1_z']['p_CM_coeff'] /= (self.CM_S[each_point]['rp1_z']['dR'] * self.CM_S[each_point]['rp1_z']['dR2'])

            # check r, z-1 (Z direction)
            if (each_r_node, each_z_node-1) in tg_points:
                self.CM_S[each_point]['r_zm1']['n_CM_coeff']  = self.Vtm * self.CM_S[each_point]['r_zm1']['mu_n'] * \
                                                                self.CM_S[each_point]['r_zm1']['geometry']
                self.CM_S[each_point]['r_zm1']['n_CM_coeff'] /= (self.CM_S[each_point]['r_zm1']['dZ'] * self.CM_S[each_point]['r_zm1']['dZ2'])
                self.CM_S[each_point]['r_zm1']['p_CM_coeff']  = self.Vtm * self.CM_S[each_point]['r_zm1']['mu_p'] * \
                                                                self.CM_S[each_point]['r_zm1']['geometry']
                self.CM_S[each_point]['r_zm1']['p_CM_coeff'] /= (self.CM_S[each_point]['r_zm1']['dZ'] * self.CM_S[each_point]['r_zm1']['dZ2'])

        # CPU time
        end = time.time()

        # CPU time
        return end-start
        
    # ===== making poisson matrix  =====
    def make_poisson_matrix(self, bl_mat_no, sl_mat_no):
        # CPU time
        start = time.time()
        
        # STEP0: making sparse matrix
        self.PM   = sc.sparse.dok_matrix((self.RZ_nodes_len, self.RZ_nodes_len))        # sparse matrix
        self.PM_B = np.zeros(self.RZ_nodes_len)                                         # BL ohmic contact vector
        self.PM_S = np.zeros(self.RZ_nodes_len)                                         # SL ohmic contact vector

        # STEP1: dirichlet boundary conditions (electrodes)
        for each_mat_no in self.RZ_MIS['M'].keys():
            for each_r, each_z in self.RZ_MIS['M'][each_mat_no]:
                # 1D serialization index
                index_r_z = self.R_nodes_len * (each_z+0) + (each_r+0)
                # sparse matrix (directly related to electric potential)
                self.PM[index_r_z, index_r_z] += 1.0

        # STEP2-1: BL ohmic contact @Z = 0
        for each_r, each_z in self.RZ_MIS['O'][bl_mat_no]:
            # 1D serialization index
            index_r_z   = self.R_nodes_len * (each_z+0) + (each_r+0)
            index_rm1_z = self.R_nodes_len * (each_z+0) + (each_r-1)
            index_rp1_z = self.R_nodes_len * (each_z+0) + (each_r+1)
            index_r_zm1 = self.R_nodes_len * (each_z-1) + (each_r+0)        # BL ohmic contact (invalid index)
            index_r_zp1 = self.R_nodes_len * (each_z+1) + (each_r+0)
            # if not defined
            if self.PM[index_r_z, index_r_z] == 0.0:
                # geometry factors in r direction
                geometry_effect_rm1_z  = (self.RZ_R[each_r+0,each_z+0]-(self.RZ_R[each_r+0,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0)
                geometry_effect_rm1_z /=  self.RZ_R[each_r+0,each_z+0]
                geometry_effect_rp1_z  = (self.RZ_R[each_r+0,each_z+0]+(self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r+0,each_z+0])/2.0)
                geometry_effect_rp1_z /=  self.RZ_R[each_r+0,each_z+0]
                geometry_effect_r_zm1  = 1.0                                # BL ohmic contact
                geometry_effect_r_zp1  = 1.0
                # 2nd derivatives
                geometry_effect_rm1_z /= (self.RZ_R[each_r+0,each_z+0]-self.RZ_R[each_r-1,each_z+0])
                geometry_effect_rm1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0
                geometry_effect_rp1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r+0,each_z+0])
                geometry_effect_rp1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0
                geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z+0])
                geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z+0])        # BL ohmic contact
                geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z+0])
                geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z+0])
                # electric permittivity: z-1 (invalid) -> z+0
                ep_z_avg_rm1 = (self.RZ_EP[each_r-1,each_z+0]+self.RZ_EP[each_r-1,each_z+0])/2.0
                ep_z_avg_rp1 = (self.RZ_EP[each_r+0,each_z+0]+self.RZ_EP[each_r+0,each_z+0])/2.0
                ep_r_avg_zm1 = (self.RZ_EP[each_r-1,each_z+0]+self.RZ_EP[each_r+0,each_z+0])/2.0
                ep_r_avg_zp1 = (self.RZ_EP[each_r-1,each_z+0]+self.RZ_EP[each_r+0,each_z+0])/2.0
                # elements
                pm_rm1_z = geometry_effect_rm1_z * ep_z_avg_rm1
                pm_rp1_z = geometry_effect_rp1_z * ep_z_avg_rp1
                pm_r_zm1 = geometry_effect_r_zm1 * ep_r_avg_zm1             # BL ohmic contact
                pm_r_zp1 = geometry_effect_r_zp1 * ep_r_avg_zp1
                # sparse matrix
                self.PM[index_r_z, index_r_z  ] += +pm_rm1_z + pm_rp1_z + pm_r_zp1
                self.PM[index_r_z, index_rm1_z] += -pm_rm1_z
                self.PM[index_r_z, index_rp1_z] += -pm_rp1_z
                self.PM[index_r_z, index_r_zp1] += -pm_r_zp1
                # BL ohmic contact vector
                self.PM[index_r_z, index_r_z  ] += +pm_r_zm1
                self.PM_B[index_r_z           ] += +pm_r_zm1        

        # STEP2-2: SL ohmic contact @Z = -1
        for each_r, each_z in self.RZ_MIS['O'][sl_mat_no]:
            # 1D serialization index
            index_r_z   = self.R_nodes_len * (each_z+0) + (each_r+0)
            index_rm1_z = self.R_nodes_len * (each_z+0) + (each_r-1)
            index_rp1_z = self.R_nodes_len * (each_z+0) + (each_r+1)
            index_r_zm1 = self.R_nodes_len * (each_z-1) + (each_r+0)
            index_r_zp1 = self.R_nodes_len * (each_z+1) + (each_r+0)        # SL ohmic contact
            # if not defined
            if self.PM[index_r_z, index_r_z] == 0.0:
                # geometry factors in r direction
                geometry_effect_rm1_z  = (self.RZ_R[each_r+0,each_z+0]-(self.RZ_R[each_r+0,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0)
                geometry_effect_rm1_z /=  self.RZ_R[each_r+0,each_z+0]
                geometry_effect_rp1_z  = (self.RZ_R[each_r+0,each_z+0]+(self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r+0,each_z+0])/2.0)
                geometry_effect_rp1_z /=  self.RZ_R[each_r+0,each_z+0]
                geometry_effect_r_zm1  = 1.0
                geometry_effect_r_zp1  = 1.0                                # SL ohmic contact
                # 2nd derivatives
                geometry_effect_rm1_z /= (self.RZ_R[each_r+0,each_z+0]-self.RZ_R[each_r-1,each_z+0])
                geometry_effect_rm1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0
                geometry_effect_rp1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r+0,each_z+0])
                geometry_effect_rp1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0
                geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+0]-self.RZ_Z[each_r+0,each_z-1])
                geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+0]-self.RZ_Z[each_r+0,each_z-1])
                geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+0]-self.RZ_Z[each_r+0,each_z-1])
                geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+0]-self.RZ_Z[each_r+0,each_z-1])        # SL ohmic contact
                # electric permittivity: z+0 (invalud) -> z-1
                ep_z_avg_rm1 = (self.RZ_EP[each_r-1,each_z-1]+self.RZ_EP[each_r-1,each_z-1])/2.0
                ep_z_avg_rp1 = (self.RZ_EP[each_r+0,each_z-1]+self.RZ_EP[each_r+0,each_z-1])/2.0
                ep_r_avg_zm1 = (self.RZ_EP[each_r-1,each_z-1]+self.RZ_EP[each_r+0,each_z-1])/2.0
                ep_r_avg_zp1 = (self.RZ_EP[each_r-1,each_z-1]+self.RZ_EP[each_r+0,each_z-1])/2.0
                # elements
                pm_rm1_z = geometry_effect_rm1_z * ep_z_avg_rm1
                pm_rp1_z = geometry_effect_rp1_z * ep_z_avg_rp1
                pm_r_zm1 = geometry_effect_r_zm1 * ep_r_avg_zm1
                pm_r_zp1 = geometry_effect_r_zp1 * ep_r_avg_zp1             # SL ohmic contact
                # sparse matrix
                self.PM[index_r_z, index_r_z  ] += +pm_rm1_z + pm_rp1_z + pm_r_zm1
                self.PM[index_r_z, index_rm1_z] += -pm_rm1_z
                self.PM[index_r_z, index_rp1_z] += -pm_rp1_z
                self.PM[index_r_z, index_r_zm1] += -pm_r_zm1
                # SL ohmic contact vector
                self.PM[index_r_z, index_r_z  ] += +pm_r_zp1
                self.PM_S[index_r_z           ] += +pm_r_zp1

        # STEP3-1: neumann boundary conditions (Z boundaries)
        for each_z in [0, self.Z_nodes_len-1]:
            for each_r in range(self.R_nodes_len):
                # 1D serialization index
                index_r_z   = self.R_nodes_len * (each_z+0) + (each_r+0)
                index_rm1_z = self.R_nodes_len * (each_z+0) + (each_r-1)
                index_rp1_z = self.R_nodes_len * (each_z+0) + (each_r+1)
                index_r_zm1 = self.R_nodes_len * (each_z-1) + (each_r+0)
                index_r_zp1 = self.R_nodes_len * (each_z+1) + (each_r+0)
                # if not defined
                if self.PM[index_r_z, index_r_z] == 0.0:
                    # boundary
                    if (each_z == 0) and (each_r == 0):
                        # neumann boundary conditions
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rp1_z] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zp1] += -1.0
                    elif (each_z == 0) and (each_r == (self.R_nodes_len-1)):
                        # neumann boundary conditions
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rm1_z] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zp1] += -1.0
                    elif (each_z == 0):
                        # geometry factors in r direction
                        geometry_effect_rm1_z  = (self.RZ_R[each_r+0,each_z+0]-(self.RZ_R[each_r+0,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0)
                        geometry_effect_rm1_z /=  self.RZ_R[each_r+0,each_z+0]
                        geometry_effect_rp1_z  = (self.RZ_R[each_r+0,each_z+0]+(self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r+0,each_z+0])/2.0)
                        geometry_effect_rp1_z /=  self.RZ_R[each_r+0,each_z+0]
                        # 2nd derivatives
                        geometry_effect_rm1_z /= (self.RZ_R[each_r+0,each_z+0]-self.RZ_R[each_r-1,each_z+0])
                        geometry_effect_rm1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0
                        geometry_effect_rp1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r+0,each_z+0])
                        geometry_effect_rp1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0
                        # electric permittivity
                        ep_z_avg_rm1 = (self.RZ_EP[each_r-1,each_z+0]+self.RZ_EP[each_r-1,each_z+0])/2.0
                        ep_z_avg_rp1 = (self.RZ_EP[each_r+0,each_z+0]+self.RZ_EP[each_r+0,each_z+0])/2.0
                        # elements
                        pm_rm1_z = geometry_effect_rm1_z * ep_z_avg_rm1
                        pm_rp1_z = geometry_effect_rp1_z * ep_z_avg_rp1
                        # poisson matrix
                        self.PM[index_r_z, index_r_z  ] += +pm_rm1_z + pm_rp1_z
                        self.PM[index_r_z, index_rm1_z] += -pm_rm1_z
                        self.PM[index_r_z, index_rp1_z] += -pm_rp1_z
                        # neumann boundary conditions
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zp1] += -1.0
                    # boundary
                    if each_z == (self.Z_nodes_len-1) and (each_r == 0):
                        # neumann boundary conditions
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rp1_z] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zm1] += -1.0
                    elif each_z == (self.Z_nodes_len-1) and (each_r == (self.R_nodes_len-1)):
                        # neumann boundary conditions
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rm1_z] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zm1] += -1.0
                    elif each_z == (self.Z_nodes_len-1):
                        # geometry factors in r direction
                        geometry_effect_rm1_z  = (self.RZ_R[each_r+0,each_z+0]-(self.RZ_R[each_r+0,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0)
                        geometry_effect_rm1_z /=  self.RZ_R[each_r+0,each_z+0]
                        geometry_effect_rp1_z  = (self.RZ_R[each_r+0,each_z+0]+(self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r+0,each_z+0])/2.0)
                        geometry_effect_rp1_z /=  self.RZ_R[each_r+0,each_z+0]
                        # 2nd derivatives
                        geometry_effect_rm1_z /= (self.RZ_R[each_r+0,each_z+0]-self.RZ_R[each_r-1,each_z+0])
                        geometry_effect_rm1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0
                        geometry_effect_rp1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r+0,each_z+0])
                        geometry_effect_rp1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0
                        # electric permittivity
                        ep_z_avg_rm1 = (self.RZ_EP[each_r-1,each_z-1]+self.RZ_EP[each_r-1,each_z-1])/2.0
                        ep_z_avg_rp1 = (self.RZ_EP[each_r+0,each_z-1]+self.RZ_EP[each_r+0,each_z-1])/2.0
                        # elements
                        pm_rm1_z = geometry_effect_rm1_z * ep_z_avg_rm1
                        pm_rp1_z = geometry_effect_rp1_z * ep_z_avg_rp1
                        # poisson matrix
                        self.PM[index_r_z, index_r_z  ] += +pm_rm1_z + pm_rp1_z
                        self.PM[index_r_z, index_rm1_z] += -pm_rm1_z
                        self.PM[index_r_z, index_rp1_z] += -pm_rp1_z
                        # neumann boundary conditions
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zm1] += -1.0

        # STEP3-2: neumann boundary conditions (R boundaries)
        for each_r in [0, self.R_nodes_len-1]:
            for each_z in range(self.Z_nodes_len):
                # 1D serialization index
                index_r_z   = self.R_nodes_len * (each_z+0) + (each_r+0)
                index_rm1_z = self.R_nodes_len * (each_z+0) + (each_r-1)
                index_rp1_z = self.R_nodes_len * (each_z+0) + (each_r+1)
                index_r_zm1 = self.R_nodes_len * (each_z-1) + (each_r+0)
                index_r_zp1 = self.R_nodes_len * (each_z+1) + (each_r+0)
                # if not defined
                if self.PM[index_r_z, index_r_z] == 0.0:
                    # boundary
                    if (each_r == 0) and (each_z == 0):
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rp1_z] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zp1] += -1.0
                    elif (each_r == 0) and (each_z == (self.Z_nodes_len-1)):
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rp1_z] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zm1] += -1.0
                    elif (each_r == 0):
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rp1_z] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zp1] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zm1] += -1.0
                    # boundary
                    if (each_r == (self.R_nodes_len-1)) and (each_z == 0):
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rm1_z] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zp1] += -1.0
                    elif (each_r == (self.R_nodes_len-1)) and (each_z == (self.Z_nodes_len-1)):
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rm1_z] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zm1] += -1.0
                    elif (each_r == (self.R_nodes_len-1)):
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rm1_z] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zp1] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zm1] += -1.0

        # STEP4: inside boundary conditions
        for each_r in range(1, self.R_nodes_len-1):
            for each_z in range(1, self.Z_nodes_len-1):
                # 1D serialization index
                index_r_z   = self.R_nodes_len * (each_z+0) + (each_r+0)
                index_rm1_z = self.R_nodes_len * (each_z+0) + (each_r-1)
                index_rp1_z = self.R_nodes_len * (each_z+0) + (each_r+1)
                index_r_zm1 = self.R_nodes_len * (each_z-1) + (each_r+0)
                index_r_zp1 = self.R_nodes_len * (each_z+1) + (each_r+0)
                # except electrodes & boundary conditions
                if self.PM[index_r_z, index_r_z] == 0.0:
                    # geometry factors in r direction
                    geometry_effect_rm1_z  = (self.RZ_R[each_r+0,each_z+0]-(self.RZ_R[each_r+0,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0)
                    geometry_effect_rm1_z /=  self.RZ_R[each_r+0,each_z+0]
                    geometry_effect_rp1_z  = (self.RZ_R[each_r+0,each_z+0]+(self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r+0,each_z+0])/2.0)
                    geometry_effect_rp1_z /=  self.RZ_R[each_r+0,each_z+0]
                    geometry_effect_r_zm1  = 1.0
                    geometry_effect_r_zp1  = 1.0
                    # 2nd derivatives
                    geometry_effect_rm1_z /= (self.RZ_R[each_r+0,each_z+0]-self.RZ_R[each_r-1,each_z+0])
                    geometry_effect_rm1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0
                    geometry_effect_rp1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r+0,each_z+0])
                    geometry_effect_rp1_z /= (self.RZ_R[each_r+1,each_z+0]-self.RZ_R[each_r-1,each_z+0])/2.0
                    geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+0]-self.RZ_Z[each_r+0,each_z-1])
                    geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z-1])/2.0
                    geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z+0])
                    geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z-1])/2.0
                    # electric permittivity
                    ep_z_avg_rm1 = (self.RZ_EP[each_r-1,each_z-1]+self.RZ_EP[each_r-1,each_z+0])/2.0
                    ep_z_avg_rp1 = (self.RZ_EP[each_r+0,each_z-1]+self.RZ_EP[each_r+0,each_z+0])/2.0
                    ep_r_avg_zm1 = (self.RZ_EP[each_r-1,each_z-1]+self.RZ_EP[each_r+0,each_z-1])/2.0
                    ep_r_avg_zp1 = (self.RZ_EP[each_r-1,each_z+0]+self.RZ_EP[each_r+0,each_z+0])/2.0
                    # elements
                    pm_rm1_z = geometry_effect_rm1_z * ep_z_avg_rm1
                    pm_rp1_z = geometry_effect_rp1_z * ep_z_avg_rp1
                    pm_r_zm1 = geometry_effect_r_zm1 * ep_r_avg_zm1
                    pm_r_zp1 = geometry_effect_r_zp1 * ep_r_avg_zp1
                    #
                    self.PM[index_r_z, index_r_z  ] = +pm_rm1_z + pm_rp1_z + pm_r_zm1 + pm_r_zp1
                    self.PM[index_r_z, index_rm1_z] = -pm_rm1_z
                    self.PM[index_r_z, index_rp1_z] = -pm_rp1_z
                    self.PM[index_r_z, index_r_zm1] = -pm_r_zm1
                    self.PM[index_r_z, index_r_zp1] = -pm_r_zp1
        
        # sparse matrix
        self.PMcsr = self.PM.tocsr()

        # external bias vector initialization
        self.EB = np.zeros(self.RZ_nodes_len)

        # fixed charge density vector initialization
        self.FC = np.zeros(self.RZ_nodes_len)

        # CPU time
        end = time.time()

        # CPU time
        return end-start

    # ===== making continuity matrix  =====
    def make_continuity_matrix(self):
        # CPU time
        start = time.time()
        
        # making sparse matrix (time evolution)
        self.N = sc.sparse.dok_matrix((self.RZ_nodes_len, self.RZ_nodes_len))
        self.P = sc.sparse.dok_matrix((self.RZ_nodes_len, self.RZ_nodes_len))

        # identity
        for each_z in range(self.Z_nodes_len):
            for each_r in range(self.R_nodes_len):
                # 1D serialization index
                index_r_z = self.R_nodes_len * (each_z+0) + (each_r+0)
                # identity
                self.N[index_r_z, index_r_z] = 1.0
                self.P[index_r_z, index_r_z] = 1.0

        # CSR format
        self.Ncsr = self.N.tocsr()
        self.Pcsr = self.P.tocsr()

        # CPU time
        end = time.time()

        # CPU time
        return end-start


#
# CLASS: SOLVER (sparse matrix solver)
#

class SOLVER(GRID):

    # ===== making external bias vector  =====
    def make_external_bias_vector(self, external_bias_conditions):
        # CPU time
        start = time.time()
        
        # metal electrodes
        for each_mat_no in self.RZ_MIS['M'].keys():
            for each_r, each_z in self.RZ_MIS['M'][each_mat_no]:
                # 1D serialization index
                index_r_z = self.R_nodes_len * (each_z+0) + (each_r+0)
                # external bias @metal electrodes
                self.EB[index_r_z] = external_bias_conditions[each_mat_no]

        # BL, SL ohmic contact
        for each_mat_no in self.RZ_MIS['O'].keys():
            for each_r, each_z in self.RZ_MIS['O'][each_mat_no]:
                # 1D serialization index
                index_r_z = self.R_nodes_len * (each_z+0) + (each_r+0)
                # BL ohmic contact
                if each_mat_no == 10001:
                    self.EB[index_r_z] = self.Vbi[index_r_z] + self.PM_B[index_r_z] * external_bias_conditions[each_mat_no]
                # SL ohmic contact
                if each_mat_no == 10002:
                    self.EB[index_r_z] = self.Vbi[index_r_z] + self.PM_S[index_r_z] * external_bias_conditions[each_mat_no]

        # CPU time
        end = time.time()

        # CPU time
        return end-start

    # ===== making fixed charge vector  =====
    def make_fixed_charge_vector(self, fixed_charge_density):
        # CPU time
        start = time.time()
        
        # sweep material type
        for each_mat_mis in ['I', 'S', 'O']:
            # sweep material no.
            for each_mat_no in self.RZ_MIS[each_mat_mis].keys():
                # check
                if each_mat_no in fixed_charge_density.keys():
                    for each_r, each_z in self.RZ_MIS[each_mat_mis][each_mat_no]:
                        # 1D serialization index
                        index_r_z = self.R_nodes_len * (each_z+0) + (each_r+0)
                        # fixed charge density 
                        self.FC[index_r_z] = fixed_charge_density[each_mat_no]

        # CPU time
        end = time.time()

        # CPU time
        return end-start

    # ===== solving poisson equation  =====
    def solve_poisson_equation(self):
        # CPU time
        start = time.time()
        
        # sparse matrix solver for poisson equation
        self.V1 = sc.sparse.linalg.spsolve(self.PMcsr, self.EB + self.q*(self.FC + self.p1 - self.n1 + self.DP) )

        # 2D visualization
        self.V2 = self.V1.reshape(self.Z_nodes_len, self.R_nodes_len).T
        self.Er = ( self.V2[1:,:] - self.V2[:-1,:] ) / self.RZ_dR
        self.Ez = ( self.V2[:,1:] - self.V2[:,:-1] ) / self.RZ_dZ
        self.E  = np.sqrt( self.Er[:,:-1]**2 + self.Ez[:-1,:]**2 )
        
        # post processing 1 (for continuity equations)
        self.dVr_f = ( self.V2[1:,:] - self.V2[:-1,:] ) / self.Vtm
        self.dVr_b = ( self.V2[:-1,:] - self.V2[1:,:] ) / self.Vtm
        self.dVz_f = ( self.V2[:,1:] - self.V2[:,:-1] ) / self.Vtm
        self.dVz_b = ( self.V2[:,:-1] - self.V2[:,1:] ) / self.Vtm

        # post processing 2 (for continuity equations)
        B_tol = 1e-10
        self.Br_f = np.where( np.abs(self.dVr_f) > B_tol, self.dVr_f / ( np.exp(self.dVr_f) - 1.0 ), 1.0)
        self.Br_b = np.where( np.abs(self.dVr_b) > B_tol, self.dVr_b / ( np.exp(self.dVr_b) - 1.0 ), 1.0)
        self.Bz_f = np.where( np.abs(self.dVz_f) > B_tol, self.dVz_f / ( np.exp(self.dVz_f) - 1.0 ), 1.0)
        self.Bz_b = np.where( np.abs(self.dVz_b) > B_tol, self.dVz_b / ( np.exp(self.dVz_b) - 1.0 ), 1.0)

        # CPU time
        end = time.time()

        # CPU time
        return end-start

    # ===== making N P matrix  =====
    def make_N_P_matrix(self, dt):
        # CPU time
        start = time.time()

        # making sparse matrix (continuity equation)
        self.dN = sc.sparse.dok_matrix((self.RZ_nodes_len, self.RZ_nodes_len))
        self.dP = sc.sparse.dok_matrix((self.RZ_nodes_len, self.RZ_nodes_len))

        # sweep target points
        for each_point in self.CM.keys():
            # selected target point
            each_r, each_z = each_point
            tg_index = self.R_nodes_len * each_z + each_r
            
            # sweep neighbor points around selected target point
            for neighbor_point in self.CM[each_point].keys():
                # selected neighbor point
                neighbor_index = self.CM[each_point][neighbor_point]['index']
                n_CM_coeff = self.CM[each_point][neighbor_point]['n_CM_coeff']
                p_CM_coeff = self.CM[each_point][neighbor_point]['p_CM_coeff']
                
                # r-1, z
                if neighbor_point == 'rm1_z':   
                    # change in electron density
                    self.dN[tg_index, tg_index      ] += +n_CM_coeff * self.Br_f[ each_r-1, each_z+0 ] * dt
                    self.dN[tg_index, neighbor_index] += -n_CM_coeff * self.Br_b[ each_r-1, each_z+0 ] * dt
                    # change in hole density
                    self.dP[tg_index, tg_index      ] += +p_CM_coeff * self.Br_b[ each_r-1, each_z+0 ] * dt
                    self.dP[tg_index, neighbor_index] += -p_CM_coeff * self.Br_f[ each_r-1, each_z+0 ] * dt
                    
                # r+1, z
                if neighbor_point == 'rp1_z':   
                    # change in electron density
                    self.dN[tg_index, tg_index      ] += +n_CM_coeff * self.Br_b[ each_r+0, each_z+0 ] * dt
                    self.dN[tg_index, neighbor_index] += -n_CM_coeff * self.Br_f[ each_r+0, each_z+0 ] * dt
                    # change in hole density
                    self.dP[tg_index, tg_index      ] += +p_CM_coeff * self.Br_f[ each_r+0, each_z+0 ] * dt
                    self.dP[tg_index, neighbor_index] += -p_CM_coeff * self.Br_b[ each_r+0, each_z+0 ] * dt
                    
                # r, z-1
                if neighbor_point == 'r_zm1':   
                    # change in electron density
                    self.dN[tg_index, tg_index      ] += +n_CM_coeff * self.Bz_f[ each_r+0, each_z-1 ] * dt
                    self.dN[tg_index, neighbor_index] += -n_CM_coeff * self.Bz_b[ each_r+0, each_z-1 ] * dt
                    # change in hole density
                    self.dP[tg_index, tg_index      ] += +p_CM_coeff * self.Bz_b[ each_r+0, each_z-1 ] * dt
                    self.dP[tg_index, neighbor_index] += -p_CM_coeff * self.Bz_f[ each_r+0, each_z-1 ] * dt
                    
                # r, z+1
                if neighbor_point == 'r_zp1':   
                    # change in electron density
                    self.dN[tg_index, tg_index      ] += +n_CM_coeff * self.Bz_b[ each_r+0, each_z+0 ] * dt
                    self.dN[tg_index, neighbor_index] += -n_CM_coeff * self.Bz_f[ each_r+0, each_z+0 ] * dt
                    # change in hole density
                    self.dP[tg_index, tg_index      ] += +p_CM_coeff * self.Bz_f[ each_r+0, each_z+0 ] * dt
                    self.dP[tg_index, neighbor_index] += -p_CM_coeff * self.Bz_b[ each_r+0, each_z+0 ] * dt

        # CSR format
        self.dNcsr = self.dN.tocsr()
        self.dPcsr = self.dP.tocsr()

        # CPU time
        end = time.time()

        # CPU time
        return end-start

    # ===== solving continuity equation  =====
    def solve_continuity_equation(self, dt, output_filename=False):
        # CPU time
        start = time.time()
        
        # updating N, P matrix for continuity equation
        self.make_N_P_matrix(dt)
        
        # sparse matrix solver for continuity equation
        self.n1 = sc.sparse.linalg.spsolve(self.Ncsr + self.dNcsr, self.n1 )
        self.p1 = sc.sparse.linalg.spsolve(self.Pcsr + self.dPcsr, self.p1 )

        # 2D visualization 
        self.n2 = self.n1.reshape(self.Z_nodes_len, self.R_nodes_len).T
        self.p2 = self.p1.reshape(self.Z_nodes_len, self.R_nodes_len).T
        
        # debugging
        if output_filename != False:
            #
            fig, ax = plt.subplots(2, 2, figsize=(10,8))
            ax00 = ax[0,0].imshow(self.V2, origin='lower')
            ax[0,0].set_title('electric potential')
            plt.colorbar(ax00)
            #
            ax01 = ax[0,1].imshow(self.E, origin='lower')
            ax[0,1].set_title('electric field')
            plt.colorbar(ax01)
            #
            ax10 = ax[1,0].imshow(self.n2, origin='lower')
            ax[1,0].set_title('electron density')
            plt.colorbar(ax10)
            #
            ax11 = ax[1,1].imshow(self.p2, origin='lower')
            ax[1,1].set_title('hole density')
            plt.colorbar(ax11)
            #
            plt.savefig(output_filename)
            #
            plt.close()

        # CPU time
        end = time.time()

        # CPU time
        return end-start

    # ===== calculating BL SL current  =====
    def cal_bl_sl_current(self, bl_mat_no, sl_mat_no):
        # CPU time
        start = time.time()
        
        # initialization
        In_bl, Ip_bl, In_sl, Ip_sl = 0.0, 0.0, 0.0, 0.0
        
        # get BL ohmic contact points list
        bl_points, sl_points  = list(self.RZ_MIS['M'][bl_mat_no]), list(self.RZ_MIS['M'][sl_mat_no])
        
        # check every BL ohmic contact points
        for each_r_index, each_z_index in bl_points:
            # calculate perimeter
            perimeter = 2.0 * np.pi * self.RZ_R[each_r_index, each_z_index]
            # calculate area
            area = perimeter * self.RZ_dR[each_r_index, each_z_index]
            # calculate Jn_bl, Jp_bl
            Jn_bl  = +self.q * self.MAT['SI']['mu_n'] * self.Vtm / self.RZ_dZ[each_r_index, each_z_index]
            Jn_bl *= ( self.Bz_f[each_r_index, each_z_index] * self.n2[each_r_index, each_z_index+1] - \
                       self.Bz_b[each_r_index, each_z_index] * self.n2[each_r_index, each_z_index+0] )
            Jp_bl  = -self.q * self.MAT['SI']['mu_p'] * self.Vtm / self.RZ_dZ[each_r_index, each_z_index]
            Jp_bl *= ( self.Bz_b[each_r_index, each_z_index] * self.p2[each_r_index, each_z_index+1] - \
                       self.Bz_f[each_r_index, each_z_index] * self.p2[each_r_index, each_z_index+0] )
            # calculate I_bl, I_bl
            In_bl += area * Jn_bl
            Ip_bl += area * Jp_bl
            
        # check every SL ohmic contact points
        for each_r_index, each_z_index in sl_points:
            # calculate perimeter
            perimeter = 2.0 * np.pi * self.RZ_R[each_r_index, each_z_index]
            # calculate area
            area = perimeter * self.RZ_dR[each_r_index, each_z_index-1]
            # calculate Jn_sl, Jp_sl
            Jn_sl  = +self.q * self.MAT['SI']['mu_n'] * self.Vtm / self.RZ_dZ[each_r_index, each_z_index-1]
            Jn_sl *= ( self.Bz_f[each_r_index, each_z_index-1] * self.n2[each_r_index, each_z_index+0] - \
                       self.Bz_b[each_r_index, each_z_index-1] * self.n2[each_r_index, each_z_index-1] )
            Jp_sl  = -self.q * self.MAT['SI']['mu_p'] * self.Vtm / self.RZ_dZ[each_r_index, each_z_index-1]
            Jp_sl *= ( self.Bz_b[each_r_index, each_z_index-1] * self.p2[each_r_index, each_z_index+0] - \
                       self.Bz_f[each_r_index, each_z_index-1] * self.p2[each_r_index, each_z_index-1] )
            # calculate I_bl, I_bl
            In_sl += area * Jn_sl
            Ip_sl += area * Jp_sl
            
        # return
        return [In_bl, Ip_bl, In_sl, Ip_sl]


#
# MAIN
#

# number of wls
wl_ea = 7

# material para
mat_para_dictionary = {}
mat_para_dictionary['TOX']       = {'mat_no':30,  'type':'I', 'k':4.8,  'qf':0.0}
mat_para_dictionary['CTN']       = {'mat_no':31,  'type':'I', 'k':7.5,  'qf':0.0}
mat_para_dictionary['BOX_SIO2']  = {'mat_no':32,  'type':'I', 'k':5.0,  'qf':0.0}
mat_para_dictionary['BOX_AL2O3'] = {'mat_no':33,  'type':'I', 'k':9.0,  'qf':0.0}
mat_para_dictionary['LINER']     = {'mat_no':11,  'type':'I', 'k':3.9,  'qf':0.0}
mat_para_dictionary['VOID']      = {'mat_no':10,  'type':'I', 'k':1.0,  'qf':0.0}
mat_para_dictionary['ON_SIO2']   = {'mat_no':34,  'type':'I', 'k':3.9,  'qf':0.0}
mat_para_dictionary['SI']        = {'mat_no':20,  'type':'S', 'k':11.7, 'qf':0.0, \
                                    'n_int':1.5e16, 'mu_n':0.14, 'mu_p':0.045, 'tau_n':1e-6, 'tau_p':1e-5}
for each_wl in range(wl_ea):
    each_wl_name = 'WL%03i' % each_wl
    each_wl_no   = 100 + each_wl
    mat_para_dictionary[each_wl_name]    = {'mat_no':each_wl_no, 'type':'M', 'k':1e5,  'qf':0.0, 'wf':4.8}

# inside plug
uc_inward_thk_dr = {}
uc_inward_thk_dr['CD']         = 1200                                       # angstrom
uc_inward_thk_dr['BOX_SIO2']   = {'mat_no':32, 'thk':70.0,  'dr':5.0}       # angstrom (1st layer)
uc_inward_thk_dr['CTN']        = {'mat_no':31, 'thk':50.0,  'dr':5.0}       # angstrom (2nd layer)
uc_inward_thk_dr['TOX']        = {'mat_no':30, 'thk':50.0,  'dr':5.0}       # angstrom (3rd layer)
uc_inward_thk_dr['SI']         = {'mat_no':20, 'thk':70.0,  'dr':5.0}       # angstrom (4th layer)
uc_inward_thk_dr['LINER']      = {'mat_no':11, 'thk':120.0, 'dr':5.0}       # angstrom (5th layer)
uc_inward_thk_dr['VOID']       = {'mat_no':10, 'thk':-1,    'dr':5.0}       # angstrom (6th layer)

# outside plug & z stacks
uc_outward_thk_dr = {}
uc_z_on_thk_dz = {}

for each_wl in range(wl_ea):
    each_wl_name = 'WL%03i' % each_wl
    each_wl_no   = 100 + each_wl
    #
    uc_outward_thk_dr[each_wl_name+'_ON_O1'] = {}
    uc_outward_thk_dr[each_wl_name+'_ON_O1']['ON_SIO2']    = {'mat_no':34,         'thk':100.0, 'dr':5.0}     # angstrom (1st layer)
    uc_z_on_thk_dz[   each_wl_name+'_ON_O1'] = {'thk':90.0,  'dz':5.0}   # angstrom
    #
    uc_outward_thk_dr[each_wl_name+'_ON_N1'] = {}
    uc_outward_thk_dr[each_wl_name+'_ON_N1']['BOX_AL2O3']  = {'mat_no':33,         'thk':100.0, 'dr':5.0}     # angstrom (1st layer)
    uc_z_on_thk_dz[   each_wl_name+'_ON_N1'] = {'thk':30.0,  'dz':5.0}   # angstrom
    #
    uc_outward_thk_dr[each_wl_name+'_ON_N2'] = {}
    uc_outward_thk_dr[each_wl_name+'_ON_N2']['BOX_AL2O3']  = {'mat_no':33,         'thk':30.0,  'dr':5.0}     # angstrom (1st layer)
    uc_outward_thk_dr[each_wl_name+'_ON_N2'][each_wl_name] = {'mat_no':each_wl_no, 'thk':70.0,  'dr':5.0}     # angstrom (2nd layer)
    uc_z_on_thk_dz[   each_wl_name+'_ON_N2'] = {'thk':205.0, 'dz':5.0}   # angstrom
    #
    uc_outward_thk_dr[each_wl_name+'_ON_N3'] = {}
    uc_outward_thk_dr[each_wl_name+'_ON_N3']['BOX_AL2O3']  = {'mat_no':33,         'thk':100.0, 'dr':5.0}     # angstrom (1st layer)
    uc_z_on_thk_dz[   each_wl_name+'_ON_N3'] = {'thk':30.0,  'dz':5.0}   # angstrom
    #
    uc_outward_thk_dr[each_wl_name+'_ON_O2'] = {}
    uc_outward_thk_dr[each_wl_name+'_ON_O2']['ON_SIO2']    = {'mat_no':34,         'thk':100.0, 'dr':5.0}     # angstrom (1st layer)
    uc_z_on_thk_dz[   each_wl_name+'_ON_O2'] = {'thk':95.0,  'dz':5.0}   # angstrom

# preparing grid
grid_solver = SOLVER()
cpu_time_1 = grid_solver.add_material_parameters(mat_para_dictionary)
cpu_time_2 = grid_solver.set_unit_cell_R_grid(inward_thk_dr=uc_inward_thk_dr, outward_thk_dr=uc_outward_thk_dr)
cpu_time_3 = grid_solver.set_unit_cell_Z_grid(z_on_thk_dz=uc_z_on_thk_dz, z_offset=0.0)
cpu_time_4 = grid_solver.set_unit_cell_RZ_grid()
cpu_time_5 = grid_solver.set_unit_cell_RZ_mis_region()
cpu_time_6 = grid_solver.add_ohmic_contact(si_mat_no=20, bl_mat_no=10001, sl_mat_no=10002)
cpu_time_7 = grid_solver.set_semiconductor_parameters(op_temperature=25.0, si_mat_no=20, bl_mat_no=10001, sl_mat_no=10002, doping=['n', 1e20])
cpu_time_8 = grid_solver.make_poisson_matrix(bl_mat_no=10001, sl_mat_no=10002)
cpu_time_9 = grid_solver.make_continuity_matrix()

# poisson equation solver
bl_bias, sl_bias, wl_bias = 0.0, 0.0, 1.0
ext_bias = {10001:bl_bias, 10002:sl_bias}                                                           # BL, SL ext. bias
for each_wl in range(wl_ea):
    each_wl_mat_no = 100 + each_wl
    ext_bias.update({each_wl_mat_no:wl_bias})                                                       # WL ext. bias
cpu_time_10 = grid_solver.make_external_bias_vector(external_bias_conditions=ext_bias)
ctn_fixed_charge_density = {31:0.0e24}                                                              # fixed charge
cpu_time_11 = grid_solver.make_fixed_charge_vector(fixed_charge_density=ctn_fixed_charge_density)
cpu_time_12 = grid_solver.solve_poisson_equation()

# visualization
if False:
    fig, ax = plt.subplots(2, 2)
    ax00 = ax[0,0].imshow(grid_solver.V2, origin='lower')
    ax01 = ax[0,1].imshow(grid_solver.E, origin='lower')
    ax10 = ax[1,0].imshow(grid_solver.Er, origin='lower')
    ax11 = ax[1,1].imshow(grid_solver.Ez, origin='lower')
    plt.colorbar(ax00)
    plt.colorbar(ax01)
    plt.colorbar(ax10)
    plt.colorbar(ax11)
    plt.show()

# continuity equation solver
cpu_time_13 = grid_solver.make_N_P_matrix(dt=1e-10)
cpu_time_14 = grid_solver.solve_continuity_equation(dt=1e-10, output_filename=False)
cpu_time_15 = grid_solver.solve_poisson_equation()

# visualization
if True:
    fig, ax = plt.subplots(2, 2)
    ax00 = ax[0,0].imshow(grid_solver.V2, origin='lower')
    ax01 = ax[0,1].imshow(grid_solver.E, origin='lower')
    ax10 = ax[1,0].imshow(grid_solver.n2, origin='lower')
    ax11 = ax[1,1].imshow(grid_solver.p2, origin='lower')
    plt.colorbar(ax00)
    plt.colorbar(ax01)
    plt.colorbar(ax10)
    plt.colorbar(ax11)
    plt.show()

# FDM size
print('FDM size')
print('  R nodes = %i, Z nodes = %i' % (grid_solver.R_nodes_len, grid_solver.Z_nodes_len))
print('  RZ nodes = %i (sparse matrix size)' % (grid_solver.RZ_nodes_len))

# CPU time check
print('CPU time check list')
print('  %s %s %s threads' % (cpuinfo.get_cpu_info()['brand_raw'], platform.processor(), psutil.cpu_count(logical=True)))
print('  @add_material_parameters() = %.1e sec' % cpu_time_1)
print('  @set_unit_cell_R_grid() = %.1e sec' % cpu_time_2)
print('  @set_unit_cell_Z_grid() = %.1e sec' % cpu_time_3)
print('  @set_unit_cell_RZ_grid() = %.1e sec' % cpu_time_4)
print('  @set_unit_cell_RZ_mis_region() = %.1e sec' % cpu_time_5)
print('  @add_ohmic_contact() = %.1e sec' % cpu_time_6)
print('  @set_semiconductor_parameters() = %.1e sec' % cpu_time_7)
print('  @make_poisson_matrix() = %.1e sec' % cpu_time_8)
print('  @make_continuity_matrix() = %.1e sec' % cpu_time_9)
print('  @make_external_bias_vector() = %.1e sec' % cpu_time_10)
print('  @make_fixed_charge_vector() = %.1e sec' % cpu_time_11)
print('  @solve_poisson_equation() = %.1e sec' % cpu_time_12)
print('  @make_N_P_matrix() = %.1e sec' % cpu_time_13)
print('  @solve_continuity_equation() = %.1e sec' % (cpu_time_14-cpu_time_13))
print('  @solve_poisson_equation() = %.1e sec' % cpu_time_15)

# Poisson equation solver check
print('Poisson equation solver check list')
print('  @V   = [%.3f %.3f]' % (np.min(grid_solver.V1), np.max(grid_solver.V1)))
print('  @E   = [%.3e %.3e]' % (np.min(grid_solver.E), np.max(grid_solver.E)))
print('  @Er  = [%.3e %.3e]' % (np.min(grid_solver.Er), np.max(grid_solver.Er)))
print('  @Ez  = [%.3e %.3e]' % (np.min(grid_solver.Ez), np.max(grid_solver.Ez)))
print('  @Vbi = [%.3f %.3f]' % (np.min(grid_solver.Vbi), np.max(grid_solver.Vbi)))

# Continuity equation solver check
print('Continuity equation solver check list')
print('  @dN   = [%.3e %.3e]' % (np.min(grid_solver.dNcsr), np.max(grid_solver.dNcsr)))
print('  @dP   = [%.3e %.3e]' % (np.min(grid_solver.dPcsr), np.max(grid_solver.dPcsr)))







if False:
    # WL bias sweep
    WL_sweep_range = np.linspace(0.0, 4.0, 41)
    WL_sweep_range = np.linspace(1.5, 3.0, 16)

    for WL_bias in WL_sweep_range:
        # CPU time
        start = time.time()
        
        # poisson equation solver

        # injected charge
        Qn_bl, Qp_bl, Qn_sl, Qp_sl = 0.0, 0.0, 0.0, 0.0

        # error control
        prev_Qn_bl, prev_Qp_bl, prev_Qn_sl, prev_Qp_sl = 1.0, 1.0, 1.0, 1.0
        err_Qn_bl,  err_Qp_bl,  err_Qn_sl,  err_Qp_sl  = 1.0, 1.0, 1.0, 1.0

        # SG scheme
        timeline = np.linspace(1e-15, 5e-10, 5001)
        for index, each_time in enumerate(list(timeline)):
            # calculating dt, elapsed time
            if index == 0:
                dt = each_time
            else:
                dt = each_time - timeline[index-1]
            # output filename
            if index % 500 == 0:
                output_filename = 'SG_scheme_%05i_Vg_%.2f_elapsed_time_%.2e_dt_%.2e.png' % (index, WL_bias, each_time, dt)
            else:
                output_filename = False
                
            # poission equation solver
            grid_solver.solve_poisson_equation()
            # continuity equation solver
            grid_solver.solve_continuity_equation(dt=dt, output_filename=output_filename)
            
            # calculate BL current
            In_bl, Ip_bl, In_sl, Ip_sl = grid_solver.cal_bl_sl_current(bl_mat_no=10001, sl_mat_no=10002)
            Qn_bl += In_bl * dt
            Qp_bl += Ip_bl * dt
            Qn_sl += In_sl * dt
            Qp_sl += Ip_sl * dt

            # error control
            err_Qn_bl, err_Qp_bl = np.abs(Qn_bl-prev_Qn_bl)/np.absolute(Qn_bl), np.abs(Qp_bl-prev_Qp_bl)/np.absolute(Qp_bl)
            err_Qn_sl, err_Qp_sl = np.abs(Qn_sl-prev_Qn_sl)/np.absolute(Qn_sl), np.abs(Qp_sl-prev_Qp_sl)/np.absolute(Qp_sl)
            if (err_Qn_bl < 1e-5) and (err_Qn_sl < 1e-5):
                break
            else:
                prev_Qn_bl, prev_Qp_bl = Qn_bl, Qp_bl
                prev_Qn_sl, prev_Qp_sl = Qn_sl, Qp_sl
            
            # progress check
            if index % 100 == 0:
                output_string = '%i,%.2f,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e' % \
                                (index, WL_bias, each_time, dt, Qn_bl, Qp_bl, Qn_sl, Qp_sl)
                print(output_string)
            
        # CPU time
        end = time.time()
        output_string = ' CPU time %.3f sec (%s),%.2f,%i,%.3e,%.3e' % \
                        (end-start, time.ctime(), WL_bias, index, Qn_bl, Qn_sl)
        print(output_string)



