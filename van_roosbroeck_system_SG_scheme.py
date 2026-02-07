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
        # poisson equation (1813) solution (for SOLVER class)
        self.V1 = []        # electric potential 1D (sparse matrix solution)
        self.V2 = []        # electric potential 2D
        self.E  = []        # electric field magnitude 2D
        self.Er = []        # electric field r direction 2D
        self.Ez = []        # electric field z direction 2D
        self.EB = []        # external bias vector
        self.FC = []        # fixed charge density vector
        
        # semiconductor continuity equation (1950) solution (for SOLVER class)
        self.n1 = []        # electron density 1D (sparse matrix solution)
        self.p1 = []        # hole density 1D (sparse matrix solution)
        self.n2 = []        # electron density 2D
        self.p2 = []        # hole density 2D
        self.Jn = []        # electron current density magnitude 2D
        self.Jp = []        # hole current density magnitude 2D
        self.Jn_r = []      # electron current density in r direction 2D
        self.Jn_z = []      # electron current density in z direction 2D
        self.Jp_r = []      # hole current density in r direction 2D
        self.Jp_z = []      # hole current density in z direction 2D

    # ===== adding material parameters =====
    def add_material_parameters(self, mat_para_dictionary):
        # user input
        self.MAT = mat_para_dictionary

        # key: mis, value: mat. no. array
        self.MIS_MAT_no = {}
        for mat_name in self.MAT.keys():
            mat_mis = self.MAT[mat_name]['type']
            mat_no  = self.MAT[mat_name]['mat_no']
            #
            if mat_mis not in self.MIS_MAT_no.keys():
                self.MIS_MAT_no[mat_mis] = []
            #
            self.MIS_MAT_no[mat_mis].append(mat_no)

        # debugging
        if True:
            print('debugging > add_material_parameters()')
            print(self.MIS_MAT_no)
    
    # ===== setting unit cell R direction grid (angstrom) =====
    def set_unit_cell_R_grid(self, inward_thk_dr, outward_thk_dr):
        # user input
        self.R_inward  = inward_thk_dr
        self.R_outward = outward_thk_dr
        
        # r coordinate (angstrom)
        self.cd = 0.0
        self.R_IN = []
        # r material name & number
        self.R_IN_MAT_name = []
        self.R_IN_MAT_mis = []
        self.R_IN_MAT_ep = []
        self.R_IN_MAT_no = []
        
        # check inward_thk_dr
        for each_layer in self.R_inward.keys():
            # CD
            if each_layer == 'CD':
                self.cd = self.R_inward[each_layer]
                self.R_IN.append(self.cd/2.0)
            # each layer
            else:
                # each layer information
                mat_no = self.R_inward[each_layer]['mat_no']    # float
                mis = self.MAT[each_layer]['type']              # string
                ep = self.ep0 * self.MAT[each_layer]['k']       # float
                thk = self.R_inward[each_layer]['thk']          # float
                dr = self.R_inward[each_layer]['dr']            # float
                
                # float
                if thk == -1:
                    r_array = list(np.arange(0.0, self.R_IN[0], dr))
                    self.R_IN = r_array + self.R_IN
                else:
                    r_array = list(np.arange(self.R_IN[0]-thk, self.R_IN[0], dr))
                    self.R_IN = r_array + self.R_IN
                # string
                mat_name_array = [each_layer] * len(r_array)
                self.R_IN_MAT_name = mat_name_array + self.R_IN_MAT_name
                # string
                mis_array = [mis] * len(r_array)
                self.R_IN_MAT_mis = mis_array + self.R_IN_MAT_mis
                # float
                mat_ep_array = [ep] * len(r_array)
                self.R_IN_MAT_ep = mat_ep_array + self.R_IN_MAT_ep
                # float
                mat_no_array = [mat_no] * len(r_array)
                self.R_IN_MAT_no = mat_no_array + self.R_IN_MAT_no
        
        # r coordinate (angstrom)
        self.R_OUT = {}
        # r material name & number
        self.R_OUT_MAT_name = {}
        self.R_OUT_MAT_mis = {}
        self.R_OUT_MAT_ep = {}
        self.R_OUT_MAT_no = {}

        # check outward_thk_dr
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
                mat_no = self.R_outward[each_region][each_layer]['mat_no']      # float
                mis = self.MAT[each_layer]['type']                              # string
                ep = self.ep0 * self.MAT[each_layer]['k']                       # float
                thk = self.R_outward[each_region][each_layer]['thk']            # float
                dr = self.R_outward[each_region][each_layer]['dr']              # float
                
                # float
                if each_index == 0:
                    r_array = list(np.arange(self.cd/2.0+dr, self.cd/2.0+dr+thk, dr))
                else:
                    r_array = list(np.arange(self.R_OUT[each_region][-1]+dr, self.R_OUT[each_region][-1]+dr+thk, dr))
                self.R_OUT[each_region] = self.R_OUT[each_region] + r_array
                # string
                mat_name_array = [each_layer] * len(r_array)
                self.R_OUT_MAT_name[each_region] = self.R_OUT_MAT_name[each_region] + mat_name_array
                # string
                mis_array = [mis] * len(r_array)
                self.R_OUT_MAT_mis[each_region] = self.R_OUT_MAT_mis[each_region] + mis_array
                # float
                ep_array = [ep] * len(r_array)
                self.R_OUT_MAT_ep[each_region] = self.R_OUT_MAT_ep[each_region] + ep_array
                # float
                mat_no_array = [mat_no] * len(r_array)
                self.R_OUT_MAT_no[each_region] = self.R_OUT_MAT_no[each_region] + mat_no_array

        # r coordinate (angstrom)
        self.R = {}
        # r material name & number
        self.R_MAT_name = {}
        self.R_MAT_mis = {}
        self.R_MAT_ep = {}
        self.R_MAT_no = {}

        # merge > inward_thk_dr + outward_thk_dr
        for each_region in self.R_OUT.keys():
            # merge
            self.R[each_region] = self.R_IN + self.R_OUT[each_region]                               # node
            self.R_MAT_name[each_region] = self.R_IN_MAT_name + self.R_OUT_MAT_name[each_region]    # element
            self.R_MAT_mis[each_region] = self.R_IN_MAT_mis + self.R_OUT_MAT_mis[each_region]       # element
            self.R_MAT_ep[each_region] = self.R_IN_MAT_ep + self.R_OUT_MAT_ep[each_region]          # element
            self.R_MAT_no[each_region] = self.R_IN_MAT_no + self.R_OUT_MAT_no[each_region]          # element

        # debugging
        if True:
            print('debugging > set_unit_cell_R_grid()')
            for each_region in self.R.keys():
                print(each_region, len(self.R[each_region]), \
                      len(self.R_MAT_name[each_region]), len(self.R_MAT_mis[each_region]), \
                      len(self.R_MAT_ep[each_region]), len(self.R_MAT_no[each_region]))
                #print(self.R[each_region])
                #print(self.R_MAT_name[each_region])
                #print(self.R_MAT_mis[each_region])
                #print(self.R_MAT_ep[each_region])
                #print(self.R_MAT_no[each_region])
                #plt.plot(self.R[each_region][1:],self.R_MAT_no[each_region],'o-')
                #plt.show()

    # ===== setting unit cell Z direction grid (angstrom) =====
    def set_unit_cell_Z_grid(self, z_on_thk_dz, z_offset):
        # user input
        self.Z_stack = z_on_thk_dz
        self.Z_start = z_offset             # angstrom
        
        # z coordinate (angstrom)
        self.Z = [self.Z_start]             # node
        # z region name & mis flag
        self.Z_REGION = []                  # element
        self.Z_MAT_mis = []                 # element
        self.Z_MAT_ep = []                  # element
        self.Z_MAT_no = []                  # element

        #
        for index, each_region in enumerate(list(self.Z_stack.keys())):
            # each region information
            thk = self.Z_stack[each_region]['thk']
            dz = self.Z_stack[each_region]['dz']
            
            #
            if index == 0:
                z_array = list( np.arange(self.Z[0]+dz, self.Z[0]+dz+thk, dz) )
                self.Z = self.Z + z_array
            else:
                z_array =list( np.arange(self.Z[-1]+dz, self.Z[-1]+dz+thk, dz) )
                self.Z = self.Z + z_array
            # material
            z_region_array = [each_region] * len(z_array)
            self.Z_REGION = self.Z_REGION + z_region_array
            # material
            for each_z in z_array:
                self.Z_MAT_mis.append( self.R_MAT_mis[each_region] )        # element
                self.Z_MAT_ep.append( self.R_MAT_ep[each_region] )          # element
                self.Z_MAT_no.append( self.R_MAT_no[each_region] )          # element
 
        # debugging
        if True:
            print('debugging > set_unit_cell_Z_grid()')
            print(len(self.Z), len(self.Z_REGION))
            #print(set(self.Z))
            #print(set(self.Z_REGION))

    # ===== setting unit cell RZ 2D grid (angstrom) =====
    def set_unit_cell_RZ_grid(self):
        # 1D array (angstrom)
        self.R_nodes = copy.copy( self.R[ list(self.R.keys())[0] ] )
        self.Z_nodes = copy.copy( self.Z )

        # 1D array length
        self.R_nodes_len = len(self.R_nodes)
        self.Z_nodes_len = len(self.Z_nodes)
        self.R_elmts_len = self.R_nodes_len - 1
        self.Z_elmts_len = self.Z_nodes_len - 1

        # sparse matrix size
        self.RZ_nodes_len = self.R_nodes_len * self.Z_nodes_len
        
        # 2D array
        self.RZ_R = np.zeros([self.R_nodes_len, self.Z_nodes_len])
        self.RZ_Z = np.zeros([self.R_nodes_len, self.Z_nodes_len])

        # 2D array R (angstrom -> m)
        for each_z_node in range(self.Z_nodes_len):
            self.RZ_R[:,each_z_node] = copy.copy( self.R_nodes )
        self.RZ_R *= 1e-10
        self.RZ_dR = self.RZ_R[1:,:] - self.RZ_R[:-1,:]
        
        # 2D array Z (angstrom -> m)
        for each_r_node in range(self.R_nodes_len):
            self.RZ_Z[each_r_node,:] = copy.copy( self.Z_nodes )
        self.RZ_Z *= 1e-10
        self.RZ_dZ = self.RZ_Z[:,1:] - self.RZ_Z[:,:-1]

        # 2D array
        self.RZ_EP = np.zeros([self.R_elmts_len, self.Z_elmts_len])
        for each_z_elmt in range(self.Z_elmts_len):
            self.RZ_EP[:,each_z_elmt] = copy.copy( self.Z_MAT_ep[each_z_elmt] )

        # 2D array
        self.RZ_MATno = np.zeros([self.R_elmts_len, self.Z_elmts_len])
        for each_z_elmt in range(self.Z_elmts_len):
            self.RZ_MATno[:,each_z_elmt] = copy.copy( self.Z_MAT_no[each_z_elmt] )

        # debugging
        if True:
            print('debugging > set_unit_cell_RZ_grid()')
            print(self.RZ_R.shape)
            print(self.RZ_Z.shape)
            print(self.RZ_EP.shape)

    # ===== setting metal-insulator-semiconductor region =====
    def set_unit_cell_RZ_mis_region(self):
        #
        self.RZ_MIS = {}

        # sweep mat_mis keys
        for each_mat_mis in self.MIS_MAT_no.keys():
            # check each mat_mis key
            if each_mat_mis not in self.RZ_MIS.keys():
                # if not, make mat_mis key
                self.RZ_MIS[each_mat_mis] = {}
                # check each mat_no array
                for each_mat_no in self.MIS_MAT_no[each_mat_mis]:
                    # if not, make mat_no set
                    self.RZ_MIS[each_mat_mis][each_mat_no] = set()
                    # sweep RZ coordinate
                    for each_z_elmt in range(self.Z_elmts_len):
                        for each_r_elmt in range(self.R_elmts_len):
                            # check RZ coordinate
                            if (self.Z_MAT_mis[each_z_elmt][each_r_elmt] == each_mat_mis) and (self.Z_MAT_no[each_z_elmt][each_r_elmt] == each_mat_no):
                                self.RZ_MIS[each_mat_mis][each_mat_no].add( (each_r_elmt+0, each_z_elmt+0) )     # element
                                self.RZ_MIS[each_mat_mis][each_mat_no].add( (each_r_elmt+1, each_z_elmt+0) )     # adding node
                                self.RZ_MIS[each_mat_mis][each_mat_no].add( (each_r_elmt+0, each_z_elmt+1) )     # adding node
                                self.RZ_MIS[each_mat_mis][each_mat_no].add( (each_r_elmt+1, each_z_elmt+1) )     # adding node

        # debugging
        if True:
            print('debugging > set_unit_cell_RZ_mis_region()')
            for each_mat_mis in self.RZ_MIS.keys():
                for each_mat_no in self.RZ_MIS[each_mat_mis]:
                    print(each_mat_mis, each_mat_no, len(self.RZ_MIS[each_mat_mis][each_mat_no]))

    # ===== adding ohmic contact =====
    def add_ohmic_contact(self, before_info, after_info):
        # 
        before_mat_mis = list(before_info.keys())[0]
        before_mat_no  = before_info[before_mat_mis]['mat_no']
        before_z_coord = before_info[before_mat_mis]['z_coord']
        if before_z_coord == -1:
            before_z_coord = self.Z_nodes_len-1

        #
        after_mat_mis = list(after_info.keys())[0]
        after_mat_no  = after_info[after_mat_mis]['mat_no']

        # check nodes
        add_nodes = set()
        for each_r_node, each_z_node in list(self.RZ_MIS[before_mat_mis][before_mat_no]):
            if each_z_node == before_z_coord:
                add_nodes.add((each_r_node, each_z_node))

        # change nodes
        self.RZ_MIS[before_mat_mis][before_mat_no] = self.RZ_MIS[before_mat_mis][before_mat_no].difference(add_nodes)
        self.RZ_MIS[after_mat_mis][after_mat_no] = set()
        self.RZ_MIS[after_mat_mis][after_mat_no] = self.RZ_MIS[after_mat_mis][after_mat_no].union(add_nodes)

        # debugging
        if True:
            print('debugging > add_ohmic_contact()')
            print(before_mat_mis, before_mat_no, before_z_coord)
            print(len(self.RZ_MIS[before_mat_mis][before_mat_no]), -len(self.RZ_MIS[after_mat_mis][after_mat_no]))
        
    # ===== making poisson matrix  =====
    def make_poisson_matrix(self):
        # making sparse matrix
        self.PM = sc.sparse.dok_matrix((self.RZ_nodes_len, self.RZ_nodes_len))

        # dirichlet boundary conditions
        for each_mat_no in self.RZ_MIS['M'].keys():
            for each_r, each_z in self.RZ_MIS['M'][each_mat_no]:
                # 1D serialization index
                index_r_z = self.R_nodes_len * (each_z+0) + (each_r+0)
                # metal contact
                self.PM[index_r_z, index_r_z] = 1.0

        # neumann boundary conditions
        for each_z in [0, self.Z_nodes_len-1]:
            for each_r in range(self.R_nodes_len):
                # 1D serialization index
                index_r_z   = self.R_nodes_len * (each_z+0) + (each_r+0)
                index_r_zm1 = self.R_nodes_len * (each_z-1) + (each_r+0)
                index_r_zp1 = self.R_nodes_len * (each_z+1) + (each_r+0)
                # except metal contacts
                if self.PM[index_r_z, index_r_z] != 1:
                    #
                    if each_z == 0:
                        self.PM[index_r_z, index_r_z  ] = +1.0
                        self.PM[index_r_z, index_r_zp1] = -1.0
                    #
                    if each_z == (self.Z_nodes_len-1):
                        self.PM[index_r_z, index_r_z  ] = +1.0
                        self.PM[index_r_z, index_r_zm1] = -1.0

        # neumann boundary conditions
        for each_r in [0, self.R_nodes_len-1]:
            for each_z in range(self.Z_nodes_len):
                # 1D serialization index
                index_r_z   = self.R_nodes_len * (each_z+0) + (each_r+0)
                index_rm1_z = self.R_nodes_len * (each_z+0) + (each_r-1)
                index_rp1_z = self.R_nodes_len * (each_z+0) + (each_r+1)
                # except metal contacts & boundary conditions
                if self.PM[index_r_z, index_r_z] != 1:
                    #
                    if each_r == 0:
                        self.PM[index_r_z, index_r_z  ] = +1.0
                        self.PM[index_r_z, index_rp1_z] = -1.0
                    #
                    if each_r == (self.R_nodes_len-1):
                        self.PM[index_r_z, index_r_z  ] = +1.0
                        self.PM[index_r_z, index_rm1_z] = -1.0

        # inside boundary conditions
        for each_r in range(1, self.R_nodes_len):
            for each_z in range(1, self.Z_nodes_len):
                # 1D serialization index
                index_r_z   = self.R_nodes_len * (each_z+0) + (each_r+0)
                index_rm1_z = self.R_nodes_len * (each_z+0) + (each_r-1)
                index_rp1_z = self.R_nodes_len * (each_z+0) + (each_r+1)
                index_r_zm1 = self.R_nodes_len * (each_z-1) + (each_r+0)
                index_r_zp1 = self.R_nodes_len * (each_z+1) + (each_r+0)
                # except metal contacts & boundary conditions
                if self.PM[index_r_z, index_r_z] != 1:
                    # geometry factors in r direction
                    geometry_effect_rm1_z  = self.RZ_R[each_r+0,each_z+0] - ( self.RZ_R[each_r+0,each_z+0] - self.RZ_R[each_r-1,each_z+0] ) / 2.0 
                    geometry_effect_rm1_z /= self.RZ_R[each_r+0,each_z+0]
                    geometry_effect_rp1_z  = self.RZ_R[each_r+0,each_z+0] + ( self.RZ_R[each_r+1,each_z+0] - self.RZ_R[each_r+0,each_z+0] ) / 2.0 
                    geometry_effect_rp1_z /= self.RZ_R[each_r+0,each_z+0]
                    geometry_effect_r_zm1  = 1.0
                    geometry_effect_r_zp1  = 1.0
                    # 2nd derivatives
                    geometry_effect_rm1_z /= (self.RZ_R[each_r+0,each_z+0] - self.RZ_R[each_r-1,each_z+0])
                    geometry_effect_rm1_z /= (self.RZ_R[each_r+1,each_z+0] - self.RZ_R[each_r-1,each_z+0])/2.0
                    geometry_effect_rp1_z /= (self.RZ_R[each_r+1,each_z+0] - self.RZ_R[each_r+0,each_z+0])
                    geometry_effect_rp1_z /= (self.RZ_R[each_r+1,each_z+0] - self.RZ_R[each_r-1,each_z+0])/2.0
                    geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+0] - self.RZ_Z[each_r+0,each_z-1])
                    geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+1] - self.RZ_Z[each_r+0,each_z-1])/2.0
                    geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+1] - self.RZ_Z[each_r+0,each_z+0])
                    geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+1] - self.RZ_Z[each_r+0,each_z-1])/2.0
                    # electric permittivity
                    ep_z_avg_rm1 = (self.RZ_EP[each_r-1,each_z-1] + self.RZ_EP[each_r-1,each_z+0]) / 2.0
                    ep_z_avg_rp1 = (self.RZ_EP[each_r+0,each_z-1] + self.RZ_EP[each_r+0,each_z+0]) / 2.0
                    ep_r_avg_zm1 = (self.RZ_EP[each_r-1,each_z-1] + self.RZ_EP[each_r+0,each_z-1]) / 2.0
                    ep_r_avg_zp1 = (self.RZ_EP[each_r-1,each_z+0] + self.RZ_EP[each_r+0,each_z+0]) / 2.0
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

        # debugging
        if True:
            print('debugging > make_poisson_matrix()')
            for each_mat_no in self.RZ_MIS['M'].keys():
                print('M', each_mat_no, len(self.RZ_MIS['M'][each_mat_no]))


#
# CLASS: SOLVER (sparse matrix solver)
#

class SOLVER(GRID):

    # ===== making external bias vector  =====
    def make_external_bias_vector(self, external_bias_conditions):
        # sweep
        for each_mat_mis in ['M']:
            for each_mat_no in self.RZ_MIS[each_mat_mis].keys():
                for each_r, each_z in self.RZ_MIS[each_mat_mis][each_mat_no]:
                    # 1D serialization index
                    index_r_z = self.R_nodes_len * (each_z+0) + (each_r+0)
                    # external bias @ metal contact 
                    self.EB[index_r_z] = external_bias_conditions[each_mat_no]

    # ===== making fixed charge vector  =====
    def make_fixed_charge_vector(self, fixed_charge_density):
        # sweep
        for each_mat_mis in ['I', 'S']:
            for each_mat_no in self.RZ_MIS[each_mat_mis].keys():
                # check
                if each_mat_no in fixed_charge_density.keys():
                    for each_r, each_z in self.RZ_MIS[each_mat_mis][each_mat_no]:
                        # 1D serialization index
                        index_r_z = self.R_nodes_len * (each_z+0) + (each_r+0)
                        # fixed charge density 
                        self.FC[index_r_z] = fixed_charge_density[each_mat_no]

    # ===== solving poisson equation  =====
    def solve_poisson_equation(self):
        # sparse matrix solver
        self.V1 = sc.sparse.linalg.spsolve(self.PMcsr, self.EB + self.q*(self.FC))

        # post processing
        self.V2 = self.V1.reshape(self.Z_nodes_len, self.R_nodes_len).T
        self.Er = ( self.V2[1:,:] - self.V2[:-1,:] ) / self.RZ_dR
        self.Ez = ( self.V2[:,1:] - self.V2[:,:-1] ) / self.RZ_dZ
        self.E  = np.sqrt( self.Er[:,:-1]**2 + self.Ez[:-1,:]**2 ) 

        # debugging
        if True:
            fig, ax = plt.subplots(2,1)
            ax[0].imshow(self.V2, origin='lower')
            ax[1].imshow(self.E, origin='lower')
            plt.show()
    
            


#
# MAIN
#

# number of wls
wl_ea = 5

# material para
mat_para_dictionary = {}
mat_para_dictionary['TOX']       = {'mat_no':30,  'type':'I', 'k':4.8,  'qf':0.0}
mat_para_dictionary['CTN']       = {'mat_no':31,  'type':'I', 'k':7.5,  'qf':0.0}
mat_para_dictionary['BOX_SIO2']  = {'mat_no':32,  'type':'I', 'k':5.0,  'qf':0.0}
mat_para_dictionary['BOX_AL2O3'] = {'mat_no':33,  'type':'I', 'k':9.0,  'qf':0.0}
mat_para_dictionary['LINER']     = {'mat_no':11,  'type':'I', 'k':3.9,  'qf':0.0}
mat_para_dictionary['VOID']      = {'mat_no':10,  'type':'I', 'k':1.0,  'qf':0.0}
mat_para_dictionary['ON_SIO2']   = {'mat_no':34,  'type':'I', 'k':3.9,  'qf':0.0}
mat_para_dictionary['SI']        = {'mat_no':20,  'type':'S', 'k':11.7, 'qf':0.0, 'n_int':1.5e16, 'mu_n':0.14, 'mu_p':0.045, 'tau_n':1e-6, 'tau_p':1e-5}
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

#
grid_solver = SOLVER()
grid_solver.add_material_parameters(mat_para_dictionary)
grid_solver.set_unit_cell_R_grid(inward_thk_dr=uc_inward_thk_dr, outward_thk_dr=uc_outward_thk_dr)
grid_solver.set_unit_cell_Z_grid(z_on_thk_dz=uc_z_on_thk_dz, z_offset=0.0)
grid_solver.set_unit_cell_RZ_grid()
grid_solver.set_unit_cell_RZ_mis_region()
grid_solver.add_ohmic_contact(before_info={'S':{'mat_no':20, 'z_coord':0 }}, after_info={'M':{'mat_no':10001}})     # BL
grid_solver.add_ohmic_contact(before_info={'S':{'mat_no':20, 'z_coord':-1}}, after_info={'M':{'mat_no':10002}})     # SL
grid_solver.make_poisson_matrix()

#
ext_bias = {10001:0.0, 10002:0.0, 100:0.0, 101:0.0, 102:1.0, 103:0.0, 104:0.0}
grid_solver.make_external_bias_vector(external_bias_conditions=ext_bias)
ctn_fixed_charge_density = {20:0.0e24}
grid_solver.make_fixed_charge_vector(fixed_charge_density=ctn_fixed_charge_density)
grid_solver.solve_poisson_equation()
