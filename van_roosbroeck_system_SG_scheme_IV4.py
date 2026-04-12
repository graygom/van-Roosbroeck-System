#
# TITLE: solving the van Roosbroeck system numerically
# AUTHOR: Hyunseung Yoo
# PURPOSE: 
# REVISION: 
# REFERENCE: a numerical study of the van Roosbroeck system for semiconductor (SJSU, 2018)
#            multi-physics modeling and simulation of photovoltaic devices and systems (MSU, 2020)
#

import sys, time, copy, psutil    # platform, cpuinfo
import numpy as np
import scipy as sc
import sympy as sp
import matplotlib.cm as cm
import matplotlib.pyplot as plt

#
# CLASS: GRID (finite difference method)
#
# MAT, MIS_MAT_no
# R, R_MAT_name, R_MAT_mis, R_MAT_ep, R_MAT_no
# Z, Z_REGION, Z_MAT_mis, Z_MAT_ep, Z_MAT_no
# R_nodes, Z_nodes
# R_nodes_len, Z_nodes_len, R_elmts_len, Z_elmts_len, RZ_nodes_len, RZ_elmts_len
#
# RZ_R, RZ_dR, RZ_Z, RZ_dZ
# RZ_EP, RZ_MATno, CB_ref, CB_offset
# RZ_MIS, RZ_MIS_index_min_max
#
#
#
#
#
#
#
#

class GRID:

    # fundamental constants
    q = 1.60217663e-19      # electron charge, [C]
    kb = 1.380649e-23       # Boltzmann constant, [m]^2 [kg] [K]^-1 [s]^-2 
    ep0 = 8.854187e-12      # elelctric permittivity of free space, [F] [m]^-1
    me = 9.1093837e-31      # electron mass, [kg]
    h = 6.62607015e-34      # Planck constant, [m]^2 [kg] [s]^-1
    hbar = h / (2.0*np.pi)
    
    # ===== constructor =====
    def __init__(self):
        
        # poisson equation (1813) solution (for SOLVER class, metal & semiconductor)
        self.V1 = []        # electric potential 1D (sparse matrix solution)
        self.V2 = []        # electric potential 2D
        self.E  = []        # electric field magnitude 2D
        self.Er = []        # electric field r direction 2D
        self.Ez = []        # electric field z direction 2D
        self.EB = []        # external bias vector
        self.FC = []        # fixed charge density vector
        
        # poisson equation (1813) solution (for SOLVER class, metal only)
        self.V1_mim = []        # electric potential 1D (sparse matrix solution)
        self.V2_mim = []        # electric potential 2D
        self.E_mim  = []        # electric field magnitude 2D
        self.Er_mim = []        # electric field r direction 2D
        self.Ez_mim = []        # electric field z direction 2D
        self.EB_mim = []        # external bias vector
        self.FC_mim = []        # fixed charge density vector
        
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
                mat_no = self.R_inward[each_layer]['mat_no']    # float
                mis = self.MAT[each_layer]['type']              # string
                ep = self.ep0 * self.MAT[each_layer]['k']       # float
                thk = self.R_inward[each_layer]['thk']          # float
                dr = self.R_inward[each_layer]['dr']            # float
                
                # thickness, float (angstrom)
                if thk == -1:
                    r_array = list(np.arange(0.0, self.R_IN[0], dr))
                    self.R_IN = r_array + self.R_IN                                 # forward adding
                else:
                    r_array = list(np.arange(self.R_IN[0]-thk, self.R_IN[0], dr))
                    self.R_IN = r_array + self.R_IN                                 # forward adding
                    
                # material name, string
                mat_name_array = [each_layer] * len(r_array)
                self.R_IN_MAT_name = mat_name_array + self.R_IN_MAT_name            # forward adding
                
                # material type, string
                mis_array = [mis] * len(r_array)
                self.R_IN_MAT_mis = mis_array + self.R_IN_MAT_mis                   # forward adding
                
                # electric permittivity, float, in SI
                mat_ep_array = [ep] * len(r_array)
                self.R_IN_MAT_ep = mat_ep_array + self.R_IN_MAT_ep                  # forward adding
                
                # material number, integer (identifier)
                mat_no_array = [mat_no] * len(r_array)
                self.R_IN_MAT_no = mat_no_array + self.R_IN_MAT_no                  # forward adding
        
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
                mat_no = self.R_outward[each_region][each_layer]['mat_no']      # float
                mis = self.MAT[each_layer]['type']                              # string
                ep = self.ep0 * self.MAT[each_layer]['k']                       # float
                thk = self.R_outward[each_region][each_layer]['thk']            # float
                dr = self.R_outward[each_region][each_layer]['dr']              # float
                
                # thickness, float (angstrom)
                if each_index == 0:
                    r_array = list(np.arange(self.cd/2.0+dr, self.cd/2.0+dr+thk, dr))
                else:
                    r_array = list(np.arange(self.R_OUT[each_region][-1]+dr, self.R_OUT[each_region][-1]+dr+thk, dr))
                self.R_OUT[each_region] = self.R_OUT[each_region] + r_array                                 # backwrad adding
                
                # material name, string
                mat_name_array = [each_layer] * len(r_array)
                self.R_OUT_MAT_name[each_region] = self.R_OUT_MAT_name[each_region] + mat_name_array        # backwrad adding
                
                # material type, string
                mis_array = [mis] * len(r_array)
                self.R_OUT_MAT_mis[each_region] = self.R_OUT_MAT_mis[each_region] + mis_array               # backwrad adding
                
                # electric permittivity, float, in SI
                ep_array = [ep] * len(r_array)
                self.R_OUT_MAT_ep[each_region] = self.R_OUT_MAT_ep[each_region] + ep_array                  # backwrad adding
                
                # material number, integer (identifier)
                mat_no_array = [mat_no] * len(r_array)
                self.R_OUT_MAT_no[each_region] = self.R_OUT_MAT_no[each_region] + mat_no_array              # backwrad adding

        # STEP 2: r coordinate (angstrom)
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
            if index == 0:              # first layer
                z_array = list( np.arange(self.Z[0]+dz, self.Z[0]+dz+thk, dz) )
                self.Z = self.Z + z_array           # backward adding
            else:                       # others
                z_array =list( np.arange(self.Z[-1]+dz, self.Z[-1]+dz+thk, dz) )
                self.Z = self.Z + z_array           # backward adding
                
            # for elements
            # material name (string), material type (string), electric permittivity (float), material number (integer)
            for each_z in z_array:
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
        self.R_nodes = copy.copy( self.R[ list(self.R.keys())[0] ] )        # one of region
        self.Z_nodes = copy.copy( self.Z )

        # 1D array length (used in array making)
        self.R_nodes_len = len(self.R_nodes)
        self.Z_nodes_len = len(self.Z_nodes)
        self.R_elmts_len = self.R_nodes_len - 1
        self.Z_elmts_len = self.Z_nodes_len - 1

        # sparse matrix size (used in array making)
        self.RZ_nodes_len = self.R_nodes_len * self.Z_nodes_len
        self.RZ_elmts_len = self.R_elmts_len * self.Z_elmts_len

        # debugging
        if True:
            print('RZ nodes = [R %iea, Z %iea], total nodes = %iea' %  (self.R_nodes_len, self.Z_nodes_len, self.RZ_nodes_len))
            print('RZ elememts = [R %iea, Z %iea], total elememts = %iea' % (self.R_elmts_len, self.Z_elmts_len, self.RZ_elmts_len))
        
        # 2D array R (angstrom -> m)
        self.RZ_R = np.zeros([self.R_nodes_len, self.Z_nodes_len])          # R nodes, Z nodes
        for each_z_node in range(self.Z_nodes_len):
            self.RZ_R[:,each_z_node] = copy.copy( self.R_nodes )            # stacking in Z direction
        self.RZ_R *= 1e-10                                                  # angstrom -> m
        self.RZ_dR = self.RZ_R[1:,:] - self.RZ_R[:-1,:]                     # R elements, Z nodes

        # 2D array Z (angstrom -> m)
        self.RZ_Z = np.zeros([self.R_nodes_len, self.Z_nodes_len])          # R nodes, Z nodes
        for each_r_node in range(self.R_nodes_len):
            self.RZ_Z[each_r_node,:] = copy.copy( self.Z_nodes )            # stacking in Z direction
        self.RZ_Z *= 1e-10                                                  # angstrom -> m
        self.RZ_dZ = self.RZ_Z[:,1:] - self.RZ_Z[:,:-1]                     # R elements, Z nodes

        # 2D array (electric perimittivity)
        self.RZ_EP = np.zeros([self.R_elmts_len, self.Z_elmts_len])                     # R elements, Z elements
        for each_z_elmt in range(self.Z_elmts_len):
            self.RZ_EP[:,each_z_elmt] = copy.copy( self.Z_MAT_ep[each_z_elmt] )         # stacking in Z direction

        # 2D array (material number)
        self.RZ_MATno = np.zeros([self.R_elmts_len, self.Z_elmts_len])                  # R elements, Z elements
        for each_z_elmt in range(self.Z_elmts_len):
            self.RZ_MATno[:,each_z_elmt] = copy.copy( self.Z_MAT_no[each_z_elmt] )      # stacking in Z direction

        # 2D array (conduction band diagram)
        self.CB_ref    = np.zeros([self.R_elmts_len, self.Z_elmts_len])                 # R elements, Z elements
        self.CB_offset = np.zeros([self.R_elmts_len, self.Z_elmts_len])                 # R elements, Z elements
        #
        for mat_name in self.MAT.keys():
            # mat info
            mat_no    = self.MAT[mat_name]['mat_no']
            cb_offset = self.MAT[mat_name]['cb']
            #
            if (mat_no == 11) or (mat_no == 10):
                pass
            else:
                self.CB_offset += np.where( self.RZ_MATno == mat_no, cb_offset, 0.0 )

        # CPU time
        end = time.time()

        # CPU time
        return end-start
    

    # ===== setting metal-insulator-semiconductor region =====
    def set_unit_cell_RZ_mis_region(self):
        # CPU time
        start = time.time()
        
        # making dictionary
        self.RZ_MIS = {}                    # list of (r, z) tuple 
        self.RZ_MIS_index_min_max = {}      # index min, max

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
                self.RZ_MIS_index_min_max[each_mat_mis] = {}
                
            # sweep mat_no (integer) list (second key)
            for each_mat_no in self.MIS_MAT_no[each_mat_mis]:
                # check each mat_no array
                if each_mat_no not in self.RZ_MIS[each_mat_mis].keys():
                    self.RZ_MIS[each_mat_mis][each_mat_no] = set()
                    self.RZ_MIS_index_min_max[each_mat_mis][each_mat_no] = {}
                    self.RZ_MIS_index_min_max[each_mat_mis][each_mat_no]['r'] = []
                    self.RZ_MIS_index_min_max[each_mat_mis][each_mat_no]['z'] = []
                    
                # get specific R, Z coordinates array in self.RZ_MATno array (elements) having the same each mat_no
                r_index_array, z_index_array = np.where( self.RZ_MATno == each_mat_no )

                # calculating R, Z index min. max. (RZ_MIS_index_min_max)
                r_index_array_min, r_index_array_max = np.min( r_index_array ), np.max( r_index_array )
                z_index_array_min, z_index_array_max = np.min( z_index_array ), np.max( z_index_array )
                self.RZ_MIS_index_min_max[each_mat_mis][each_mat_no]['r'] = [r_index_array_min, r_index_array_max]
                self.RZ_MIS_index_min_max[each_mat_mis][each_mat_no]['z'] = [z_index_array_min, z_index_array_max]
                
                # mapping: 1 element -> 4 nodes (2D structure)
                for each_point in range(len(r_index_array)):
                    # nodes
                    self.RZ_MIS[each_mat_mis][each_mat_no].add( ( r_index_array[each_point]+0, z_index_array[each_point]+0 ) )
                    self.RZ_MIS[each_mat_mis][each_mat_no].add( ( r_index_array[each_point]+0, z_index_array[each_point]+1 ) )
                    self.RZ_MIS[each_mat_mis][each_mat_no].add( ( r_index_array[each_point]+1, z_index_array[each_point]+0 ) )
                    self.RZ_MIS[each_mat_mis][each_mat_no].add( ( r_index_array[each_point]+1, z_index_array[each_point]+1 ) )

        # set difference: I - I (dielectrics, excluding double counting points)
        for tg_mat_no in self.MIS_MAT_no['I']:
            for diff_mat_no in self.MIS_MAT_no['I']:
                if tg_mat_no != diff_mat_no:
                    intersection_points = self.RZ_MIS['I'][tg_mat_no].intersection(self.RZ_MIS['I'][diff_mat_no])
                    self.RZ_MIS['I'][tg_mat_no] = self.RZ_MIS['I'][tg_mat_no].difference(self.RZ_MIS['I'][diff_mat_no])
                    # debugging
                    if True:
                        if len(intersection_points) != 0:
                            print('I (%i) intersect I (%i) = %iea  >> I (%i) net nodes = %iea' % \
                                  (tg_mat_no, diff_mat_no, len(intersection_points), tg_mat_no, len(self.RZ_MIS['I'][tg_mat_no])))
        
        # set difference: I - M (electrodes, dirichlet BC)
        for tg_mat_no in self.MIS_MAT_no['I']:
            for diff_mat_no in self.MIS_MAT_no['M']:
                intersection_points = self.RZ_MIS['I'][tg_mat_no].intersection(self.RZ_MIS['M'][diff_mat_no])
                self.RZ_MIS['I'][tg_mat_no] = self.RZ_MIS['I'][tg_mat_no].difference(self.RZ_MIS['M'][diff_mat_no])
                # debugging
                if True:
                    if len(intersection_points) != 0:
                        print('I (%i) intersect M (%i) = %iea  >> I (%i) net nodes = %iea' % \
                              (tg_mat_no, diff_mat_no, len(intersection_points), tg_mat_no, len(self.RZ_MIS['I'][tg_mat_no])))
                
        # set difference: I - S (semiconductors, Scharfetter-Gummel scheme, continuity equations)
        for tg_mat_no in self.MIS_MAT_no['I']:
            for diff_mat_no in self.MIS_MAT_no['S']:
                intersection_points = self.RZ_MIS['I'][tg_mat_no].intersection(self.RZ_MIS['S'][diff_mat_no])
                self.RZ_MIS['I'][tg_mat_no] = self.RZ_MIS['I'][tg_mat_no].difference(self.RZ_MIS['S'][diff_mat_no])
                # debugging
                if True:
                    if len(intersection_points) != 0:
                        print('I (%i) intersect S (%i) = %iea  >> I (%i) net nodes = %iea' % \
                              (tg_mat_no, diff_mat_no, len(intersection_points), tg_mat_no, len(self.RZ_MIS['I'][tg_mat_no])))
        
        # CPU time
        end = time.time()

        # CPU time
        return end-start
    

    # ===== adding ohmic contact =====
    def add_ohmic_contact(self, before_info, after_info):
        # CPU time
        start = time.time()
        
        # before info
        before_mat_mis = list(before_info.keys())[0]
        before_mat_no  = before_info[before_mat_mis]['mat_no']
        before_z_coord = before_info[before_mat_mis]['z_coord']
        if before_z_coord == -1:
            before_z_coord = self.Z_nodes_len-1

        # after info
        after_mat_mis = list(after_info.keys())[0]
        after_mat_no  = after_info[after_mat_mis]['mat_no']

        # check before nodes
        add_nodes = set()
        for each_r_node, each_z_node in list(self.RZ_MIS[before_mat_mis][before_mat_no]):
            if each_z_node == before_z_coord:
                add_nodes.add( (each_r_node, each_z_node) )         # tuple, point

        # debugging
        if True:
            print('%s (%i) to %s (%i) conversion nodes = %iea at Z = %i' % \
                  (before_mat_mis, before_mat_no, after_mat_mis, after_mat_no, len(add_nodes), before_z_coord) )

        # ohmic contact -> metal
        self.RZ_MIS[before_mat_mis][before_mat_no] = self.RZ_MIS[before_mat_mis][before_mat_no].difference(add_nodes)
        self.RZ_MIS[after_mat_mis][after_mat_no] = set()
        self.RZ_MIS[after_mat_mis][after_mat_no] = self.RZ_MIS[after_mat_mis][after_mat_no].union(add_nodes)
        
        # CPU time
        end = time.time()

        # CPU time
        return end-start
    

    # ===== setting semiconductor parameters =====
    def set_semiconductor_parameters(self, op_temperature, tg_region, bl_mat_no, sl_mat_no, doping, ct_doping):
        # CPU time
        start = time.time()
        
        # operating temperature (initialization)
        self.TEMP = op_temperature + 273.15             # celsius -> kelvin
        self.Vtm = self.kb * self.TEMP / self.q

        # semicondutor region (initialization)
        tg_mat_mis = list(tg_region.keys())[0]
        tg_mat_no  = tg_region[tg_mat_mis]['mat_no']
        tg_points  = list(self.RZ_MIS[tg_mat_mis][tg_mat_no])               # semiconductor (points)
        
        bl_points  = list(self.RZ_MIS['M'][bl_mat_no])                      # BL ohmic contact (points)
        sl_points  = list(self.RZ_MIS['M'][sl_mat_no])                      # SL ohmic contact (points)
        bl_sl_points = bl_points + sl_points                                # BL + SL ohmic contact (points)

        # debugging
        if True:
            print('Operating temperature: %.2f Kelvin (%.1f Celsius), thermal voltage = %.3f eV' % \
                  (self.TEMP, op_temperature, self.Vtm) )
            print('S (%i) channel %iea, M (%i) BL %iea, M (%i) SL %iea' % \
                  (tg_mat_no, len(tg_points), bl_mat_no, len(bl_points), sl_mat_no, len(sl_points)) )

        # semicondutor region (flag)
        self.CH_FLAG = np.zeros([self.R_nodes_len, self.Z_nodes_len])           # 2D nodes
        self.CH_FLAG_serial = np.zeros(self.R_nodes_len*self.Z_nodes_len)       # 1D serialization
        for each_tg_points in tg_points:                                        # semiconductor region only
            each_r, each_z = each_tg_points
            index_r_z = self.R_nodes_len * (each_z+0) + (each_r+0)
            self.CH_FLAG[each_r, each_z] = 1.0
            self.CH_FLAG_serial[index_r_z] = 1.0

        # doping profile (initialization)
        self.DP = 1e-9 * np.ones(self.RZ_nodes_len)                             # 1D array
        dopant_type, dopant_density = doping[0], doping[1]                      # 'n' or 'p', [m]^-3
        
        for each_point in (tg_points + bl_sl_points):                           # semiconductor + ohmic contacts
            r_node, z_node = each_point
            index_r_z = self.R_nodes_len * (z_node+0) + (r_node+0)              # 1D array index
            if dopant_type =='n':
                self.DP[index_r_z] = +dopant_density                            # w/ ionized polarity
            elif dopant_type =='p':
                self.DP[index_r_z] = -dopant_density                            # w/ ionized polarity
            else:
                print('set_semiconductor_parameters() > invalid dopant type')

        # ohmic contact doping profile (initialization)
        z_region_names = list(self.Z_stack.keys())
        
        bl_doping_const_thk = self.Z_stack[z_region_names[0]]['thk']            # BL constant doping
        bl_doping_const_dz = self.Z_stack[z_region_names[0]]['dz']        
        bl_doping_grad_thk = self.Z_stack[z_region_names[1]]['thk']             # BL gradient doping
        bl_doping_grad_dz = self.Z_stack[z_region_names[1]]['dz']
        bl_doping_const_length = int(bl_doping_const_thk/bl_doping_const_dz)
        bl_doping_grad_length = int(bl_doping_const_length*0.5)
        
        sl_doping_const_thk = self.Z_stack[z_region_names[-1]]['thk']           # SL constant doping
        sl_doping_const_dz = self.Z_stack[z_region_names[-1]]['dz']
        sl_doping_grad_thk = self.Z_stack[z_region_names[-2]]['thk']            # SL gradient doping
        sl_doping_grad_dz = self.Z_stack[z_region_names[-2]]['dz']
        sl_doping_const_length = int(sl_doping_const_thk/sl_doping_const_dz)
        sl_doping_grad_length = int(sl_doping_const_length*0.5)
            
        cont_length = int( bl_doping_const_length * 1.5 )                       # constant doping length
        grad_length = int( bl_doping_grad_length * 1.5 )                        # gradient doping length
        
        ct_dopant_type, ct_dopant_density = ct_doping[0], ct_doping[1]          # 'n' or 'p', [m]^-3
        ct_dopant_density_grad = np.logspace( np.log10(ct_dopant_density[0]), \
                                              np.log10(ct_dopant_density[1]), \
                                              grad_length)
        
        # making BL, SL doping profile
        for each_point in (bl_sl_points):
            r_node, z_node = each_point
            
            # dopant density (constant region)
            for z_node_add in range(cont_length):
                
                # BL contact
                if (z_node == 0):
                    index_r_z = self.R_nodes_len * (z_node+z_node_add) + (r_node+0)     # 1D array index
                    if ct_dopant_type =='n':
                        self.DP[index_r_z]   = +ct_dopant_density[0]                    # w/ ionized polarity
                    elif ct_dopant_type =='p':
                        self.DP[index_r_z]   = -ct_dopant_density[0]                    # w/ ionized polarity
                    else:
                        print('set_semiconductor_parameters() > invalid dopant type, contact')
                        
                # SL contact
                if (z_node == (self.Z_nodes_len-1)):
                    index_r_z = self.R_nodes_len * (z_node-z_node_add) + (r_node+0)     # 1D array index
                    if ct_dopant_type =='n':
                        self.DP[index_r_z]   = +ct_dopant_density[0]                    # w/ ionized polarity
                    elif ct_dopant_type =='p':
                        self.DP[index_r_z]   = -ct_dopant_density[0]                    # w/ ionized polarity
                    else:
                        print('set_semiconductor_parameters() > invalid dopant type, contact')
                        
            # dopant density gradiant region
            for z_node_add in range(grad_length):
                
                # BL contact
                if (z_node == 0):
                    index_r_z = self.R_nodes_len * (z_node+cont_length+z_node_add) + (r_node+0)     # 1D array index
                    if ct_dopant_type =='n':
                        self.DP[index_r_z]   = +ct_dopant_density_grad[z_node_add]                  # w/ ionized polarity
                    elif ct_dopant_type =='p':
                        self.DP[index_r_z]   = -ct_dopant_density_grad[z_node_add]                  # w/ ionized polarity
                    else:
                        print('set_semiconductor_parameters() > invalid dopant type, contact')
                        
                # SL contact
                if (z_node == (self.Z_nodes_len-1)):
                    index_r_z = self.R_nodes_len * (z_node-cont_length-z_node_add) + (r_node+0)     # 1D array index
                    if ct_dopant_type =='n':
                        self.DP[index_r_z]   = +ct_dopant_density_grad[z_node_add]                  # w/ ionized polarity
                    elif ct_dopant_type =='p':
                        self.DP[index_r_z]   = -ct_dopant_density_grad[z_node_add]                  # w/ ionized polarity
                    else:
                        print('set_semiconductor_parameters() > invalid dopant type, contact')

        # 2D visualization (materials, doping profile)
        self.DP2 = self.DP.reshape(self.Z_nodes_len, self.R_nodes_len).T                # 2D visualization
        self.RZ_MATno2 = np.where(self.RZ_MATno>=100, np.max(self.DP2), 1.0)            # 2D visualization
        self.RZ_MATno3 = np.where(self.RZ_MATno>=100, 0.0, self.RZ_MATno % 13)          # 2D visualization

        z = range(self.Z_nodes_len)             # for contour map
        r = range(self.R_nodes_len)             # for contour map
        Z, R = np.meshgrid(z, r)                # for contour map

        fig, ax = plt.subplots(2, 1, figsize=(15,7))
   
        ax0 = ax[0].imshow(self.RZ_MATno3, origin='lower', cmap='gray')
        ax[0].set_title('vRB_SG_GI_w991_on_580_rev03_20260412.py 13 w/ RZ nodes = [R %iea, Z %iea], total nodes = %iea' % \
                        (self.R_nodes_len, self.Z_nodes_len, self.RZ_nodes_len))
        plt.colorbar(ax0)
        
        ax1 = ax[1].imshow((self.RZ_MATno2 + self.DP2[:-1,:-1]), origin='lower', cmap='coolwarm')
        ax[1].contour(Z, R, self.DP2, colors='k', linewidths=0.01, levels=np.logspace(-1.0, 27.0, 29*2))
        ax[1].set_title('dopant density w/ electrodes [m^-3]')
        plt.colorbar(ax1)
        
        plt.savefig('materials_doping_profile.pdf')
        plt.close()

        # intrinsic carrier density (initialization)
        self.N_INT = 1e-9 * np.ones(self.RZ_nodes_len)                              # 1D array
        
        for each_point in (tg_points + bl_sl_points):
            r_node, z_node = each_point
            index_r_z = self.R_nodes_len * (z_node+0) + (r_node+0)          
            #
            self.N_INT[index_r_z] = self.MAT['SI']['n_int']                         # 1D array

        # free carrier density (initialization)
        self.n1  = ( np.sqrt( self.DP**2 + 4.0*self.N_INT**2 ) + self.DP ) / 2      # 1D array
        self.p1  = ( np.sqrt( self.DP**2 + 4.0*self.N_INT**2 ) - self.DP ) / 2      # 1D array
        
        self.n2 = self.n1.reshape(self.Z_nodes_len, self.R_nodes_len).T             # 2D array
        self.p2 = self.p1.reshape(self.Z_nodes_len, self.R_nodes_len).T             # 2D array

        # built-in potential (initialization)
        self.Vbi = self.Vtm * np.log( ( self.DP + \
                                        np.sqrt( self.DP**2 + 4.0*self.N_INT**2 ) + 1.0 ) / ( 2.0*self.N_INT + 1.0 ) )      # 1D array

        self.Vbi2 = self.Vbi.reshape(self.Z_nodes_len, self.R_nodes_len).T          # 2D array

        # coefficient of continuity equation matrix (initialization)
        self.CM = {}
        
        # STEP1: check neighbor points (semiconductor, excluding BL & SL contacts)
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
                R_r_z     = self.RZ_R[each_r_node+0, each_z_node+0]
                dR_rm1_z  = self.RZ_R[each_r_node+0, each_z_node+0] - self.RZ_R[each_r_node-1, each_z_node+0]
                self.CM[each_point]['rm1_z']['geometry'] = ( R_r_z - dR_rm1_z/2.0 ) / R_r_z                         # divergence
                self.CM[each_point]['rm1_z']['dR'] = dR_rm1_z                                                       # electric field
                self.CM[each_point]['rm1_z']['dR2'] = dR_rm1_z/2.0                                                  # divergence

            # check r+1, z (R direction)
            if (each_r_node+1, each_z_node) in tg_points:
                self.CM[each_point]['rp1_z'] = {}
                self.CM[each_point]['rp1_z']['index'] = self.R_nodes_len * (each_z_node+0) + (each_r_node+1)
                self.CM[each_point]['rp1_z']['mu_n'] = self.MAT['SI']['mu_n']
                self.CM[each_point]['rp1_z']['mu_p'] = self.MAT['SI']['mu_p']
                R_r_z     = self.RZ_R[each_r_node+0, each_z_node+0]
                dR_rp1_z  = self.RZ_R[each_r_node+1, each_z_node+0] - self.RZ_R[each_r_node+0, each_z_node+0]
                self.CM[each_point]['rp1_z']['geometry'] = ( R_r_z + dR_rp1_z/2.0 ) / R_r_z
                self.CM[each_point]['rp1_z']['dR'] = dR_rp1_z
                self.CM[each_point]['rp1_z']['dR2'] = dR_rp1_z/2.0

            # check r, z-1 (Z direction)
            if (each_r_node, each_z_node-1) in (tg_points + bl_sl_points):
                self.CM[each_point]['r_zm1'] = {}
                self.CM[each_point]['r_zm1']['index'] = self.R_nodes_len * (each_z_node-1) + (each_r_node+0)
                self.CM[each_point]['r_zm1']['mu_n'] = self.MAT['SI']['mu_n']
                self.CM[each_point]['r_zm1']['mu_p'] = self.MAT['SI']['mu_p']
                Z_r_z     = self.RZ_Z[each_r_node+0, each_z_node+0]
                dZ_r_zm1  = self.RZ_Z[each_r_node+0, each_z_node+0] - self.RZ_Z[each_r_node+0, each_z_node-1]
                self.CM[each_point]['r_zm1']['geometry'] = 1.0
                self.CM[each_point]['r_zm1']['dZ'] = dZ_r_zm1
                self.CM[each_point]['r_zm1']['dZ2'] = dZ_r_zm1/2.0
                
            # check r, z+1 (Z direction)
            if (each_r_node, each_z_node+1) in (tg_points + bl_sl_points):
                self.CM[each_point]['r_zp1'] = {}
                self.CM[each_point]['r_zp1']['index'] = self.R_nodes_len * (each_z_node+1) + (each_r_node+0)
                self.CM[each_point]['r_zp1']['mu_n'] = self.MAT['SI']['mu_n']
                self.CM[each_point]['r_zp1']['mu_p'] = self.MAT['SI']['mu_p']
                Z_r_z     = self.RZ_Z[each_r_node+0, each_z_node+0]
                dZ_r_zp1  = self.RZ_Z[each_r_node+0, each_z_node+1] - self.RZ_Z[each_r_node+0, each_z_node+0]
                self.CM[each_point]['r_zp1']['geometry'] = 1.0
                self.CM[each_point]['r_zp1']['dZ'] = dZ_r_zp1
                self.CM[each_point]['r_zp1']['dZ2'] = dZ_r_zp1/2.0
  
        # STEP2: updating neighbor points (semiconductor, excluding BL & SL contacts)
        for each_point in tg_points:
            
            # selected point in semiconductor region
            each_r_node, each_z_node = each_point
            each_point_neighbor = list(self.CM[each_point].keys())
            
            # updating dR2 (R direction)
            if ('rm1_z' in each_point_neighbor) and ('rp1_z' in each_point_neighbor):
                new_dR2 = self.CM[each_point]['rm1_z']['dR2'] + self.CM[each_point]['rp1_z']['dR2']
                self.CM[each_point]['rm1_z']['dR2'] = new_dR2
                self.CM[each_point]['rp1_z']['dR2'] = new_dR2
            elif ('rm1_z' in each_point_neighbor):
                new_dR2 = self.CM[each_point]['rm1_z']['dR2'] * 2.0
                self.CM[each_point]['rm1_z']['dR2'] = new_dR2
            elif ('rp1_z' in each_point_neighbor):
                new_dR2 = self.CM[each_point]['rp1_z']['dR2'] * 2.0
                self.CM[each_point]['rp1_z']['dR2'] = new_dR2
                
            # updating dZ2 (Z direction)
            if ('r_zm1' in each_point_neighbor) and ('r_zp1' in each_point_neighbor):
                new_dZ2 = self.CM[each_point]['r_zm1']['dZ2'] + self.CM[each_point]['r_zp1']['dZ2']
                self.CM[each_point]['r_zm1']['dZ2'] = new_dZ2
                self.CM[each_point]['r_zp1']['dZ2'] = new_dZ2
            elif ('r_zm1' in each_point_neighbor):
                new_dZ2 = self.CM[each_point]['r_zm1']['dZ2'] * 2.0
                self.CM[each_point]['r_zm1']['dZ2'] = new_dZ2
            elif ('r_zp1' in each_point_neighbor):
                new_dZ2 = self.CM[each_point]['r_zp1']['dZ2'] * 2.0
                self.CM[each_point]['r_zp1']['dZ2'] = new_dZ2
                
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
            if (each_r_node, each_z_node-1) in (tg_points + bl_sl_points):
                self.CM[each_point]['r_zm1']['n_CM_coeff']  = self.Vtm * self.CM[each_point]['r_zm1']['mu_n'] * 1.0
                self.CM[each_point]['r_zm1']['n_CM_coeff'] /= (self.CM[each_point]['r_zm1']['dZ'] * self.CM[each_point]['r_zm1']['dZ2'])
                self.CM[each_point]['r_zm1']['p_CM_coeff']  = self.Vtm * self.CM[each_point]['r_zm1']['mu_p']
                self.CM[each_point]['r_zm1']['p_CM_coeff'] /= (self.CM[each_point]['r_zm1']['dZ'] * self.CM[each_point]['r_zm1']['dZ2'])

            # check r, z+1 (Z direction)
            if (each_r_node, each_z_node+1) in (tg_points + bl_sl_points):
                self.CM[each_point]['r_zp1']['n_CM_coeff']  = self.Vtm * self.CM[each_point]['r_zp1']['mu_n'] * 1.0
                self.CM[each_point]['r_zp1']['n_CM_coeff'] /= (self.CM[each_point]['r_zp1']['dZ'] * self.CM[each_point]['r_zp1']['dZ2'])
                self.CM[each_point]['r_zp1']['p_CM_coeff']  = self.Vtm * self.CM[each_point]['r_zp1']['mu_p']
                self.CM[each_point]['r_zp1']['p_CM_coeff'] /= (self.CM[each_point]['r_zp1']['dZ'] * self.CM[each_point]['r_zp1']['dZ2'])

        # CPU time
        end = time.time()

        # CPU time
        return end-start


    # ===== making poisson matrix  =====
    def make_poisson_matrix(self):
        # CPU time
        start = time.time()
        
        # STEP0: making sparse matrix
        PM_shape = (self.RZ_nodes_len, self.RZ_nodes_len)
        
        self.PM = sc.sparse.dok_matrix( PM_shape )                  # sparse matrix (MIS model)
        self.PM_mim = sc.sparse.dok_matrix( PM_shape )              # sparse matrix (MIM model)

        # STEP1: dirichlet boundary conditions (electrodes)
        for each_mat_type in ['M', 'S']:
            for each_mat_no in self.RZ_MIS[each_mat_type].keys():
                for each_r, each_z in self.RZ_MIS[each_mat_type][each_mat_no]:
                    
                    # 1D serialization index
                    index_r_z = self.R_nodes_len * (each_z+0) + (each_r+0)

                    # for MIS model
                    if each_mat_type in ['M']:
                        # if not assigned
                        if self.PM[index_r_z, index_r_z] == 0.0:
                            # sparse matrix (dirichlet conditions)
                            self.PM[index_r_z, index_r_z] += 1.0
                    
                    # for MIM model: 'S' -> 'M'
                    if each_mat_type in ['M', 'S']:
                        # if not assigned
                        if self.PM_mim[index_r_z, index_r_z] == 0.0:
                            # sparse matrix (dirichlet conditions)
                            self.PM_mim[index_r_z, index_r_z] += 1.0

        # STEP2: neumann boundary conditions (Z boundaries)
        for each_z in [0, self.Z_nodes_len-1]:
            for each_r in range(self.R_nodes_len):
                # 1D serialization index
                index_r_z   = self.R_nodes_len * (each_z+0) + (each_r+0)
                index_rm1_z = self.R_nodes_len * (each_z+0) + (each_r-1)
                index_rp1_z = self.R_nodes_len * (each_z+0) + (each_r+1)
                index_r_zm1 = self.R_nodes_len * (each_z-1) + (each_r+0)
                index_r_zp1 = self.R_nodes_len * (each_z+1) + (each_r+0)
                
                # if not assigned (for MIS model)
                if self.PM[index_r_z, index_r_z] == 0.0:
                    
                    # Z = 0 boundary
                    if (each_z == 0) and (each_r == 0):
                        # sparse matrix (neumann conditions)
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zp1] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rp1_z] += -1.0
                        
                    elif (each_z == 0) and (each_r == (self.R_nodes_len-1)):
                        # sparse matrix (neumann conditions)
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zp1] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rm1_z] += -1.0
                        
                    elif (each_z == 0):
                        # sparse matrix (neumann conditions)
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zp1] += -1.0
                        
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
                        
                        # electric permittivity z-1 (invalid) -> z+0
                        ep_z_avg_rm1 = (self.RZ_EP[each_r-1,each_z+0]+self.RZ_EP[each_r-1,each_z+0])/2.0
                        ep_z_avg_rp1 = (self.RZ_EP[each_r+0,each_z+0]+self.RZ_EP[each_r+0,each_z+0])/2.0
                        
                        # elements
                        pm_rm1_z = geometry_effect_rm1_z * ep_z_avg_rm1
                        pm_rp1_z = geometry_effect_rp1_z * ep_z_avg_rp1
                        
                        # sparse matrix (poisson equation)
                        self.PM[index_r_z, index_r_z  ] += +pm_rm1_z + pm_rp1_z
                        self.PM[index_r_z, index_rm1_z] += -pm_rm1_z
                        self.PM[index_r_z, index_rp1_z] += -pm_rp1_z
                        
                    # Z = -1 boundary
                    if (each_z == (self.Z_nodes_len-1)) and (each_r == 0):
                        # sparse matrix (neumann conditions)
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zm1] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rp1_z] += -1.0
                        
                    elif (each_z == (self.Z_nodes_len-1)) and (each_r == (self.R_nodes_len-1)):
                        # sparse matrix (neumann conditions)
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zm1] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rm1_z] += -1.0
                        
                    elif (each_z == (self.Z_nodes_len-1)):
                        # sparse matrix (neumann conditions)
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zm1] += -1.0
                        
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
                        # electric permittivity: z+0 (invalid) -> z-1
                        ep_z_avg_rm1 = (self.RZ_EP[each_r-1,each_z-1]+self.RZ_EP[each_r-1,each_z-1])/2.0
                        ep_z_avg_rp1 = (self.RZ_EP[each_r+0,each_z-1]+self.RZ_EP[each_r+0,each_z-1])/2.0
                        # elements
                        pm_rm1_z = geometry_effect_rm1_z * ep_z_avg_rm1
                        pm_rp1_z = geometry_effect_rp1_z * ep_z_avg_rp1
                        #  sparse matrix (poisson equation)
                        self.PM[index_r_z, index_r_z  ] += +pm_rm1_z + pm_rp1_z
                        self.PM[index_r_z, index_rm1_z] += -pm_rm1_z
                        self.PM[index_r_z, index_rp1_z] += -pm_rp1_z

                # if not assigned (for MIM model)
                if self.PM_mim[index_r_z, index_r_z] == 0.0:
                    
                    # Z = 0 boundary
                    if (each_z == 0) and (each_r == 0):
                        # sparse matrix (neumann conditions)
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_r_zp1] += -1.0
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_rp1_z] += -1.0
                        
                    elif (each_z == 0) and (each_r == (self.R_nodes_len-1)):
                        # sparse matrix (neumann conditions)
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_r_zp1] += -1.0
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_rm1_z] += -1.0
                        
                    elif (each_z == 0):
                        # sparse matrix (neumann conditions)
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_r_zp1] += -1.0
                        
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
                        
                        # electric permittivity z-1 (invalid) -> z+0
                        ep_z_avg_rm1 = (self.RZ_EP[each_r-1,each_z+0]+self.RZ_EP[each_r-1,each_z+0])/2.0
                        ep_z_avg_rp1 = (self.RZ_EP[each_r+0,each_z+0]+self.RZ_EP[each_r+0,each_z+0])/2.0
                        
                        # elements
                        pm_rm1_z = geometry_effect_rm1_z * ep_z_avg_rm1
                        pm_rp1_z = geometry_effect_rp1_z * ep_z_avg_rp1
                        
                        # sparse matrix (poisson equation)
                        self.PM_mim[index_r_z, index_r_z  ] += +pm_rm1_z + pm_rp1_z
                        self.PM_mim[index_r_z, index_rm1_z] += -pm_rm1_z
                        self.PM_mim[index_r_z, index_rp1_z] += -pm_rp1_z
                        
                    # Z = -1 boundary
                    if (each_z == (self.Z_nodes_len-1)) and (each_r == 0):
                        # sparse matrix (neumann conditions)
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_r_zm1] += -1.0
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_rp1_z] += -1.0
                        
                    elif (each_z == (self.Z_nodes_len-1)) and (each_r == (self.R_nodes_len-1)):
                        # sparse matrix (neumann conditions)
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_r_zm1] += -1.0
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_rm1_z] += -1.0
                        
                    elif (each_z == (self.Z_nodes_len-1)):
                        # sparse matrix (neumann conditions)
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_r_zm1] += -1.0
                        
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
                        
                        # electric permittivity: z+0 (invalid) -> z-1
                        ep_z_avg_rm1 = (self.RZ_EP[each_r-1,each_z-1]+self.RZ_EP[each_r-1,each_z-1])/2.0
                        ep_z_avg_rp1 = (self.RZ_EP[each_r+0,each_z-1]+self.RZ_EP[each_r+0,each_z-1])/2.0
                        
                        # elements
                        pm_rm1_z = geometry_effect_rm1_z * ep_z_avg_rm1
                        pm_rp1_z = geometry_effect_rp1_z * ep_z_avg_rp1
                        
                        #  sparse matrix (poisson equation)
                        self.PM_mim[index_r_z, index_r_z  ] += +pm_rm1_z + pm_rp1_z
                        self.PM_mim[index_r_z, index_rm1_z] += -pm_rm1_z
                        self.PM_mim[index_r_z, index_rp1_z] += -pm_rp1_z

        # STEP3: neumann boundary conditions (R boundaries)
        for each_r in [0, self.R_nodes_len-1]:
            for each_z in range(self.Z_nodes_len):
                # 1D serialization index
                index_r_z   = self.R_nodes_len * (each_z+0) + (each_r+0)
                index_rm1_z = self.R_nodes_len * (each_z+0) + (each_r-1)
                index_rp1_z = self.R_nodes_len * (each_z+0) + (each_r+1)
                index_r_zm1 = self.R_nodes_len * (each_z-1) + (each_r+0)
                index_r_zp1 = self.R_nodes_len * (each_z+1) + (each_r+0)
                
                # if not assigned (MIS model)
                if self.PM[index_r_z, index_r_z] == 0.0:
                    
                    # R = 0 boundary
                    if (each_r == 0) and (each_z == 0):
                        # sparse matrix (neumann conditions)
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rp1_z] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zp1] += -1.0
                        
                    elif (each_r == 0) and (each_z == (self.Z_nodes_len-1)):
                        # sparse matrix (neumann conditions)
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rp1_z] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zm1] += -1.0
                        
                    elif (each_r == 0):
                        # sparse matrix (neumann conditions)
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rp1_z] += -1.0
                        # geometry factors in r direction
                        #geometry_effect_r_zm1  = 1.0
                        #geometry_effect_r_zp1  = 1.0
                        # 2nd derivatives
                        #geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+0]-self.RZ_Z[each_r+0,each_z-1])
                        #geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z-1])/2.0
                        #geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z+0])
                        #geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z-1])/2.0
                        # electric permittivity: r-1 (invalid) -> r+0
                        #ep_r_avg_zm1 = (self.RZ_EP[each_r+0,each_z-1]+self.RZ_EP[each_r+0,each_z-1])/2.0
                        #ep_r_avg_zp1 = (self.RZ_EP[each_r+0,each_z+0]+self.RZ_EP[each_r+0,each_z+0])/2.0
                        # elements
                        #pm_r_zm1 = geometry_effect_r_zm1 * ep_r_avg_zm1
                        #pm_r_zp1 = geometry_effect_r_zp1 * ep_r_avg_zp1
                        # sparse matrix (poission equation)
                        #self.PM[index_r_z, index_r_z  ] += +pm_r_zm1 + pm_r_zp1
                        #self.PM[index_r_z, index_r_zm1] += -pm_r_zm1
                        #self.PM[index_r_z, index_r_zp1] += -pm_r_zp1
                        
                    # R = -1 boundary
                    if (each_r == (self.R_nodes_len-1)) and (each_z == 0):
                        # sparse matrix (neumann conditions)
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rm1_z] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zp1] += -1.0
                        
                    elif (each_r == (self.R_nodes_len-1)) and (each_z == (self.Z_nodes_len-1)):
                        # sparse matrix (neumann conditions)
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rm1_z] += -1.0
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_r_zm1] += -1.0
                        
                    elif (each_r == (self.R_nodes_len-1)):
                        # sparse matrix (neumann conditions)
                        self.PM[index_r_z, index_r_z  ] += +1.0
                        self.PM[index_r_z, index_rm1_z] += -1.0
                        # geometry factors in r direction
                        geometry_effect_r_zm1  = 1.0
                        geometry_effect_r_zp1  = 1.0
                        # 2nd derivatives
                        geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+0]-self.RZ_Z[each_r+0,each_z-1])
                        geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z-1])/2.0
                        geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z+0])
                        geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z-1])/2.0
                        # electric permittivity: r+0 (invalid) -> r-1
                        ep_r_avg_zm1 = (self.RZ_EP[each_r-1,each_z-1]+self.RZ_EP[each_r-1,each_z-1])/2.0
                        ep_r_avg_zp1 = (self.RZ_EP[each_r-1,each_z+0]+self.RZ_EP[each_r-1,each_z+0])/2.0
                        # elements
                        pm_r_zm1 = geometry_effect_r_zm1 * ep_r_avg_zm1
                        pm_r_zp1 = geometry_effect_r_zp1 * ep_r_avg_zp1
                        #  sparse matrix (poission equation)
                        self.PM[index_r_z, index_r_z  ] += +pm_r_zm1 + pm_r_zp1
                        self.PM[index_r_z, index_r_zm1] += -pm_r_zm1
                        self.PM[index_r_z, index_r_zp1] += -pm_r_zp1

                # if not assigned (MIM model)
                if self.PM_mim[index_r_z, index_r_z] == 0.0:
                    
                    # R = 0 boundary
                    if (each_r == 0) and (each_z == 0):
                        # sparse matrix (neumann conditions)
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_rp1_z] += -1.0
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_r_zp1] += -1.0
                        
                    elif (each_r == 0) and (each_z == (self.Z_nodes_len-1)):
                        # sparse matrix (neumann conditions)
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_rp1_z] += -1.0
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_r_zm1] += -1.0
                        
                    elif (each_r == 0):
                        # sparse matrix (neumann conditions)
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_rp1_z] += -1.0
                        # geometry factors in r direction
                        #geometry_effect_r_zm1  = 1.0
                        #geometry_effect_r_zp1  = 1.0
                        # 2nd derivatives
                        #geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+0]-self.RZ_Z[each_r+0,each_z-1])
                        #geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z-1])/2.0
                        #geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z+0])
                        #geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z-1])/2.0
                        # electric permittivity: r-1 (invalid) -> r+0
                        #ep_r_avg_zm1 = (self.RZ_EP[each_r+0,each_z-1]+self.RZ_EP[each_r+0,each_z-1])/2.0
                        #ep_r_avg_zp1 = (self.RZ_EP[each_r+0,each_z+0]+self.RZ_EP[each_r+0,each_z+0])/2.0
                        # elements
                        #pm_r_zm1 = geometry_effect_r_zm1 * ep_r_avg_zm1
                        #pm_r_zp1 = geometry_effect_r_zp1 * ep_r_avg_zp1
                        # sparse matrix (poission equation)
                        #self.PM_mim[index_r_z, index_r_z  ] += +pm_r_zm1 + pm_r_zp1
                        #self.PM_mim[index_r_z, index_r_zm1] += -pm_r_zm1
                        #self.PM_mim[index_r_z, index_r_zp1] += -pm_r_zp1
                        
                    # R = -1 boundary
                    if (each_r == (self.R_nodes_len-1)) and (each_z == 0):
                        # sparse matrix (neumann conditions)
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_rm1_z] += -1.0
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_r_zp1] += -1.0
                        
                    elif (each_r == (self.R_nodes_len-1)) and (each_z == (self.Z_nodes_len-1)):
                        # sparse matrix (neumann conditions)
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_rm1_z] += -1.0
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_r_zm1] += -1.0
                        
                    elif (each_r == (self.R_nodes_len-1)):
                        # sparse matrix (neumann conditions)
                        self.PM_mim[index_r_z, index_r_z  ] += +1.0
                        self.PM_mim[index_r_z, index_rm1_z] += -1.0
                        # geometry factors in r direction
                        geometry_effect_r_zm1  = 1.0
                        geometry_effect_r_zp1  = 1.0
                        # 2nd derivatives
                        geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+0]-self.RZ_Z[each_r+0,each_z-1])
                        geometry_effect_r_zm1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z-1])/2.0
                        geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z+0])
                        geometry_effect_r_zp1 /= (self.RZ_Z[each_r+0,each_z+1]-self.RZ_Z[each_r+0,each_z-1])/2.0
                        # electric permittivity: r+0 (invalid) -> r-1
                        ep_r_avg_zm1 = (self.RZ_EP[each_r-1,each_z-1]+self.RZ_EP[each_r-1,each_z-1])/2.0
                        ep_r_avg_zp1 = (self.RZ_EP[each_r-1,each_z+0]+self.RZ_EP[each_r-1,each_z+0])/2.0
                        # elements
                        pm_r_zm1 = geometry_effect_r_zm1 * ep_r_avg_zm1
                        pm_r_zp1 = geometry_effect_r_zp1 * ep_r_avg_zp1
                        #  sparse matrix (poission equation)
                        self.PM_mim[index_r_z, index_r_z  ] += +pm_r_zm1 + pm_r_zp1
                        self.PM_mim[index_r_z, index_r_zm1] += -pm_r_zm1
                        self.PM_mim[index_r_z, index_r_zp1] += -pm_r_zp1

        # STEP4: inside boundary conditions
        for each_r in range(1, self.R_nodes_len-1):
            for each_z in range(1, self.Z_nodes_len-1):
                # 1D serialization index
                index_r_z   = self.R_nodes_len * (each_z+0) + (each_r+0)
                index_rm1_z = self.R_nodes_len * (each_z+0) + (each_r-1)
                index_rp1_z = self.R_nodes_len * (each_z+0) + (each_r+1)
                index_r_zm1 = self.R_nodes_len * (each_z-1) + (each_r+0)
                index_r_zp1 = self.R_nodes_len * (each_z+1) + (each_r+0)
                
                # if not assigned (MIS model)
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
                    
                    # sparse matrix (poisson equation)
                    self.PM[index_r_z, index_r_z  ] += +pm_rm1_z + pm_rp1_z + pm_r_zm1 + pm_r_zp1
                    self.PM[index_r_z, index_rm1_z] += -pm_rm1_z
                    self.PM[index_r_z, index_rp1_z] += -pm_rp1_z
                    self.PM[index_r_z, index_r_zm1] += -pm_r_zm1
                    self.PM[index_r_z, index_r_zp1] += -pm_r_zp1

            # if not assigned (MIM model)
                if self.PM_mim[index_r_z, index_r_z] == 0.0:
                    
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
                    
                    # sparse matrix (poisson equation)
                    self.PM_mim[index_r_z, index_r_z  ] += +pm_rm1_z + pm_rp1_z + pm_r_zm1 + pm_r_zp1
                    self.PM_mim[index_r_z, index_rm1_z] += -pm_rm1_z
                    self.PM_mim[index_r_z, index_rp1_z] += -pm_rp1_z
                    self.PM_mim[index_r_z, index_r_zm1] += -pm_r_zm1
                    self.PM_mim[index_r_z, index_r_zp1] += -pm_r_zp1
        
        # sparse matrix
        self.PMcsr = self.PM.tocsr()
        self.PM_mimcsr = self.PM_mim.tocsr()

        # external bias vector initialization
        self.EB = np.zeros(self.RZ_nodes_len)
        self.EB_mim = np.zeros(self.RZ_nodes_len)
        self.EB2 = self.EB.reshape(self.Z_nodes_len, self.R_nodes_len).T
        self.EB_mim2 = self.EB_mim.reshape(self.Z_nodes_len, self.R_nodes_len).T

        # fixed charge density vector initialization
        self.FC = np.zeros(self.RZ_nodes_len)
        self.FC_mim = np.zeros(self.RZ_nodes_len)
        self.FC2 = self.FC.reshape(self.Z_nodes_len, self.R_nodes_len).T
        self.FC_mim2 = self.FC_mim.reshape(self.Z_nodes_len, self.R_nodes_len).T

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
                self.N[index_r_z, index_r_z] += 1.0
                self.P[index_r_z, index_r_z] += 1.0

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
    def make_external_bias_vector(self, external_bias_conditions, workfunction, model_type):
        # CPU time
        start = time.time()

        # MIS model
        if model_type == 'MIS':
            
            # sweep material type
            for each_mat_mis in ['M']:
                for each_mat_no in self.RZ_MIS[each_mat_mis].keys():
                    
                    # check points
                    for each_r, each_z in self.RZ_MIS[each_mat_mis][each_mat_no]:
                        # 1D serialization index
                        index_r_z = self.R_nodes_len * (each_z+0) + (each_r+0)
                        
                        # external bias @metal contact
                        if (each_mat_no != 10001) and (each_mat_no != 10002):
                            self.EB[index_r_z]  = 4.05 + 1.12 / 2.0 - workfunction
                            self.EB[index_r_z] += external_bias_conditions[each_mat_no]
                        # external bias @BL, SL contact
                        else:
                            self.EB[index_r_z]  = -self.Vbi[index_r_z]
                            self.EB[index_r_z] += external_bias_conditions[each_mat_no]

            # visualization
            self.EB2 = self.EB.reshape(self.Z_nodes_len, self.R_nodes_len).T

        # MIM model: 'S' -> 'M'
        if model_type == 'MIM':
            
            # sweep material type
            for each_mat_mis in ['M', 'S']:
                for each_mat_no in self.RZ_MIS[each_mat_mis].keys():
                    
                    # check points
                    for each_r, each_z in self.RZ_MIS[each_mat_mis][each_mat_no]:
                        # 1D serialization index
                        index_r_z = self.R_nodes_len * (each_z+0) + (each_r+0)

                        # external bias @metal contact
                        if each_mat_mis == 'M':
                            self.EB_mim[index_r_z] = external_bias_conditions[each_mat_no]
                        # external bias @channel
                        if each_mat_mis == 'S':
                            self.EB_mim[index_r_z] = external_bias_conditions[each_mat_no]

            # visualization
            self.EB_mim2 = self.EB_mim.reshape(self.Z_nodes_len, self.R_nodes_len).T

        # CPU time
        end = time.time()

        # CPU time
        return end-start
    

    # ===== making fixed charge vector  =====
    def make_fixed_charge_vector(self, fixed_charge_density, model_type):
        # CPU time
        start = time.time()

        # MIS model
        if model_type == 'MIS':
            
            # sweep material type
            for each_mat_mis in ['I', 'S']:
                for each_mat_no in self.RZ_MIS[each_mat_mis].keys():
                    
                    # check material number
                    if each_mat_no in fixed_charge_density.keys():
                        # check points
                        for each_r, each_z in self.RZ_MIS[each_mat_mis][each_mat_no]:
                            # 1D serialization index
                            index_r_z = self.R_nodes_len * (each_z+0) + (each_r+0)
                            # fixed charge density 
                            self.FC[index_r_z] = fixed_charge_density[each_mat_no]

            # visualization
            self.FC2 = self.FC.reshape(self.Z_nodes_len, self.R_nodes_len).T

        # MIM model
        if model_type == 'MIM':
            
            # sweep material type
            for each_mat_mis in ['I']:
                for each_mat_no in self.RZ_MIS[each_mat_mis].keys():
                    
                    # check material number
                    if each_mat_no in fixed_charge_density.keys():
                        # check points
                        for each_r, each_z in self.RZ_MIS[each_mat_mis][each_mat_no]:
                            # 1D serialization index
                            index_r_z = self.R_nodes_len * (each_z+0) + (each_r+0)
                            # fixed charge density 
                            self.FC_mim[index_r_z] = fixed_charge_density[each_mat_no]

            # visualization
            self.FC_mim2 = self.FC_mim.reshape(self.Z_nodes_len, self.R_nodes_len).T

        # CPU time
        end = time.time()

        # CPU time
        return end-start
    

    # ===== solving poisson equation  =====
    def solve_poisson_equation(self, model_type):
        # CPU time
        start = time.time()

        # for MIS model
        if model_type == 'MIS':
            
            # sparse matrix solver for poisson equation
            self.V1 = sc.sparse.linalg.spsolve( self.PMcsr, self.EB + self.q*(self.FC + self.p1 - self.n1 + self.DP) )

            # 2D visualization
            self.V2 = self.V1.reshape(self.Z_nodes_len, self.R_nodes_len).T
            self.Er = ( self.V2[1:,:] - self.V2[:-1,:] ) / self.RZ_dR
            self.Ez = ( self.V2[:,1:] - self.V2[:,:-1] ) / self.RZ_dZ
            self.E  = np.sqrt( self.Er[:,:-1]**2 + self.Ez[:-1,:]**2 )

        # for MIM model
        if model_type == 'MIM':

            # sparse matrix solver for poisson equation
            self.V1_mim = sc.sparse.linalg.spsolve( self.PM_mimcsr, self.EB_mim + self.q*self.FC_mim )

            # 2D visualization
            self.V2_mim = self.V1_mim.reshape(self.Z_nodes_len, self.R_nodes_len).T
            self.Er_mim = ( self.V2_mim[1:,:] - self.V2_mim[:-1,:] ) / self.RZ_dR
            self.Ez_mim = ( self.V2_mim[:,1:] - self.V2_mim[:,:-1] ) / self.RZ_dZ
            self.E_mim  = np.sqrt( self.Er_mim[:,:-1]**2 + self.Ez_mim[:-1,:]**2 )

        # CPU time
        end = time.time()

        # CPU time
        return end-start
    

    # ===== calculate surface induced charge on channel (MIM model only)  =====
    def cal_channel_induced_charge(self, mat_no_ch, mat_no_tox):
        # charge profile, mat_no profile, Z direction profile
        Q_profile = []
        mat_no_profile = []
        Z_profile = []
        E_profile = []
        E_Q_profile = []
        
        # sweep Z
        for each_z in range(1, self.Z_elmts_len):
            # sweep R
            for each_r in range(1, self.R_elmts_len-1):
                
                # finding channel - TOX interface
                if (self.RZ_MATno[each_r-1, each_z] == mat_no_ch) and (self.RZ_MATno[each_r, each_z] == mat_no_tox):
                    
                    # geometry
                    r    = self.RZ_R[each_r, each_z]
                    z    = self.RZ_Z[each_r, each_z]
                    dz   = self.RZ_dZ[each_r, each_z]
                    area = 2.0 * np.pi * r * dz
                    
                    # electric displacement
                    ep_tox = self.RZ_EP[each_r, each_z]
                    E_tox  = self.E_mim[each_r, each_z]
                    D_tox  = ep_tox * E_tox
                    E_profile.append(E_tox)
                    E_Q_profile.append((D_tox/self.q)**1.5)
                    
                    # dQ, charge profile
                    dQ = D_tox * area
                    Q_profile.append(dQ)
                    
                    # mat_no profile
                    mat_no = self.RZ_MATno[self.R_elmts_len-1, each_z]
                    mat_no_profile.append(mat_no)
                    
                    # Z direction profile
                    Z_profile.append(z)
                    
        # total charge
        Q = np.sum(Q_profile)
        
        return [Q, Q_profile, mat_no_profile, Z_profile, E_profile, E_Q_profile]
    

    # ===== calculate tunneling probability (MIM model only)  =====
    def cal_tunneling_probability(self, mat_no_tox, meff):
        # WKB approximation profile, mat_no profile, Z direction profile
        WKB_profile = []
        WKB_profile2 = []
        WKB_length_profile = []
        WKB_length_profile2 = []
        mat_no_profile = []
        Z_profile = []
        
        # conduction band offset
        CB_offset = self.CB_offset - self.V2_mim[:-1,:-1]
        
        # sweep Z
        for each_z in range(1, self.Z_elmts_len):
            
            # Z direction position
            Z_profile.append( self.RZ_Z[0, each_z] )
            
            # mat_no profile
            mat_no = self.RZ_MATno[self.R_elmts_len-1, each_z]
            mat_no_profile.append(mat_no)

            # defining TOX region
            r_min, r_max = self.RZ_MIS_index_min_max['I'][mat_no_tox]['r']
            r_range = range(r_min, r_max+1)
            delta_r = self.RZ_R[r_max+1, each_z] - self.RZ_R[r_max, each_z]
            delta_CB_offset = CB_offset[r_max, each_z] - CB_offset[r_max-1, each_z]
            
            # finding interpolation function: conduction band offset
            RZ_R_r  = list( self.RZ_R[r_range, each_z] )
            RZ_R_r += [ RZ_R_r[-1] + RZ_R_r[-1] - RZ_R_r[-2] ]
            RZ_dR_r  = list( self.RZ_dR[r_range, each_z] )
            RZ_dR_r += [ RZ_R_r[-1] - RZ_R_r[-2] ]
            CB_offset_r  = list( CB_offset[r_range, each_z] )
            CB_offset_r += [ CB_offset_r[-1] + CB_offset_r[-1] - CB_offset_r[-2] ]
            CB_offset_r  = np.array( CB_offset_r )
            CB_offset_f = sc.interpolate.interp1d(CB_offset_r, RZ_R_r, kind='cubic')
            
            # finding interpolation function: WKB approximation
            WKB_approx_r = np.where( CB_offset_r > 0.0, \
                                    np.sqrt( 2.0 * (self.me * meff) * (CB_offset_r * self.q) ), \
                                    0.0 )
            WKB_approx_f = sc.interpolate.interp1d(RZ_R_r, WKB_approx_r, kind='cubic')
            
            # finding tunneling out position
            if CB_offset_r[-1] < 0.0:
                r_tunneling_out = CB_offset_f( 0.0 )
            else:
                r_tunneling_out = RZ_R_r[-1]

            # WKB approximation: case 2
            WKB_approx_tunneling_out, error = sc.integrate.quad(WKB_approx_f, RZ_R_r[0], r_tunneling_out)
            WKB_approx_tunneling_out = np.exp( -2.0 / self.hbar * WKB_approx_tunneling_out )
            WKB_profile2.append( WKB_approx_tunneling_out )
            WKB_length_profile2.append( r_tunneling_out - RZ_R_r[0] )
            
            # WKB approximation: case 1
            WKB_approx = 0.0
            WKB_length = 0
            
            # sweep R
            for each_r in range(1, self.R_elmts_len-1):
                # finding TOX
                if self.RZ_MATno[each_r, each_z] == mat_no_tox:
                    # decaying region
                    if CB_offset[each_r, each_z] > 0.0:
                        # distance
                        dr = ( self.RZ_R[each_r+1, each_z] - self.RZ_R[each_r, each_z] )
                        WKB_length += dr
                        # conduction band electron Tunneling only
                        WKB_approx += np.sqrt( 2.0 * (self.me * meff) * CB_offset[each_r, each_z] * self.q ) * dr
                        
            #
            WKB_approx = np.exp( -2.0 / self.hbar * WKB_approx)
            WKB_profile.append( WKB_approx )
            WKB_length_profile.append( WKB_length ) 
            
        #
        return [WKB_profile, WKB_profile2, WKB_length_profile, WKB_length_profile2, mat_no_profile, Z_profile]


    # ===== calculate thermal velocity (MIM model only)  =====
    def cal_thermal_velocity(self):
        # thermal energy
        thermal_energy = self.Vtm * self.q

        # thermal velocity
        thermal_velocity = np.sqrt( thermal_energy / self.me )

        # return
        return thermal_velocity


    # ===== charge trap nitride trap model (MIM model only)  =====
    def cal_ctn_trap_model_1d(self, dt, tox_meff, mat_no_ch, mat_no_tox, mat_no_ctn, ctn_peak_pos, cnt_ccs_array, ctn_density_array):
        # calculating induced charge profile
        Q, Q_profile, mat_no_profile, Z_profile, E_profile, E_Q_profile = self.cal_channel_induced_charge(mat_no_ch, mat_no_tox)
        
        # calculating tunneling probability
        WKB_profile, WKB_profile2, WKB_length_profile, WKB_length_profile2, mat_no_profile, Z_profile = self.cal_tunneling_probability(mat_no_tox, tox_meff)

        # calculating thermal velocity
        thermal_velocity = self.cal_thermal_velocity()
        
        # sweep Z
        for each_z in range(1, self.Z_elmts_len):
            # position
            r_ch_tox,  r_ch_tox_index  = 0.0, 0.0
            r_tox_ctn, r_tox_ctn_index = 0.0, 0.0
            r_ctn,     r_ctn_index     = [], []
            k_ctn = []

            # STEP0: [induced charge density * freq * dt], [tunneling probability]
            # E_Q_profile = (D_tox/self.q)**1.5
            induced_Q_density = E_Q_profile[each_z-1] * ( thermal_vel**2 / 70e-10 ) * dt
            tunneling_prob = WKB_profile2[each_z-1]
                
            # STEP1: sweep R
            for each_r in range(1, self.R_elmts_len-1):
                # geometry
                r  = self.RZ_R[each_r,  each_z]
                z  = self.RZ_Z[each_r,  each_z]
                dz = self.RZ_dZ[each_r, each_z]
                    
                # finding channel - TOX interface
                if (self.RZ_MATno[each_r-1, each_z] == mat_no_ch) and (self.RZ_MATno[each_r, each_z] == mat_no_tox):
                    r_ch_tox_index = each_r
                    r_ch_tox = r

                # finding TOX interface - CTN interface
                if (self.RZ_MATno[each_r, each_z] == mat_no_tox) and (self.RZ_MATno[each_r+1, each_z] == mat_no_ctn):
                    r_tox_ctn_index = each_r
                    r_tox_ctn = r

                # finding CTN region
                if (self.RZ_MATno[each_r, each_z] == mat_no_ctn):
                    r_ctn_index.append(each_r)
                    r_ctn.append(r)

            # STEP2: revising [induced charge density * freq * dt]
            induced_Q_density *= r_ch_tox / r_tox_ctn

            # STEP3: CTN material parameter interpolation
            x = [r_ctn[0], r_ctn[int(len(r_ctn_index)*ctn_peak_pos)], r_ctn[-1]]
            f_ctn_ccs = sc.interpolate.interp1d( x, cnt_ccs_array, kind='linear' )
            f_ctn_density = sc.interpolate.interp1d( x, ctn_density_array, kind='linear' )

            # STEP4: CTN ccs * CTN density (empty state)
            f_ctn_density_new = np.where( np.abs( self.FC_mim2[r_ctn_index, each_z] ) > f_ctn_density(r_ctn), \
                                          f_ctn_density(r_ctn), np.abs( self.FC_mim2[r_ctn_index, each_z] ) )
            ctn_ccs_density = f_ctn_ccs(r_ctn) * ( f_ctn_density(r_ctn) - f_ctn_density_new  )
            ctn_r_array_dr = r_ctn[1] - r_ctn[0]
            ctn_ccs_density_cum = np.cumsum( ctn_ccs_density ) * ctn_r_array_dr

            # STEP5: finding CTN trapped flux
            ctn_pass_flux  = r_tox_ctn * induced_Q_density * tunneling_prob * ( 1 - np.exp( -ctn_ccs_density_cum ) ) / np.array(r_ctn)
            ctn_trap_flux  = list( ctn_pass_flux[1:] - ctn_pass_flux[:-1] )
            ctn_trap_flux  = np.array( [ctn_trap_flux[0]/2] + ctn_trap_flux )

            # STEP6: updating Fixed charge density
            self.FC_mim2[r_ctn_index, each_z] += -ctn_trap_flux
            self.FC_mim2[r_ctn_index, each_z]  = np.where( np.abs( self.FC_mim2[r_ctn_index, each_z] ) > f_ctn_density(r_ctn), \
                                                           -f_ctn_density(r_ctn), self.FC_mim2[r_ctn_index, each_z] )

            # visualization
            if False:
                if each_z == int(self.R_elmts_len/2.0):
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    ax[0].plot(r_ctn, f_ctn_ccs(r_ctn), 'o-')
                    ax[0].grid(ls=':')
                    ax[1].plot(r_ctn, f_ctn_density(r_ctn), 'o-')
                    ax[1].grid(ls=':')
                    plt.show()
                    
            if True:
                if (each_z == 1):
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    ax[0].plot(r_ctn, ctn_trap_flux, 'o-')
                    ax[0].grid(ls=':')
                    ax[1].plot(r_ctn, f_ctn_density(r_ctn), 'o-')
                    ax[1].plot(r_ctn, np.abs(self.FC_mim2[r_ctn_index, each_z]), 'o-')
                    ax[1].grid(ls=':')
                    #print(each_z, induced_Q_density, tunneling_prob)
                    
                elif (each_z == int(self.Z_elmts_len/2.0)):
                    ax[0].plot(r_ctn, ctn_trap_flux, 'o-')
                    ax[0].grid(ls=':')
                    ax[1].plot(r_ctn, f_ctn_density(r_ctn), 'o-')
                    ax[1].plot(r_ctn, np.abs(self.FC_mim2[r_ctn_index, each_z]), 'o-')
                    ax[1].grid(ls=':')
                    #print(each_z, induced_Q_density, tunneling_prob)
                    plt.show()

                    # debugging
                    print(each_z, r_ch_tox, r_tox_ctn, induced_Q_density, tunneling_prob, ctn_trap_flux)
                        

    # ===== plot conduction band diagram  =====
    def plot_conduction_band_diagram(self, output_filename, model_type):

        # MIM model
        if model_type == 'MIM':
            
            # geometry
            R = self.RZ_R[60:-1,:-1]
            Z = self.RZ_Z[60:-1,:-1]
            # 
            CB_offset = self.CB_offset[60:,:] - self.V2_mim[60:-1,:-1]
            wl_bias = np.max( self.V2_mim )
            # visualization
            fig, ax = plt.subplots(1, 3)    # subplot_kw={"projection": "3d"}
            # visualization
            ax[0].imshow(self.V2_mim[60:-1,:-1], origin='lower')
            ax[0].set_axis_off()
            ax[0].grid(ls=':')
            ax[1].imshow(self.E_mim[60:-1,:-1], origin='lower')
            ax[1].set_axis_off()
            ax[1].grid(ls=':')
            #ax.plot_surface(R, Z, CB_offset, linewidth=5, vmin=0.0, cmap=cm.coolwarm)
            ax[2].contourf(Z.T, R.T, CB_offset.T, levels=[-wl_bias, 0.0], colors='k' )    # cmap=cm.coolwarm
            ax[2].set_axis_off()
            ax[2].grid(ls=':')
            plt.axis('equal')
            plt.savefig(output_filename)
            plt.show()
            
        # MIS model
        if model_type == 'MIS':
            CB_offset = self.CB_offset[:,:] - self.V2[:-1,:-1]


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
    def solve_continuity_equation(self, dt):
        # CPU time
        start = time.time()

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
        if True:
            # Scharffer Gummel scheme
            B_tol = 1e-10   
            self.Br_f = np.where( np.abs(self.dVr_f) > B_tol, self.dVr_f / ( np.exp(self.dVr_f) - 1.0 + 1e-12), 1.0)
            self.Br_b = np.where( np.abs(self.dVr_b) > B_tol, self.dVr_b / ( np.exp(self.dVr_b) - 1.0 + 1e-12), 1.0)
            self.Bz_f = np.where( np.abs(self.dVz_f) > B_tol, self.dVz_f / ( np.exp(self.dVz_f) - 1.0 + 1e-12), 1.0)
            self.Bz_b = np.where( np.abs(self.dVz_b) > B_tol, self.dVz_b / ( np.exp(self.dVz_b) - 1.0 + 1e-12), 1.0)

        if False:
            # Slotboom scheme
            self.Br_f = np.exp(-self.dVr_f/2.0)
            self.Br_b = np.exp(-self.dVr_b/2.0)
            self.Bz_f = np.exp(-self.dVz_f/2.0)
            self.Bz_b = np.exp(-self.dVz_b/2.0)
            
        # updating N, P matrix for continuity equation
        self.make_N_P_matrix(dt)
        
        # sparse matrix solver for continuity equation
        self.n1 = np.abs( sc.sparse.linalg.spsolve(self.Ncsr + self.dNcsr, self.n1 ) )
        self.p1 = np.abs( sc.sparse.linalg.spsolve(self.Pcsr + self.dPcsr, self.p1 ) )

        # 2D visualization 
        self.n2 = self.n1.reshape(self.Z_nodes_len, self.R_nodes_len).T
        self.p2 = self.p1.reshape(self.Z_nodes_len, self.R_nodes_len).T

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
            Jn_bl *= ( self.Bz_f[each_r_index, each_z_index+0] * self.n2[each_r_index, each_z_index+1] - \
                       self.Bz_b[each_r_index, each_z_index+0] * self.n2[each_r_index, each_z_index+0] )
            Jp_bl  = -self.q * self.MAT['SI']['mu_p'] * self.Vtm / self.RZ_dZ[each_r_index, each_z_index]
            Jp_bl *= ( self.Bz_b[each_r_index, each_z_index+0] * self.p2[each_r_index, each_z_index+1] - \
                       self.Bz_f[each_r_index, each_z_index+0] * self.p2[each_r_index, each_z_index+0] )
            
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
    

    # ===== save solutions (TXT file) =====
    def save_SG_scheme_solutions_in_txt(self, output_filename):
        # CPU time
        start = time.time()
        
        #
        fid_out = open(output_filename + '_sol.txt', 'w')
        
        #
        header    = 'R_index,Z_index,R,Z,EP,MATno,V,Er,Ez,E,FC,n,p,DP,Vbi' + '\n'
        data_type = 'Integer,Integer,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real' + '\n'
        fid_out.write(header)
        fid_out.write(data_type)
        
        #
        output_format = '%i,%i,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e' + '\n'

        for r_index in range(self.R_nodes_len):
            for z_index in range(self.Z_nodes_len):
                #
                r = self.RZ_R[r_index, z_index]
                z = self.RZ_Z[r_index, z_index]
                #
                try:
                    ep = self.RZ_EP[r_index, z_index]
                except IndexError:
                    ep = 1.0e50
                #
                try:
                    mat_no = self.RZ_MATno[r_index, z_index]
                except IndexError:
                    mat_no = 1.0e50
                #
                poisson_v  = self.V2[r_index, z_index]
                poisson_fc = self.FC2[r_index, z_index]
                #
                try:
                    poisson_er = self.Er[r_index, z_index]
                except IndexError:
                    poisson_er = 1.0e50
                #
                try:
                    poisson_ez = self.Ez[r_index, z_index]
                except IndexError:
                    poisson_ez = 1.0e50
                #
                try:
                    poisson_e  = self.E[r_index, z_index]
                except IndexError:
                    poisson_e  = 1.0e50
                #
                continuity_n = self.n2[r_index, z_index]
                continuity_p = self.p2[r_index, z_index]

                #
                doping   = self.DP2[r_index, z_index]
                builtin  = self.Vbi2[r_index, z_index]

                #
                output_values = [r_index, z_index, r, z, ep, mat_no, \
                                 poisson_v, poisson_er, poisson_ez, poisson_e, poisson_fc, \
                                 continuity_n, continuity_p, doping, builtin]

                #
                fid_out.write(output_format % tuple(output_values))

        #
        fid_out.close()

        # CPU time
        end = time.time()

        # CPU time
        return end-start
    

    # ===== save solutions (PDF file) =====
    def save_SG_scheme_solutions_in_pdf(self, output_filename):
        # CPU time
        start = time.time()

        # meshgrid
        z = range(self.Z_nodes_len)
        r = range(self.R_nodes_len)
        Z, R = np.meshgrid(z, r)
            
        # === CASE 0
        fig, ax = plt.subplots(5, 1, figsize=(8,14))
        
        # doping profile
        ax0 = ax[0].imshow((self.RZ_MATno2 + self.DP2[:-1,:-1]), origin='lower', cmap='coolwarm')
        ax[0].contour(Z, R, self.DP2, colors='k', linewidths=0.01, levels=np.logspace(-1.0, 27.0, 29*2))
        ax[0].set_title('dopant density w/ electrodes [m^-3]')
        plt.colorbar(ax0)
        
        # electric potential
        ax1 = ax[1].imshow(self.V2, origin='lower', cmap='coolwarm')     # 'RdBu'
        ax[1].contour(Z, R, self.V2, colors='k', linewidths=0.01, levels=np.linspace(-30.0, +30.0, 61*2))
        ax[1].set_title('electric potential [V]')
        plt.colorbar(ax1)
        
        # electric field
        ax2 = ax[2].imshow(self.E, origin='lower', cmap='coolwarm')
        ax[2].contour(Z[:-1,:-1], R[:-1,:-1], self.E, colors='k', linewidths=0.01, levels=np.linspace(0.0, 50e8, 51*4))
        ax[2].set_title('electric field [V/m]')
        plt.colorbar(ax2)
        
        # electron density
        ax3 = ax[3].imshow(np.log10(np.abs(self.n2)+1e-1), origin='lower', cmap='coolwarm')
        ax[3].contour(Z, R, np.log10(np.abs(self.n2)+1e-1), levels=np.linspace(-1.0, 27.0, 29*1), colors='k', linewidths=0.01)
        #ax3 = ax[3].imshow(self.n2, origin='lower')
        #ax[3].contour(Z, R, self.n2+1e-1, colors='k', linewidths=0.01, levels=np.logspace(-1.0, 27.0, 29*1))
        ax[3].set_title('LOG10(electron density) @channel [m^-3]')
        plt.colorbar(ax3)
        
        # hole density
        ax4 = ax[4].imshow(np.log10(np.abs(self.p2)+1e-1), origin='lower', cmap='coolwarm')
        ax[4].contour(Z, R, np.log10(np.abs(self.p2)+1e-1), levels=np.linspace(-1.0, 27.0, 29*1), colors='k', linewidths=0.01)
        #ax4 = ax[4].imshow(self.p2, origin='lower')
        #ax[4].contour(Z, R, self.p2+1e-1, colors='k', linewidths=0.01, levels=np.logspace(-1.0, 27.0, 29*1))
        ax[4].set_title('LOG10(hole density) @channel [m^-3]')
        plt.colorbar(ax4)
        
        #
        plt.savefig(output_filename+'_0.pdf')
        #
        plt.close()
            
        # === CASE 1
        fig, ax = plt.subplots(5, 1, figsize=(8,14))
        
        # doping profile
        ax0 = ax[0].imshow((self.RZ_MATno2 + self.DP2[:-1,:-1]), origin='lower', cmap='coolwarm')
        ax[0].contour(Z, R, self.DP2, colors='k', linewidths=0.01, levels=np.logspace(-1.0, 27.0, 29*2))
        ax[0].set_title('dopant density w/ electrodes [m^-3]')
        plt.colorbar(ax0)
        
        # electric potential
        ax1 = ax[1].imshow(self.V2, origin='lower', cmap='coolwarm')     # 'RdBu'
        ax[1].contour(Z, R, self.V2, colors='k', linewidths=0.01, levels=np.linspace(-30.0, +30.0, 61*2))
        ax[1].set_title('electric potential  [V]')
        plt.colorbar(ax1)
        
        #
        ax2 = ax[2].imshow(self.E, origin='lower', cmap='coolwarm')
        ax[2].contour(Z[:-1,:-1], R[:-1,:-1], self.E, colors='k', linewidths=0.01, levels=np.linspace(0.0, 50e8, 51*4))
        ax[2].set_title('electric field [V/m]')
        plt.colorbar(ax2)
        
        #
        #ax3 = ax[3].imshow(np.log10(np.abs(self.n2)+1e-1), origin='lower', cmap='coolwarm')
        #ax[3].contour(Z, R, np.log10(np.abs(self.n2)+1e-1), levels=np.linspace(-1.0, 27.0, 29*1), colors='k', linewidths=0.01)
        ax3 = ax[3].imshow(self.n2, origin='lower', cmap='coolwarm')
        ax[3].contour(Z, R, self.n2+1e-1, colors='k', linewidths=0.01, levels=np.logspace(-1.0, 27.0, 29*1))
        ax[3].set_title('electron density @channel [m^-3]')
        plt.colorbar(ax3)
        
        #
        #ax4 = ax[4].imshow(np.log10(np.abs(self.p2)+1e-1), origin='lower', cmap='coolwarm')
        #ax[4].contour(Z, R, np.log10(np.abs(self.p2)+1e-1), levels=np.linspace(-1.0, 27.0, 29*1), colors='k', linewidths=0.01)
        ax4 = ax[4].imshow(self.p2, origin='lower', cmap='coolwarm')
        ax[4].contour(Z, R, self.p2+1e-1, colors='k', linewidths=0.01, levels=np.logspace(-1.0, 27.0, 29*1))
        ax[4].set_title('hole density @channel [m^-3]')
        plt.colorbar(ax4)
        
        #
        plt.savefig(output_filename+'_1.pdf')
        #
        plt.close()

        # CPU time
        end = time.time()

        # CPU time
        return end-start



#============================================================================
# MAIN
#============================================================================

# z geometry split (USER INPUT)
z_geo_split = {}
z_geo_split[0] = [5, 100.0, 30.0, 320.0, 30.0, 100.0]   # 200_380_580
z_geo_split[1] = [5, 95.0,  30.0, 330.0, 30.0, 95.0 ]   # 190_390_580
z_geo_split[2] = [5, 90.0,  30.0, 340.0, 30.0, 90.0 ]   # 180_400_580
z_geo_split[3] = [5, 105.0, 30.0, 310.0, 30.0, 105.0]   # 210_370_580

# z geometry split
for each_z_geo_split_no in z_geo_split.keys():
    # debugging
    print('Z GEO SPLIT: %i' % each_z_geo_split_no, z_geo_split[each_z_geo_split_no])
                   
    # number of wls (USER INPUT)
    wl_ea = z_geo_split[each_z_geo_split_no][0]

    # material para (USER INPUT)
    mat_para_dictionary = {}
    mat_para_dictionary['TOX']       = {'mat_no':30,  'type':'I', 'k':4.8,  'qf':0.0, 'cb':3.10}
    mat_para_dictionary['CTN']       = {'mat_no':31,  'type':'I', 'k':7.5,  'qf':0.0, 'cb':2.10}
    mat_para_dictionary['BOX_SIO2']  = {'mat_no':32,  'type':'I', 'k':5.0,  'qf':0.0, 'cb':3.10}
    mat_para_dictionary['BOX_AL2O3'] = {'mat_no':33,  'type':'I', 'k':9.0,  'qf':0.0, 'cb':2.80}
    mat_para_dictionary['LINER']     = {'mat_no':11,  'type':'I', 'k':3.9,  'qf':0.0, 'cb':3.10}
    mat_para_dictionary['VOID']      = {'mat_no':10,  'type':'I', 'k':1.0,  'qf':0.0, 'cb':4.05}
    mat_para_dictionary['ON_SIO2']   = {'mat_no':34,  'type':'I', 'k':3.9,  'qf':0.0, 'cb':3.10}
    mat_para_dictionary['SI']        = {'mat_no':20,  'type':'S', 'k':11.7, 'qf':0.0, 'cb':0.00, \
                                        'n_int':1.5e16, 'mu_n':0.14, 'mu_p':0.045, 'tau_n':1e-6, 'tau_p':1e-5} # 'mu_n':0.14, 'mu_p':0.045
    for each_wl in range(wl_ea):
        each_wl_name = 'WL%03i' % each_wl
        each_wl_no   = 100 + each_wl
        mat_para_dictionary[each_wl_name]    = {'mat_no':each_wl_no, 'type':'M', 'k':1e5,  'qf':0.0, 'cb':0.0, 'wf':4.8}

    # inside plug (USER INPUT)
    uc_inward_thk_dr = {}
    uc_inward_thk_dr['CD']         = 1200                                       # angstrom
    uc_inward_thk_dr['BOX_SIO2']   = {'mat_no':32, 'thk':70.0,  'dr':10.0}       # angstrom (1st layer)
    uc_inward_thk_dr['CTN']        = {'mat_no':31, 'thk':50.0,  'dr':10.0}       # angstrom (2nd layer)
    uc_inward_thk_dr['TOX']        = {'mat_no':30, 'thk':50.0,  'dr':10.0}       # angstrom (3rd layer)
    uc_inward_thk_dr['SI']         = {'mat_no':20, 'thk':70.0,  'dr':10.0}       # angstrom (4th layer)
    uc_inward_thk_dr['LINER']      = {'mat_no':11, 'thk':120.0, 'dr':20.0}       # angstrom (5th layer)
    uc_inward_thk_dr['VOID']       = {'mat_no':10, 'thk':-1,    'dr':40.0}       # angstrom (6th layer)

    # outside plug & z stacks (USER INPUT)
    uc_outward_thk_dr = {}
    uc_z_on_thk_dz = {}

    for each_wl in range(wl_ea):
        each_wl_name = 'WL%03i' % each_wl
        each_wl_no   = 100 + each_wl
        #
        uc_outward_thk_dr[each_wl_name+'_ON_O1'] = {}
        uc_outward_thk_dr[each_wl_name+'_ON_O1']['ON_SIO2']    = {'mat_no':34,         'thk':100.0, 'dr':10.0}     # angstrom (1st layer)
        uc_z_on_thk_dz[   each_wl_name+'_ON_O1'] = {'thk':z_geo_split[each_z_geo_split_no][1],  'dz':5.0}   # angstrom
        #
        uc_outward_thk_dr[each_wl_name+'_ON_N1'] = {}
        uc_outward_thk_dr[each_wl_name+'_ON_N1']['BOX_AL2O3']  = {'mat_no':33,         'thk':100.0, 'dr':10.0}     # angstrom (1st layer)
        uc_z_on_thk_dz[   each_wl_name+'_ON_N1'] = {'thk':z_geo_split[each_z_geo_split_no][2],  'dz':5.0}   # angstrom
        #
        uc_outward_thk_dr[each_wl_name+'_ON_N2'] = {}
        uc_outward_thk_dr[each_wl_name+'_ON_N2']['BOX_AL2O3']  = {'mat_no':33,         'thk':30.0,  'dr':10.0}     # angstrom (1st layer)
        uc_outward_thk_dr[each_wl_name+'_ON_N2'][each_wl_name] = {'mat_no':each_wl_no, 'thk':70.0,  'dr':10.0}     # angstrom (2nd layer)
        uc_z_on_thk_dz[   each_wl_name+'_ON_N2'] = {'thk':z_geo_split[each_z_geo_split_no][3], 'dz':5.0}   # angstrom
        #
        uc_outward_thk_dr[each_wl_name+'_ON_N3'] = {}
        uc_outward_thk_dr[each_wl_name+'_ON_N3']['BOX_AL2O3']  = {'mat_no':33,         'thk':100.0, 'dr':10.0}     # angstrom (1st layer)
        uc_z_on_thk_dz[   each_wl_name+'_ON_N3'] = {'thk':z_geo_split[each_z_geo_split_no][4],  'dz':5.0}   # angstrom
        #
        uc_outward_thk_dr[each_wl_name+'_ON_O2'] = {}
        uc_outward_thk_dr[each_wl_name+'_ON_O2']['ON_SIO2']    = {'mat_no':34,         'thk':100.0, 'dr':10.0}     # angstrom (1st layer)
        uc_z_on_thk_dz[   each_wl_name+'_ON_O2'] = {'thk':z_geo_split[each_z_geo_split_no][5],  'dz':5.0}   # angstrom

    # preparing grid (USER INPUT)
    grid_solver = SOLVER()
    cpu_time_1 = grid_solver.add_material_parameters(mat_para_dictionary)
    cpu_time_2 = grid_solver.set_unit_cell_R_grid(inward_thk_dr=uc_inward_thk_dr, outward_thk_dr=uc_outward_thk_dr)
    cpu_time_3 = grid_solver.set_unit_cell_Z_grid(z_on_thk_dz=uc_z_on_thk_dz, z_offset=0.0)
    cpu_time_4 = grid_solver.set_unit_cell_RZ_grid()
    cpu_time_5 = grid_solver.set_unit_cell_RZ_mis_region()
    cpu_time_6 = grid_solver.add_ohmic_contact(before_info={'S':{'mat_no':20, 'z_coord':0 }}, after_info={'M':{'mat_no':10001}})     # BL
    cpu_time_7 = grid_solver.add_ohmic_contact(before_info={'S':{'mat_no':20, 'z_coord':-1}}, after_info={'M':{'mat_no':10002}})     # SL
    cpu_time_8 = grid_solver.set_semiconductor_parameters(op_temperature=25.0, tg_region={'S':{'mat_no':20}}, bl_mat_no=10001, sl_mat_no=10002, \
                                                          doping=['n', 1e20], ct_doping=['n', [1e25, 1e20]])
    cpu_time_9 = grid_solver.make_poisson_matrix()

    # FDM size
    print('FDM size')
    print('  R nodes = %iea, Z nodes = %iea' % (grid_solver.R_nodes_len, grid_solver.Z_nodes_len))
    print('  RZ nodes = %iea (sparse matrix size)' % (grid_solver.RZ_nodes_len))

    # CPU time check
    print('CPU time check list')
    print('  CPU %sea (%s threads)' % (psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)))
    print('  @add_material_parameters() = %.1e sec' % cpu_time_1)
    print('  @set_unit_cell_R_grid() = %.1e sec' % cpu_time_2)
    print('  @set_unit_cell_Z_grid() = %.1e sec' % cpu_time_3)
    print('  @set_unit_cell_RZ_grid() = %.1e sec' % cpu_time_4)
    print('  @set_unit_cell_RZ_mis_region() = %.1e sec' % cpu_time_5)
    print('  @add_ohmic_contact() = %.1e sec' % cpu_time_6)
    print('  @add_ohmic_contact() = %.1e sec' % cpu_time_7)
    print('  @set_semiconductor_parameters() = %.1e sec' % cpu_time_8)
    print('  @make_poisson_matrix() = %.1e sec' % cpu_time_9)


    # MIM MODEL (Tunneling)
    if False:
        #
        timeline = np.logspace(-7, -1, 20)

        #
        for each_index, each_time in enumerate(timeline):
            #
            if each_index == 0:
                dt = each_time
            else:
                dt = each_time - timeline[each_index-1]

            #
            print(each_index, each_time, dt)
        
            # external bias
            wl_bias = 14.0
            mim_ext_bias = {10001:0.0, 10002:0.0, 20:0.0}                                           # channel ext. bias
            for each_wl in range(wl_ea):
                each_wl_mat_no = 100 + each_wl
                if each_wl == int(wl_ea/2):
                    mim_ext_bias.update({each_wl_mat_no:wl_bias})                                   # WL ext. bias
                else:
                    mim_ext_bias.update({each_wl_mat_no:0.0})                                   # WL ext. bias
            cpu_time_10 = grid_solver.make_external_bias_vector(external_bias_conditions=mim_ext_bias, workfunction=4.8, model_type='MIM')

            # fixed charge
            #ctn_fixed_charge_density = {31:0.0e24}                                                  # fixed charge
            #cpu_time_11 = grid_solver.make_fixed_charge_vector(fixed_charge_density=ctn_fixed_charge_density, model_type='MIM')

            # poisson equation
            cpu_time_12 = grid_solver.solve_poisson_equation(model_type='MIM')
            induced_Q, induced_Q_profile, mat_no_profile, Z_profile, E_profile, E_Q_profile = grid_solver.cal_channel_induced_charge(mat_no_ch=20, mat_no_tox=30)
            thermal_vel = grid_solver.cal_thermal_velocity()
            WKB_profile, WKB_profile2, WKB_length_profile, WKB_length_profile2, mat_no_profile, Z_profile = grid_solver.cal_tunneling_probability(mat_no_tox=30, meff=0.5)

            # CTM model 1D
            grid_solver.cal_ctn_trap_model_1d(dt=dt, tox_meff=0.5, mat_no_ch=20, mat_no_tox=30, mat_no_ctn=31, ctn_peak_pos=0.5, \
                                              cnt_ccs_array=[1e-19, 1e-18, 1e-19], ctn_density_array=[5e25, 5e25, 5e25])

            # external bias
            wl_bias = 0.0
            mim_ext_bias = {10001:0.0, 10002:0.0, 20:0.0}                                           # channel ext. bias
            for each_wl in range(wl_ea):
                each_wl_mat_no = 100 + each_wl
                mim_ext_bias.update({each_wl_mat_no:wl_bias})                                       # WL ext. bias
            cpu_time_10 = grid_solver.make_external_bias_vector(external_bias_conditions=mim_ext_bias, workfunction=4.8, model_type='MIM')

            # poisson equation
            cpu_time_13 = grid_solver.solve_poisson_equation(model_type='MIM')

            # visualization: poission equation
            fig, ax = plt.subplots(1, 3, figsize=(17,5))
            ax[0].imshow(grid_solver.V2_mim, origin='lower')
            ax[1].imshow(grid_solver.E_mim, origin='lower')
            ax[2].imshow(grid_solver.FC_mim2, origin='lower')
            plt.show()

        # debugging
        print(mim_ext_bias, induced_Q)
        print(cpu_time_9, cpu_time_10, cpu_time_12)
        print(grid_solver.RZ_MIS_index_min_max)

        # visualization: poission equation
        plt.imshow(grid_solver.V2_mim, origin='lower')
        plt.imshow(grid_solver.E_mim, origin='lower')
        plt.show()

        # visualization: poission equation
        fig, ax = plt.subplots(5,1)
        ax[0].plot(Z_profile, np.array(E_Q_profile) * (thermal_vel/70e-10) * 7e-6)
        ax[0].grid(ls=':')
        ax[1].plot(Z_profile, WKB_length_profile2)
        ax[1].grid(ls=':')
        ax[2].plot(Z_profile, WKB_profile2)
        ax[2].grid(ls=':')
        ax[3].plot(Z_profile, np.array(E_Q_profile) * (thermal_vel/70e-10) * 7e-6 * np.array(WKB_profile2))
        ax[3].grid(ls=':')
        ax[4].plot(Z_profile, mat_no_profile)
        ax[4].grid(ls=':')
        plt.show()

        grid_solver.plot_conduction_band_diagram(output_filename='19V.pdf', model_type='MIM')



    # MIS MODEL (Scharfetter-Gummel scheme, Gummel iteration)
    if True:
        # preparing grid (USER INPUT)
        cpu_time_10 = grid_solver.make_continuity_matrix()

        # CPU time check
        print('  @make_continuity_matrix() = %.1e sec' % cpu_time_10)

        # geometry
        cd = uc_inward_thk_dr['CD']
        ponoa_box = uc_inward_thk_dr['BOX_SIO2']['thk']
        ponoa_ctn = uc_inward_thk_dr['CTN']['thk']
        ponoa_tox = uc_inward_thk_dr['TOX']['thk']
        ponoa_ch  = uc_inward_thk_dr['SI']['thk']
        ponoa_alo = uc_outward_thk_dr['WL000_ON_N2']['BOX_AL2O3']['thk']
        
        on_o1 = uc_z_on_thk_dz['WL000_ON_O1']['thk']
        on_n1 = uc_z_on_thk_dz['WL000_ON_N1']['thk']
        on_n2 = uc_z_on_thk_dz['WL000_ON_N2']['thk']
        on_n3 = uc_z_on_thk_dz['WL000_ON_N3']['thk']
        on_o2 = uc_z_on_thk_dz['WL000_ON_O2']['thk']
        on_o  = on_o1 + on_o2
        on_n  = on_n1 + on_n2 + on_n3
        on_pitch = on_o + on_n

        identifier = 'cd_%.1f_ponoa_%i_%i_%i_%i_%i_on_%i_%i_%i' % \
                     (cd, ponoa_ch, ponoa_tox, ponoa_ctn, ponoa_box, ponoa_alo, on_o, on_n, on_pitch)

        # WL bias sweep info
        array_div = [1, 120+1, 120+1, \
                     20+1, 120+1, 10+1, 120+1, 10+1, 120+1, \
                     30+1, 120+1, 10+1, 120+1, 10+1, 120+1, 30+1, 120+1, \
                     10+1, 120+1, 10+1, 120+1]
        array_sel_wl = [-5.0, -5.0, -5.0, +7.0, \
                        +7.0, -5.0, -5.0, +7.0, +7.0, -5.0, \
                        -5.0, +7.0, +7.0, -5.0, -5.0, +7.0, +7.0, -5.0, \
                        -5.0, +7.0, +7.0, -5.0]
        array_sel_adj_wl_bl_side = [-5.0, -5.0, +7.0, +7.0, \
                                    +6.0, +6.0, +5.0, +5.0, +4.0, +4.0, \
                                    +7.0, +7.0, +7.0, +7.0, +7.0, +7.0, +4.0, +4.0, \
                                    +5.0, +5.0, +6.0, +6.0]
        array_sel_adj_wl_sl_side = [-5.0, -5.0, +7.0, +7.0, \
                                    +6.0, +6.0, +5.0, +5.0, +4.0, +4.0, \
                                    +4.0, +4.0, +5.0, +5.0, +6.0, +6.0, +7.0, +7.0, \
                                    +7.0, +7.0, +7.0, +7.0]
        array_unsel_wl = [-5.0, -5.0, +7.0, +7.0, \
                          +7.0, +7.0, +7.0, +7.0, +7.0, +7.0, \
                          +7.0, +7.0, +7.0, +7.0, +7.0, +7.0, +7.0, +7.0, \
                          +7.0, +7.0, +7.0, +7.0]
        array_bl = [0.0, 0.0, 0.5, 0.5, \
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, \
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, \
                    0.5, 0.5, 0.5, 0.5]
        array_sl = [0.0, 0.0, 0.0, 0.0, \
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                    0.0, 0.0, 0.0, 0.0]
        
        wl_bias_sweep_info = {}
        for each_loop_no in range(len(array_div)):
            wl_bias_sweep_info[each_loop_no] = {}
            wl_bias_sweep_info[each_loop_no]['div'] = array_div[each_loop_no]
            wl_bias_sweep_info[each_loop_no]['sel_wl']             = [array_sel_wl[each_loop_no+0], array_sel_wl[each_loop_no+1]]
            wl_bias_sweep_info[each_loop_no]['sel_adj_wl_bl_side'] = [array_sel_adj_wl_bl_side[each_loop_no+0], array_sel_adj_wl_bl_side[each_loop_no+1]]
            wl_bias_sweep_info[each_loop_no]['sel_adj_wl_sl_side'] = [array_sel_adj_wl_sl_side[each_loop_no+0], array_sel_adj_wl_sl_side[each_loop_no+1]]
            wl_bias_sweep_info[each_loop_no]['unsel_wl']           = [array_unsel_wl[each_loop_no+0], array_unsel_wl[each_loop_no+1]]
            wl_bias_sweep_info[each_loop_no]['bl'] = [array_bl[each_loop_no+0], array_bl[each_loop_no+1]]
            wl_bias_sweep_info[each_loop_no]['sl'] = [array_sl[each_loop_no+0], array_sl[each_loop_no+1]]

        # external resistance
        ext_R_bl = 0.0  # 2e4
        ext_R_sl = 0.0  # 1e2

        # terminal current
        In_bl, Ip_bl, In_sl, Ip_sl = 0.0, 0.0, 0.0, 0.0

        # channel region flag
        ch_region_flag = np.where(grid_solver.CH_FLAG_serial==1.0)

        # log
        cal_civ = []
        
        # LOOP 1: LOOP sweep
        for sweep_loop_no in wl_bias_sweep_info.keys():
            
            #
            range_div          = wl_bias_sweep_info[sweep_loop_no]['div']
            info_sel_wl        = wl_bias_sweep_info[sweep_loop_no]['sel_wl']
            info_sel_adj_wl_bl = wl_bias_sweep_info[sweep_loop_no]['sel_adj_wl_bl_side']
            info_sel_adj_wl_sl = wl_bias_sweep_info[sweep_loop_no]['sel_adj_wl_sl_side']
            info_unsel_wl      = wl_bias_sweep_info[sweep_loop_no]['unsel_wl']
            info_bl            = wl_bias_sweep_info[sweep_loop_no]['bl']
            info_sl            = wl_bias_sweep_info[sweep_loop_no]['sl']
            
            #
            sel_wl_range        = np.linspace(info_sel_wl[0],        info_sel_wl[1],        range_div)
            sel_adj_wl_bl_range = np.linspace(info_sel_adj_wl_bl[0], info_sel_adj_wl_bl[1], range_div)
            sel_adj_wl_sl_range = np.linspace(info_sel_adj_wl_sl[0], info_sel_adj_wl_sl[1], range_div)
            unsel_wl_range      = np.linspace(info_unsel_wl[0],      info_unsel_wl[1],      range_div)
            bl_range            = np.linspace(info_bl[0],            info_bl[1],            range_div)
            sl_range            = np.linspace(info_sl[0],            info_sl[1],            range_div)

            # Gummel iteration (GI) parameter
            gi_w_v = 0.992                      # GI convergence control parameter (>0.99)
            gi_w_np = 0.992                     # GI convergence control parameter (>0.99)
            gi_error_v = 6e-5                   # GI convergence control parameter  
            gi_error_n = 1e23                   # GI convergence control parameter (<1e23)

            # timeline
            timeline_full = [1e-5]              # GI convergence control parameter (current LKG level < 1E-12A as dt > 1e-8 sec)

            # log
            cal_log = []

            # LOOP 2: WL bias sweep
            for each_div_index in range(range_div):
                
                # CPU time
                start = time.time()

                # current time
                print('\n', time.ctime(), identifier)

                # ext. bias (initial)
                ext_R_bl_drop, ext_R_sl_drop = In_bl * ext_R_bl, In_sl * ext_R_sl
                ext_bias = {10001:bl_range[each_div_index] - ext_R_bl_drop,\
                            10002:sl_range[each_div_index] + ext_R_sl_drop}             # BL, SL ext. bias
                for each_wl in range(wl_ea):
                    each_wl_mat_no = 100 + each_wl
                    if each_wl == int(wl_ea/2):
                        ext_bias.update({each_wl_mat_no:sel_wl_range[each_div_index]})                  # sel WL ext. bias
                    elif each_wl == (int(wl_ea/2)-1):
                        ext_bias.update({each_wl_mat_no:sel_adj_wl_bl_range[each_div_index]})           # sel ADJ. WL bl side ext. bias
                    elif each_wl == (int(wl_ea/2)+1):
                        ext_bias.update({each_wl_mat_no:sel_adj_wl_sl_range[each_div_index]})           # sel ADJ. WL bl side ext. bias
                    else:
                        ext_bias.update({each_wl_mat_no:unsel_wl_range[each_div_index]})                # unsel WL ext. bias
                        
                # set up ext. bias
                grid_solver.make_external_bias_vector(external_bias_conditions=ext_bias, workfunction=4.8, model_type='MIS')

                # poission equation solver
                grid_solver.solve_poisson_equation(model_type='MIS')
                
                # error check (start)
                old_v1 = grid_solver.V1

                # LOOP 3: time evolution
                output_filename = ''
                for each_time_index, each_time in enumerate(timeline_full):

                    # CPU time
                    start = time.time()
                    
                    # calculating dt
                    if each_time_index == 0:
                        dt = each_time
                    else:
                        dt = timeline_full[each_time_index] - timeline_full[each_time_index-1]

                    # continuity equation solver
                    grid_solver.solve_continuity_equation(dt=dt)

                    # error check (start)
                    old_n1 = grid_solver.n1
                    old_p1 = grid_solver.p1

                    # mixing decoupled solutions from continuity equation solver
                    grid_solver.n1 = old_n1 * gi_w_np + grid_solver.n1 * ( 1.0 - gi_w_np )
                    grid_solver.p1 = old_p1 * gi_w_np + grid_solver.p1 * ( 1.0 - gi_w_np )

                    # poission equation solver
                    grid_solver.solve_poisson_equation(model_type='MIS')

                    # mixing decoupled solutions from poisson equation solver
                    grid_solver.V1 = old_v1 * gi_w_v + grid_solver.V1 * ( 1.0 - gi_w_v )
                        
                    # continuity equation solver
                    grid_solver.solve_continuity_equation(dt=dt)

                    # calculating error (first error check)
                    error_v = np.max( np.abs( old_v1 - grid_solver.V1 ) )
                    error_n = np.max( np.abs( old_n1 - grid_solver.n1 ) )
                    error_p = np.max( np.abs( old_p1 - grid_solver.p1 ) )

                    # error check (start)
                    old_v1 = grid_solver.V1
                    old_n1 = grid_solver.n1
                    old_p1 = grid_solver.p1

                    # debugging (convergence error check)
                    print('error check %.3e,%.3e,%.3e  ,%.3e (%.2f%%)' % (error_v, error_n, error_p, gi_error_n, error_n/gi_error_n*100))

                    # LOOP 4: Gummel iteration (for solution convergence)
                    error_v, error_n, error_p, gi_no = 1.0e40, 1.0e40, 1.0e40, 0
                    while error_n > gi_error_n:

                        # mixing decoupled solutions from continuity equation solver
                        grid_solver.n1 = old_n1 * gi_w_np + grid_solver.n1 * ( 1.0 - gi_w_np )
                        grid_solver.p1 = old_p1 * gi_w_np + grid_solver.p1 * ( 1.0 - gi_w_np )
                        
                        # poission equation solver
                        grid_solver.solve_poisson_equation(model_type='MIS')

                        # mixing decoupled solutions from poisson equation solver
                        grid_solver.V1 = old_v1 * gi_w_v + grid_solver.V1 * ( 1.0 - gi_w_v )
                        
                        # continuity equation solver
                        grid_solver.solve_continuity_equation(dt=dt)
                        
                        # calculating error
                        error_v = np.max( np.abs( old_v1 - grid_solver.V1 ) )
                        error_n = np.max( np.abs( old_n1 - grid_solver.n1 ) )
                        error_p = np.max( np.abs( old_p1 - grid_solver.p1 ) )
                        
                        # error check (Gummel iteration loop)
                        old_v1 = grid_solver.V1
                        old_n1 = grid_solver.n1
                        old_p1 = grid_solver.p1

                        # calculate BL, SL terminal current
                        In_bl, Ip_bl, In_sl, Ip_sl = grid_solver.cal_bl_sl_current(bl_mat_no=10001, sl_mat_no=10002)

                        # log
                        output_format = '%i,%i,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%i,%i,%.4f,%.4f,%.2e,%.2e,%.2e,%.2e,%.2e,%.2e,%.2e,%.2e,%.2e'
                        output_value  = [sweep_loop_no, each_div_index, \
                                         sel_wl_range[each_div_index], sel_adj_wl_bl_range[each_div_index], sel_adj_wl_sl_range[each_div_index], \
                                         unsel_wl_range[each_div_index], bl_range[each_div_index], sl_range[each_div_index], \
                                         gi_no, each_time_index, gi_w_v, gi_w_np, each_time, dt, \
                                         error_v, error_n, error_p,\
                                         In_bl, Ip_bl, In_sl, Ip_sl]
                        cal_log.append(output_value)

                        # debugging (for every 100 Gummel iteration loop count)
                        if gi_no % 100 == 0:
                            print(output_format % tuple(output_value))

                        # update Gummel iteration loop count
                        gi_no += 1

                    # CPU time
                    end = time.time()
                    cpu_time_31 = end - start
                    
                # output filename
                output_filename = '%iWL_SG_scheme_Gummel_iter_ZSPLIT%i_LOOP%i_%i_%.3f_%.3f_%.3f_%.3f_%.3f_%.3f_elapsed_time_%i_%.3e_dt_%.3e_w_%.4f_%.4f_%i' % \
                                  (wl_ea, each_z_geo_split_no, sweep_loop_no, each_div_index, \
                                   sel_wl_range[each_div_index], sel_adj_wl_bl_range[each_div_index], sel_adj_wl_sl_range[each_div_index], \
                                   unsel_wl_range[each_div_index], bl_range[each_div_index], sl_range[each_div_index], \
                                   each_time_index, each_time, dt, gi_w_v, gi_w_np, gi_no)

                # debugging (for last Gummel iteration loop count)
                print(output_format % tuple(output_value))
                print('%.2e,%.2e,%.2e, %.2e,%.2e,%.2e, %.2e,%.2e,%.2e, %.2e,%.2e' % \
                       (np.max(grid_solver.n2[:,0]),np.max(grid_solver.n1[ch_region_flag]),np.max(grid_solver.n2[:,-1]),\
                        np.max(grid_solver.p2[:,0]),np.max(grid_solver.p1[ch_region_flag]),np.max(grid_solver.p2[:,-1]),\
                        error_v, error_n, error_p, ext_R_bl_drop, ext_R_sl_drop))

                # file output 1 (every bias change conditions)
                if ( np.abs(In_bl) > 10e-9 ) and ( np.abs(In_bl) < 100e-9 ) and ( In_bl > 0) and ( In_sl > 0): 
                    cpu_time_41 = grid_solver.save_SG_scheme_solutions_in_txt(output_filename = output_filename)
                    cpu_time_42 = grid_solver.save_SG_scheme_solutions_in_pdf(output_filename = output_filename)
                    print('Gummel iter = %iea, %.3f sec, file output = %.3f sec (txt), %.3f sec (pdf)' % (gi_no, cpu_time_31, cpu_time_41, cpu_time_42))
                else:
                    print('Gummel iter = %iea, %.3f sec' % (gi_no, cpu_time_31))

                # CIV output (collecting data)
                cal_civ.append([identifier, wl_ea, each_z_geo_split_no, sweep_loop_no, gi_no, \
                                sel_wl_range[each_div_index], sel_adj_wl_bl_range[each_div_index], sel_adj_wl_sl_range[each_div_index], unsel_wl_range[each_div_index], \
                                bl_range[each_div_index], sl_range[each_div_index], \
                                In_bl, Ip_bl, In_sl, Ip_sl])

            # file output 2 (final resulats)
            fid_out = open(output_filename + '_all.txt', 'w')
            fid_out.write('IDENTIFIER,WLs,ZSPLIT,LOOP_C,LOOP_V,SEL_WL_V,SEL_ADJ_WL_BL_V,SEL_ADJ_WL_SL_V,UNSEL_WL_V,BL_V,SL_V,GUMMEL_ITER,LOOP_T,GUMMEL_W_V,GUMMEL_W_NP,TIME,TIME_dt,' + \
                          'ERROR_V,ERROR_N,ERROR_P,In_BL,Ip_BL,In_SL,Ip_SL' + '\n')
            output_format = '%s,%i,%i,%i,%i,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%i,%i,%.4f,%.4f,%.2e,%.2e,%.2e,%.2e,%.2e,%.2e,%.2e,%.2e,%.2e' + '\n'
            for each_line_data in cal_log:
                fid_out.write(output_format % tuple([identifier, wl_ea, each_z_geo_split_no] + each_line_data))
            fid_out.close()

            # CIV output (console output)
            fid_out = open(output_filename + '_civ.txt', 'w')
            fid_out.write('IDENTIFIER,WLs,ZSPLIT,LOOP_C,GUMMEL_ITER_NO,SEL_WL_V,SEL_ADJ_WL_BL_V,SEL_ADJ_WL_SL_V,UNSEL_WL_V,BL_V,SL_V,In_BL,Ip_BL,In_SL,Ip_SL' + '\n')
            civ_output_format = '%s,%i,%i,%i,%i,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3e,%.3e,%.3e,%.3e' + '\n'
            for each_cal_civ in cal_civ:
                fid_out.write( civ_output_format % tuple(each_cal_civ) )
            fid_out.close()

                
      
