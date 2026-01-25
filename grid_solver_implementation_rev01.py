#
# TITLE: solving the van Roosbroeck system numerically (poisson solver)
# AUTHOR: Hyunseung Yoo
# PURPOSE: 
# REVISION: 
# REFERENCE: a numerical study of the van Roosbroeck system for semiconductor (SJSU, 2018) 
#

import sys
import time
import copy
import numpy as np
import scipy as sc
import sympy as sp
import matplotlib.pyplot as plt

#
# CLASS: SOLVER
#

class SOLVER:

    # fundamental constants
    q = 1.60217663e-19      # [C]
    k_b = 1.380649e-23      # [m]^2 [kg] / ( [s]^2 [K] )
    ep0 = 8.854187e-12      # [F] / [m]

    # material constants
    k_si = 11.68
    n_int = 1.5e16          # 1 / [m]^3
    mu_n = 0.14             # [m]^2 / ( [V] [s] ) 
    mu_p = 0.045            # [m]^2 / ( [V] [s] )

    # operation temperature
    T = 300.0               # [K]

    # ===== constructor =====
    def __init__(self):
        # solution (poisson equation)
        self.V = []
        self.E = []
        self.Er = []
        self.Ez = []
        # solution (continuity equations)
        self.n = []
        self.p = []

    # ===== making grid =====
    def make_grid(self, cd, r_range, z_range):
        # user input
        self.cd = cd / 2.0      # angstrom

        # user input: start coordinate, end coordinate, delta
        r_st, r_ed, dr = r_range
        z_st, z_ed, dz = z_range

        # calculations: position (elements, nodes)
        r_elemts = int( (r_ed - r_st) / dr )
        r_nodes = r_elemts + 1
        z_elemts = int( (z_ed - z_st) / dz )
        z_nodes = z_elemts + 1

        # debugging
        if True:
            output_string_r1 = 'r_start = %.1f, r_end = %.1f, dr = %.1f' % tuple(r_range)
            output_string_r2 = 'r_elements = %i, r_nodes = %i' % (r_elemts, r_nodes)
            output_string_z1 = 'z_start = %.1f, z_end = %.1f, dz = %.1f' % tuple(z_range)
            output_string_z2 = 'z_elements = %i, z_nodes = %i' % (z_elemts, z_nodes)
            print(output_string_r1)
            print(output_string_r2)
            print(output_string_z1)
            print(output_string_z2)

        # calculations: 1D coordinate (nodes)
        self.R = np.linspace(r_st, r_ed, r_nodes)               # coordinate
        self.Z = np.linspace(z_st, z_ed, z_nodes)               # coordiante
        self.Rnodes  = len(self.R)                              # index
        self.Znodes  = len(self.Z)                              # index
        self.RZnodes = self.Rnodes * self.Znodes                # size

        # calculations: 1D avg. coordinate between neighor nodes (elements)
        self.Ravg = ( self.R[:-1] + self.R[1:] ) / 2.0          # coordinate
        self.Zavg = ( self.Z[:-1] + self.Z[1:] ) / 2.0          # coordinate
        self.Relmts = len(self.Ravg)                            # index
        self.Zelmts = len(self.Zavg)                            # index

        # calculations: 2D coordinate (nodes, nodes)
        self.RZ_R = np.zeros( (self.Rnodes, self.Znodes) )
        self.RZ_Z = np.zeros( (self.Rnodes, self.Znodes) )
        for z_cnt in range(self.Znodes):
            self.RZ_R[:,z_cnt] = copy.copy( self.R )            # coordinate
        for r_cnt in range(self.Rnodes):
            self.RZ_Z[r_cnt,:] = copy.copy( self.Z )            # coordinate

        # calculations: 2D avg. coordinate (elements, elements)
        self.RZ_Ravg = np.zeros( (self.Relmts, self.Zelmts) )
        self.RZ_Zavg = np.zeros( (self.Relmts, self.Zelmts) )
        for z_cnt in range(self.Zelmts):
            self.RZ_Ravg[:,z_cnt] = copy.copy( self.Ravg )      # coordinate
        for r_cnt in range(self.Relmts):
            self.RZ_Zavg[r_cnt,:] = copy.copy( self.Zavg )      # coordinate

        # calculations: 2D distance between neighbor nodes (elements, nodes)
        self.RZ_dR = self.RZ_R[1:,:] - self.RZ_R[:-1,:]         # distance
        # calculations: 2D distance between neighbor nodes (nodes, elements)
        self.RZ_dZ = self.RZ_Z[:,1:] - self.RZ_Z[:,:-1]         # distance

        # calculations: 2D distance between neighbor nodes (elements-1, elements)
        self.RZ_dRavg = self.RZ_Ravg[1:,:] - self.RZ_Ravg[:-1,:]        # distance
        # calculations: 2D distance between neighbor nodes (elements, elements-1)
        self.RZ_dZavg = self.RZ_Zavg[:,1:] - self.RZ_Zavg[:,:-1]        # distance

        # calculations: 2D material number (elements, elements)
        WL_Zavg_range = np.logical_and( self.RZ_Zavg > 100.0, self.RZ_Zavg < 330.0 )    # index of elements
        WL_Ravg_range = self.RZ_Ravg > 500.0                                            # index of elements
        CH_Ravg_range = np.logical_and( self.RZ_Ravg > 300.0, self.RZ_Ravg < 400.0 )    # index of elements

        # calculations: 2D material number (elements, elements)
        self.MAT  = np.zeros( (self.Relmts, self.Zelmts) )              # dielectrics
        self.MAT += np.where( CH_Ravg_range, 100, 0)                    # channel
        self.MAT += np.where( WL_Zavg_range * WL_Ravg_range, 110, 0)    # wordline

        # electric permittivity: 2D electric permittivity (elements, elements)
        self.EP  = np.zeros( (self.Relmts, self.Zelmts) )
        self.EP += np.where( self.MAT == 0, self.ep0 * 3.9, 0.0)                  # vacuum
        self.EP += np.where( self.MAT == 100, self.ep0 * self.k_si, 0.0)    # channel
        self.EP += np.where( self.MAT == 110, self.ep0 * 9999.0, 0.0)       # metal

        # calculations: 2D avg. electric permittivity (elements_avg)
        self.EP_Zavg = ( self.EP[:,:-1] + self.EP[:,1:] ) / 2.0         # (elements, elements-1)
        self.EP_Ravg = ( self.EP[:-1,:] + self.EP[1:,:] ) / 2.0         # (elements-1, elements)

        # doping: 2D material parameter (nodes, nodes)
        self.DP = np.zeros( (self.Rnodes, self.Znodes) )

        # mobility: 2D material parameter (elements, elements)
        self.MN = np.zeros( (self.Relmts, self.Zelmts) )
        self.MP = np.zeros( (self.Relmts, self.Zelmts) )

        # calculation: channel switch
        self.CH_sw_elmts = np.where( self.MAT == 100, 1.0, 0.0 )
        self.CH_sw_elmts_r_ext = np.zeros( (self.Rnodes, self.Relmts) )
        self.CH_sw_elmts_z_ext = np.zeros( (self.Relmts, self.Znodes) )
        self.CH_sw_nodes = np.zeros( (self.Rnodes, self.Znodes) )
        
        for r_elmt_cnt in range(self.Relmts):
            for z_elmt_cnt in range(self.Zelmts):
                if self.CH_sw_elmts[r_elmt_cnt, z_elmt_cnt] == 1.0:
                    #
                    self.CH_sw_elmts_r_ext[r_elmt_cnt+0, z_elmt_cnt+0] = 1.0
                    self.CH_sw_elmts_r_ext[r_elmt_cnt+1, z_elmt_cnt+0] = 1.0
                    #
                    self.CH_sw_elmts_z_ext[r_elmt_cnt+0, z_elmt_cnt+0] = 1.0
                    self.CH_sw_elmts_z_ext[r_elmt_cnt+0, z_elmt_cnt+1] = 1.0
                    #
                    self.CH_sw_nodes[r_elmt_cnt+0, z_elmt_cnt+0] = 1.0
                    self.CH_sw_nodes[r_elmt_cnt+1, z_elmt_cnt+0] = 1.0
                    self.CH_sw_nodes[r_elmt_cnt+0, z_elmt_cnt+1] = 1.0
                    self.CH_sw_nodes[r_elmt_cnt+1, z_elmt_cnt+1] = 1.0

        # calculation: channel region (selected element index)
        ch_r = np.where( self.MAT == 100 )[0]
        ch_z = np.where( self.MAT == 100 )[1]
        
        self.CH_elmts = set()
        self.CH_elmts_r_ext = set()
        self.CH_elmts_z_ext = set()
        self.CH_nodes = set()
        
        for each_mat_index in range(len(ch_r)):
            #
            self.CH_elmts.add( (ch_r[each_mat_index]+0, ch_z[each_mat_index]+0) )
            #
            self.CH_elmts_r_ext.add( (ch_r[each_mat_index]+0, ch_z[each_mat_index]+0) )
            self.CH_elmts_r_ext.add( (ch_r[each_mat_index]+1, ch_z[each_mat_index]+0) )
            #
            self.CH_elmts_z_ext.add( (ch_r[each_mat_index]+0, ch_z[each_mat_index]+0) )
            self.CH_elmts_z_ext.add( (ch_r[each_mat_index]+0, ch_z[each_mat_index]+1) )
            #
            self.CH_nodes.add( (ch_r[each_mat_index]+0, ch_z[each_mat_index]+0) )
            self.CH_nodes.add( (ch_r[each_mat_index]+1, ch_z[each_mat_index]+0) )
            self.CH_nodes.add( (ch_r[each_mat_index]+0, ch_z[each_mat_index]+1) )
            self.CH_nodes.add( (ch_r[each_mat_index]+1, ch_z[each_mat_index]+1) )
        #
        self.CH_elmts = list(self.CH_elmts)
        self.CH_elmts_r_ext = list(self.CH_elmts_r_ext)
        self.CH_elmts_z_ext = list(self.CH_elmts_z_ext)
        self.CH_nodes = list(self.CH_nodes)

        # debugging
        print('size of RZ_R = %.3f MB' % (sys.getsizeof(self.RZ_R)/1024**2) )

    # ===== channel doping =====
    def channel_doping(self, dopant_density):
        # (+) n-type dopant, (-) p-type dopant
        self.DP = self.CH_sw_nodes * dopant_density
        #
        self.n = ( np.sqrt( self.DP**2 + 4.0*self.n_int**2 ) + self.DP ) / 2.0
        self.p = ( np.sqrt( self.DP**2 + 4.0*self.n_int**2 ) - self.DP ) / 2.0

    # ===== calculate built-in potential =====
    def cal_built_in_potential(self, temp_celsius):
        #
        self.T = 273.15 + temp_celsius
        self.Vther = self.k_b * self.T / self.q
        #
        self.Vbin  = np.log( ( self.DP + np.sqrt( self.DP**2 + 4.0 * self.n_int ) ) / ( 2.0 * self.n_int ) )
        self.Vbin *= (self.Vther * self.CH_sw_nodes)
        
    # ===== ohmic contacts =====
    def ohmic_contacts(self, bl_mat_no, sl_mat_no):
        # BL contact
        bl_z = 0
        for each_bl_r in np.where( self.MAT[:,bl_z] == 100 )[0]:
            self.MAT[each_bl_r, bl_z] = bl_mat_no
        # SL contact
        sl_z = -1
        for each_sl_r in np.where( self.MAT[:,sl_z] == 100 )[0]:
            self.MAT[each_sl_r, sl_z] = sl_mat_no

    # ===== setting external bias =====
    def set_external_bias(self, mat_no_ext_bias_cond):
        # 1D serialization
        self.ext_bias = np.zeros(self.RZnodes)
        #
        ext_bias_region = []
        # setting external bias
        for each_mat_no in mat_no_ext_bias_cond.keys():
            # external bias applied regions
            each_region = np.where( self.MAT == each_mat_no )
            ext_bias_region.append(each_region)
            # setting external bias
            for each_point in range(len(each_region[0])):
                #
                r_node_cnt, z_node_cnt = each_region[0][each_point], each_region[1][each_point]
                #
                index_r_z = self.Rnodes * (z_node_cnt+0) + (r_node_cnt+0)
                # built-in potential
                built_in_potential = self.Vbin[r_node_cnt, z_node_cnt]
                # adding external bias
                self.ext_bias[index_r_z] = built_in_potential + mat_no_ext_bias_cond[each_mat_no]

        # return
        return ext_bias_region

    # ===== making poisson matrix =====
    def make_poisson_matrix(self, coord_type, ext_bias_region):
        #
        self.PM = sc.sparse.dok_matrix((self.RZnodes, self.RZnodes))

        # metal region: dirichlet boundary conditions
        for each_region in ext_bias_region:
            for each_point in range(len(each_region[0])):
                # 2D array index
                r_node_cnt, z_node_cnt = each_region[0][each_point], each_region[1][each_point]
                # 1D serialization
                index_r_z = self.Rnodes * (z_node_cnt+0) + (r_node_cnt+0)
                # dirichlet boundary conditions
                self.PM[index_r_z, index_r_z] = 1.0

        # r boundary: neumann boundary conditions
        for r_node_cnt in [0, self.Rnodes-1]:
            for z_node_cnt in range(1,self.Znodes-1):
                # 1D serialization
                index_r_z   = self.Rnodes * (z_node_cnt+0) + (r_node_cnt+0)
                index_rm1_z = self.Rnodes * (z_node_cnt+0) + (r_node_cnt-1)
                index_rp1_z = self.Rnodes * (z_node_cnt+0) + (r_node_cnt+1)
                # excluding metal electrodes
                if self.PM[index_r_z, index_r_z] != 1.0:
                    # neumann boundary conditions
                    self.PM[index_r_z, index_r_z  ] = 1.0
                    # neumann boundary conditions
                    if r_node_cnt == 0:
                        self.PM[index_r_z, index_rp1_z] = -1.0
                    elif r_node_cnt == (self.Rnodes-1):
                        self.PM[index_r_z, index_rm1_z] = -1.0

        # z boundary: neumann boundary conditions
        for r_node_cnt in range(self.Rnodes):
            for z_node_cnt in [0, self.Znodes-1]:
                # 1D serialization
                index_r_z   = self.Rnodes * (z_node_cnt+0) + (r_node_cnt+0)
                index_r_zm1 = self.Rnodes * (z_node_cnt-1) + (r_node_cnt+0)
                index_r_zp1 = self.Rnodes * (z_node_cnt+1) + (r_node_cnt+0)
                # excluding metal electrodes
                if self.PM[index_r_z, index_r_z] != 1.0:
                    # neumann boundary conditions
                    self.PM[index_r_z, index_r_z  ] = 1.0
                    # neumann boundary conditions
                    if z_node_cnt == 0:
                        self.PM[index_r_z, index_r_zp1] = -1.0
                    elif z_node_cnt == (self.Znodes-1):
                        self.PM[index_r_z, index_r_zm1] = -1.0

        # inside boundary
        for r_node_cnt in range(1, self.Rnodes-1):
            for z_node_cnt in range(1, self.Znodes-1):
                # 1D serialization
                index_r_z   = self.Rnodes * (z_node_cnt+0) + (r_node_cnt+0)
                index_rm1_z = self.Rnodes * (z_node_cnt+0) + (r_node_cnt-1)
                index_rp1_z = self.Rnodes * (z_node_cnt+0) + (r_node_cnt+1)
                index_r_zm1 = self.Rnodes * (z_node_cnt-1) + (r_node_cnt+0)
                index_r_zp1 = self.Rnodes * (z_node_cnt+1) + (r_node_cnt+0)
                # check inside boundary
                if self.PM[index_r_z, index_r_z] != 1.0:
                    # calculate poisson matrix
                    A_rm1_z_cyl = self.EP_Zavg[r_node_cnt-1, z_node_cnt-1] * \
                                  ( self.RZ_Ravg[r_node_cnt-1, z_node_cnt-1] / self.RZ_R[r_node_cnt+0, z_node_cnt+0] ) / \
                                  ( self.RZ_dR[r_node_cnt-1, z_node_cnt+0] * self.RZ_dRavg[r_node_cnt-1, z_node_cnt+0] * 1e-20)
                    A_rp1_z_cyl = self.EP_Zavg[r_node_cnt+0, z_node_cnt-1] * \
                                  ( self.RZ_Ravg[r_node_cnt+0, z_node_cnt-1] / self.RZ_R[r_node_cnt+0, z_node_cnt+0] ) / \
                                  ( self.RZ_dR[r_node_cnt+0, z_node_cnt+0] * self.RZ_dRavg[r_node_cnt-1, z_node_cnt+0] * 1e-20)
                    A_rm1_z_rec = self.EP_Zavg[r_node_cnt-1, z_node_cnt-1] / \
                                  ( self.RZ_dR[r_node_cnt-1, z_node_cnt+0] * self.RZ_dRavg[r_node_cnt-1, z_node_cnt+0] * 1e-20)
                    A_rp1_z_rec = self.EP_Zavg[r_node_cnt+0, z_node_cnt-1] / \
                                  ( self.RZ_dR[r_node_cnt+0, z_node_cnt+0] * self.RZ_dRavg[r_node_cnt-1, z_node_cnt+0] * 1e-20)
                    A_r_zm1_rec = self.EP_Ravg[r_node_cnt-1, z_node_cnt-1] / \
                                  ( self.RZ_dZ[r_node_cnt+0, z_node_cnt-1] * self.RZ_dZavg[r_node_cnt+0, z_node_cnt-1] * 1e-20)
                    A_r_zp1_rec = self.EP_Ravg[r_node_cnt-1, z_node_cnt+0] / \
                                  ( self.RZ_dZ[r_node_cnt+0, z_node_cnt+0] * self.RZ_dZavg[r_node_cnt+0, z_node_cnt-1] * 1e-20)
                    # select coordinates system
                    if coord_type == '2d_rec':
                        self.PM[index_r_z, index_r_z  ] =  A_rm1_z_rec + A_rp1_z_rec + A_r_zm1_rec + A_r_zp1_rec
                        self.PM[index_r_z, index_rm1_z] = -A_rm1_z_rec
                        self.PM[index_r_z, index_rp1_z] = -A_rp1_z_rec
                        self.PM[index_r_z, index_r_zm1] = -A_r_zm1_rec
                        self.PM[index_r_z, index_r_zp1] = -A_r_zp1_rec
                    elif coord_type == '2d_cyl':
                        self.PM[index_r_z, index_r_z  ] =  A_rm1_z_cyl + A_rp1_z_cyl + A_r_zm1_rec + A_r_zp1_rec
                        self.PM[index_r_z, index_rm1_z] = -A_rm1_z_cyl
                        self.PM[index_r_z, index_rp1_z] = -A_rp1_z_cyl
                        self.PM[index_r_z, index_r_zm1] = -A_r_zm1_rec
                        self.PM[index_r_z, index_r_zp1] = -A_r_zp1_rec
                    else:
                        print('make_poisson_matrix()... error... invalid coordinate system...')

        # CSR format
        self.PMcsr = self.PM.tocsr()
        self.PMdense = self.PM.todense()

    # ===== solving poisson equation =====
    def solve_poisson_equation(self, coord_type, mat_no_ext_bias_cond):
        # external bias conditions
        ext_bias_region = self.set_external_bias(mat_no_ext_bias_cond)

        # making poisson matrix
        self.make_poisson_matrix(coord_type, ext_bias_region)

        # solving poission equation (sparse matrix solver)
        self.V = sc.sparse.linalg.spsolve(self.PMcsr, self.ext_bias)

        # 1D serialization -> 2D array
        self.V2 = np.resize(self.V, (self.Znodes, self.Rnodes)).T
        self.Er = ( self.V2[1:,:] - self.V2[:-1,:] ) / self.RZ_dR * 100
        self.Ez = ( self.V2[:,1:] - self.V2[:,:-1] ) / self.RZ_dZ * 100
        self.E  = np.sqrt( self.Er[:,:-1]**2 + self.Ez[:-1,:]**2 )

        # check region
        for r_node_index, z_node_index in self.CH_nodes:
            output_string = '(%4i, %4i) %.3f' % (r_node_index, z_node_index, self.V2[r_node_index, z_node_index])

        # debugging
        fig, ax = plt.subplots(1,3,figsize=(15,5))
        ax[0].imshow(self.V2, origin='lower')
        ax[1].imshow(self.E, origin='lower')
        ax[2].imshow(self.E, origin='lower')
        R, Z = np.meshgrid(range(len(self.Zavg)), range(len(self.Ravg)))
        ax[2].quiver(R, Z, self.Ez[:-1,:], self.Er[:,:-1], width=0.005, scale=1.0/0.09)
        plt.show()

    
                

#
# MAIN
#

solver = SOLVER()
solver.make_grid(cd=1300, r_range=[0, 750, 5], z_range=[0, 430, 5])
solver.channel_doping(dopant_density=1.0e18)
solver.cal_built_in_potential(temp_celsius=25.0)
solver.ohmic_contacts(bl_mat_no=101, sl_mat_no=102)
solver.solve_poisson_equation(coord_type='2d_cyl', mat_no_ext_bias_cond={101:0.5, 102:0.0, 110:1.0})




