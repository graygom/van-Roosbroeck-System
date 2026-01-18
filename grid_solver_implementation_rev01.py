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

    # fundamental constant
    q = 1.60217663e-19      # [C]
    k_b = 1.380649e-23      # [m]^2 [kg] / ( [s]^2 [K] )
    ep0 = 8.854187e-12      # [F] / [m]

    # ===== constructor =====
    def __init__(self):
        # solution (poisson equation)
        self.V = []
        self.E = []
        self.Er = []
        self.Ez = []

    # ===== making grid =====
    def make_grid(self, cd, r_range, z_range):
        # user input
        self.cd = cd / 2.0      # angstrom

        # user input: start coordinate, end coordinate, delta
        r_st, r_ed, dr = r_range
        z_st, z_ed, dz = z_range

        # calculations: position (nodes)
        r_div = int( (r_ed - r_st) / dr )
        r_nodes = r_div + 1
        z_div = int( (z_ed - z_st) / dz )
        z_nodes = z_div + 1

        # debugging
        if True:
            output_string_r1 = 'r_start = %.1f, r_end = %.1f, dr = %.1f' % tuple(r_range)
            output_string_r2 = 'r_elements = %i, r_nodes = %i' % (r_div, r_nodes)
            output_string_z1 = 'z_start = %.1f, z_end = %.1f, dz = %.1f' % tuple(z_range)
            output_string_z2 = 'z_elements = %i, z_nodes = %i' % (z_div, z_nodes)
            print(output_string_r1)
            print(output_string_r2)
            print(output_string_z1)
            print(output_string_z2)

        # calculations: 1D coordinate (nodes)
        self.R = np.linspace(r_st, r_ed, r_nodes)
        self.Z = np.linspace(z_st, z_ed, z_nodes)
        self.Rnodes  = len(self.R)
        self.Znodes  = len(self.Z)
        self.RZnodes = self.Rnodes * self.Znodes

        # calculations: 1D avg. coordinate between neighor nodes (elements)
        self.Ravg = ( self.R[:-1] + self.R[1:] ) / 2.0
        self.Zavg = ( self.Z[:-1] + self.Z[1:] ) / 2.0
        self.Relmts = len(self.Ravg)
        self.Zelmts = len(self.Zavg)

        # calculations: 2D coordinate (nodes)
        self.RZ_R = np.zeros( (self.Rnodes, self.Znodes) )
        self.RZ_Z = np.zeros( (self.Rnodes, self.Znodes) )
        for z_cnt in range(self.Znodes):
            self.RZ_R[:,z_cnt] = copy.copy( self.R )
        for r_cnt in range(self.Rnodes):
            self.RZ_Z[r_cnt,:] = copy.copy( self.Z )

        # calculations: 2D avg. coordinate (elements)
        self.RZ_Ravg = np.zeros( (self.Relmts, self.Zelmts) )
        self.RZ_Zavg = np.zeros( (self.Relmts, self.Zelmts) )
        for z_cnt in range(self.Zelmts):
            self.RZ_Ravg[:,z_cnt] = copy.copy( self.Ravg )
        for r_cnt in range(self.Relmts):
            self.RZ_Zavg[r_cnt,:] = copy.copy( self.Zavg )

        # calculations: 2D distance between neighbor nodes (nodes + nodes_diff)
        self.RZ_dR = self.RZ_R[1:,:] - self.RZ_R[:-1,:]
        self.RZ_dZ = self.RZ_Z[:,1:] - self.RZ_Z[:,:-1]

        # calculations: 2D distance between neighbor nodes (elements + elements_diff)
        self.RZ_dRavg = self.RZ_Ravg[1:,:] - self.RZ_Ravg[:-1,:]
        self.RZ_dZavg = self.RZ_Zavg[:,1:] - self.RZ_Zavg[:,:-1]

        # calculations: 2D material number (elements)
        WL_Zavg_range = np.logical_and( self.RZ_Zavg > 100.0, self.RZ_Zavg < 330.0 )
        WL_Ravg_range = self.RZ_Ravg > 300.0
        CH_Ravg_range = np.logical_and( self.RZ_Ravg > 100.0, self.RZ_Ravg < 170.0 )
        
        self.MAT  = np.zeros( (self.Relmts, self.Zelmts) )
        self.MAT += np.where( CH_Ravg_range, 100, 0)
        self.MAT += np.where( WL_Zavg_range * WL_Ravg_range, 110, 0)

        # electric permittivity: 2D electric permittivity (elements)
        self.EP  = np.zeros( (self.Relmts, self.Zelmts) )
        self.EP += np.where( self.MAT == 0, self.ep0, 0.0)
        self.EP += np.where( self.MAT == 100, self.ep0 * 999.0, 0.0)
        self.EP += np.where( self.MAT == 110, self.ep0 * 999.0, 0.0)

        # calculations: 2D avg. electric permittivity (elements_avg)
        self.EP_Zavg = ( self.EP[:,:-1] + self.EP[:,1:] ) / 2.0
        self.EP_Ravg = ( self.EP[:-1,:] + self.EP[1:,:] ) / 2.0

    # ===== making poisson matrix =====
    def make_poisson_matrix(self):
        #
        self.PM = sc.sparse.dok_matrix((self.RZnodes, self.RZnodes))

        # ext. bias
        self.B = np.zeros(self.RZnodes)

        # finding metal region
        MAT_CH = np.where( self.MAT == 100 )
        MAT_WL = np.where( self.MAT == 110 )

        # ext. bias
        for each_region in [MAT_WL]:
            for each_point in range(len(each_region[0])):
                #
                r_node_cnt, z_node_cnt = each_region[0][each_point], each_region[1][each_point]
                #
                index_r_z = self.Rnodes * (z_node_cnt+0) + (r_node_cnt+0)
                #
                self.B[index_r_z] = 1.0

        # metal region
        for each_region in [MAT_CH, MAT_WL]:
            for each_point in range(len(each_region[0])):
                #
                r_node_cnt, z_node_cnt = each_region[0][each_point], each_region[1][each_point]
                #
                index_r_z = self.Rnodes * (z_node_cnt+0) + (r_node_cnt+0)
                #
                self.PM[index_r_z, index_r_z] = 1.0

        # r boundary
        for r_node_cnt in [0, self.Rnodes-1]:
            for z_node_cnt in range(1,self.Znodes-1):
                #
                index_r_z   = self.Rnodes * (z_node_cnt+0) + (r_node_cnt+0)
                index_rm1_z = self.Rnodes * (z_node_cnt+0) + (r_node_cnt-1)
                index_rp1_z = self.Rnodes * (z_node_cnt+0) + (r_node_cnt+1)
                #
                self.PM[index_r_z, index_r_z  ] = 1.0
                #
                if r_node_cnt == 0:
                    self.PM[index_r_z, index_rp1_z] = -1.0
                elif r_node_cnt == (self.Rnodes-1):
                    self.PM[index_r_z, index_rm1_z] = -1.0

        # z boundary
        for r_node_cnt in range(self.Rnodes):
            for z_node_cnt in [0, self.Znodes-1]:
                #
                index_r_z   = self.Rnodes * (z_node_cnt+0) + (r_node_cnt+0)
                index_r_zm1 = self.Rnodes * (z_node_cnt-1) + (r_node_cnt+0)
                index_r_zp1 = self.Rnodes * (z_node_cnt+1) + (r_node_cnt+0)
                #
                self.PM[index_r_z, index_r_z  ] = 1.0
                #
                if z_node_cnt == 0:
                    self.PM[index_r_z, index_r_zp1] = -1.0
                elif z_node_cnt == (self.Znodes-1):
                    self.PM[index_r_z, index_r_zm1] = -1.0

        # inside boundary
        for r_node_cnt in range(1, self.Rnodes-1):
            for z_node_cnt in range(1, self.Znodes-1):
                #
                index_r_z   = self.Rnodes * (z_node_cnt+0) + (r_node_cnt+0)
                index_rm1_z = self.Rnodes * (z_node_cnt+0) + (r_node_cnt-1)
                index_rp1_z = self.Rnodes * (z_node_cnt+0) + (r_node_cnt+1)
                index_r_zm1 = self.Rnodes * (z_node_cnt-1) + (r_node_cnt+0)
                index_r_zp1 = self.Rnodes * (z_node_cnt+1) + (r_node_cnt+0)
                #
                if self.PM[index_r_z, index_r_z] != 1.0:
                    #
                    A_rm1_z = self.EP_Zavg[r_node_cnt-1, z_node_cnt-1] / \
                              ( self.RZ_dR[r_node_cnt-1, z_node_cnt+0] * self.RZ_dRavg[r_node_cnt-1, z_node_cnt+0] )
                    A_rp1_z = self.EP_Zavg[r_node_cnt+0, z_node_cnt-1] / \
                              ( self.RZ_dR[r_node_cnt+0, z_node_cnt+0] * self.RZ_dRavg[r_node_cnt-1, z_node_cnt+0] )
                    A_r_zm1 = self.EP_Ravg[r_node_cnt-1, z_node_cnt-1] / \
                              ( self.RZ_dZ[r_node_cnt+0, z_node_cnt-1] * self.RZ_dZavg[r_node_cnt+0, z_node_cnt-1] )
                    A_r_zp1 = self.EP_Ravg[r_node_cnt-1, z_node_cnt+0] / \
                              ( self.RZ_dZ[r_node_cnt+0, z_node_cnt+0] * self.RZ_dZavg[r_node_cnt+0, z_node_cnt-1] )
                    #
                    self.PM[index_r_z, index_r_z  ] =  A_rm1_z + A_rp1_z + A_r_zm1 + A_r_zp1
                    self.PM[index_r_z, index_rm1_z] = -A_rm1_z
                    self.PM[index_r_z, index_rp1_z] = -A_rp1_z
                    self.PM[index_r_z, index_r_zm1] = -A_r_zm1
                    self.PM[index_r_z, index_r_zp1] = -A_r_zp1

        # CSR format
        self.PMcsr = self.PM.tocsr()
        self.PMdense = self.PM.todense()

        # solving poission equation
        self.V = sc.sparse.linalg.spsolve(self.PMcsr, self.B)

        #
        self.V2 = np.resize(self.V, (self.Znodes, self.Rnodes)).T
        self.Er = ( self.V2[1:,:] - self.V2[:-1,:] ) / self.RZ_dR * 100
        self.Ez = ( self.V2[:,1:] - self.V2[:,:-1] ) / self.RZ_dZ * 100
        self.E  = np.sqrt( self.Er[:,:-1]**2 + self.Ez[:-1,:]**2 )

        # debugging
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(self.V2)
        ax[1].imshow(self.E)
        plt.show()
                

#
# MAIN
#

poisson_solver = SOLVER()
poisson_solver.make_grid(cd=1300, r_range=[0, 750, 5], z_range=[0, 430, 5])
poisson_solver.make_poisson_matrix()


