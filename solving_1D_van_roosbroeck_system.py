#
# TITLE: solving the van Roosbroeck system numerically
# AUTHOR: Hyunseung Yoo
# PURPOSE: 
# REVISION: 
# REFERENCE: a numerical study of the van Roosbroeck system for semiconductor (SJSU, 2018) 
#


import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


#=============================================================
# CLASS: Heat Equation (2.1) ~ (2.3)
#=============================================================

class HEAT_EQ:

    # constructor
    def __init__(self):
        
        # functions
        self.u   = sp.symbols('u',   cls=sp.Function, real=True, nonzero=True)
        self.u_t = sp.symbols('u_t', cls=sp.Function, real=True, nonzero=True)
        self.u_x = sp.symbols('u_x', cls=sp.Function, real=True, nonzero=True)
        self.f   = sp.symbols('f',   cls=sp.Function, real=True, nonzero=True)
        
        # independnet variables
        self.x = sp.symbols('x', real=True)
        self.t = sp.symbols('t', real=True)
        
        # constants: length
        self.L = sp.symbols('L', real=True, positive=True)
        
        # separation constant
        self.c = sp.symbols('c', real=True, positive=True)

        # Fourier series index
        self.n = sp.symbols('n', real=True, positive=True, integer=True)

        # Fourier series coefficients
        self.Cn = sp.symbols('Cn', real=True)

                
    # (2.1) ~ (2.3)
    def solving_equation(self):
        # separation of variables
        self.u = self.u_t(self.t) * self.u_x(self.x)
        print('Separation of varibles:', self.u)

        # PDE: Heat equation w/ separation of variables
        LHS = self.u.diff(self.t) / self.u
        RHS = self.u.diff(self.x, 2) / self.u
        PDE = sp.Eq(LHS, RHS)
        print('Heat equation (x, t):', PDE)
        print('')

        # === PDE_x
        PDE_x = sp.Eq(PDE.rhs * self.u_x(self.x), -self.c * self.u_x(self.x))
        print('Heat equation (x):', PDE_x)

        # PDE_x general solution
        SOL_x = sp.dsolve(PDE_x)
        print('General solution (x):', SOL_x)

        # boundary conditions: Dirichlet boundary conditions
        BC_x_1 = sp.Eq( SOL_x.rhs.subs(self.x, 0), 0)
        BC_x_2 = sp.Eq( SOL_x.rhs.subs(self.x, self.L), 0)
        print('BC1:', BC_x_1)
        print('BC2:', BC_x_2)

        # finding c
        self.c_cal = (sp.pi * self.n / self.L)**2
        print('calculated separation constant:', self.c_cal)

        # PDE_x base
        SOL_x.rhs.subs(BC_x_1.lhs, BC_x_1.rhs)
        SOL_x_base = SOL_x.rhs.coeff('C1').subs(self.c, self.c_cal)
        print('Base (x):', SOL_x_base)
        print('')

        # === PDE_t
        PDE_t = sp.Eq(PDE.lhs * self.u_t(self.t), -self.c_cal * self.u_t(self.t))
        print('Heat equation (t):', PDE_t)

        # PDE_t general solution
        SOL_t = sp.dsolve(PDE_t)
        print('General solution (t):', SOL_t)

        # PDE_t base
        SOL_t_base = SOL_t.rhs.coeff('C1')
        print('Base (t):', SOL_t_base)
        print('')

        # === PDE x t base
        SOL_x_t_base = SOL_x_base * SOL_t_base
        print('Base (x, t):', SOL_x_t_base)

        # initial conditions
        SOL_x_t_base_IC = SOL_x_t_base.subs(self.t, 0)
        print('Base (x, t=0):', SOL_x_t_base_IC)

        # initial conditions
        SOL_x_t_base_IC_LHS = self.f(self.x) * SOL_x_t_base_IC
        SOL_x_t_base_IC_RHS = SOL_x_t_base_IC**2 * self.Cn
        SOL_x_t_base_IC_EQ = sp.Eq(SOL_x_t_base_IC_LHS, SOL_x_t_base_IC_RHS)
        print('Initial conditions (x, t=0):', SOL_x_t_base_IC_EQ)

        # initial conditions RHS
        SOL_x_t_base_IC_EQ_RHS_integral = sp.integrate(SOL_x_t_base_IC_EQ.rhs, (self.x, 0, self.L))
        SOL_x_t_base_IC_EQ_RHS_integral = SOL_x_t_base_IC_EQ_RHS_integral.subs(self.n, 1)
        print('integrating IC (x, t=0) RHS:', SOL_x_t_base_IC_EQ_RHS_integral)

        # initial conditions LHS
        SOL_x_t_base_IC_EQ_LHS_integral = sp.integrate(SOL_x_t_base_IC_EQ.lhs, (self.x, 0, self.L))
        print('integrating IC (x, t=0) LHS:', SOL_x_t_base_IC_EQ_LHS_integral)
        print('')
        
        # === finding Cn
        SOL_x_t_base_Cn_RHS = SOL_x_t_base_IC_EQ_RHS_integral * 2 / self.L
        SOL_x_t_base_Cn_LHS = SOL_x_t_base_IC_EQ_LHS_integral * 2 / self.L
        SOL_x_t_base_Cn = sp.Eq(SOL_x_t_base_Cn_LHS, SOL_x_t_base_Cn_RHS)
        print('calculated Cn:', SOL_x_t_base_Cn)
        print('')


#=============================================================
# CLASS: Finite Difference Method (2.4) ~ (2.42)
#=============================================================

class FDM:

    # constructor
    def __init__(self):
        
        # independent variables
        self.x  = sp.symbols('x',  real=True)
        self.dx = sp.symbols('dx', real=True)
        self.t  = sp.symbols('t',  real=True)
        self.dt = sp.symbols('dt', real=True)

        # dependent variables (explicit scheme)
        self.f_im1_j = sp.symbols('f_{i-1}^{j}', real=True)
        self.f_i_j   = sp.symbols('f_{i}^{j}',   real=True)
        self.f_ip1_j = sp.symbols('f_{i+1}^{j}', real=True)

        # dependent variables (implicit scheme)
        self.f_im1_jp1 = sp.symbols('f_{i-1}^{j+1}', real=True)
        self.f_i_jp1   = sp.symbols('f_{i}^{j+1}',   real=True)
        self.f_ip1_jp1 = sp.symbols('f_{i+1}^{j+1}', real=True)
        
        # constant
        self.R = sp.symbols('R', real=True)

        # stability, convergence, consistency
        self.xi = sp.symbols('xi', real=True)
        self.beta, self.h = sp.symbols('beta h', real=True)
        self.p, self.q = sp.symbols('p q', real=True)

        # drift-diffusion model
        self.D, self.v = sp.symbols('D v', real=True)
        self.R1, self.R2 = sp.symbols('R1 R2', real=True)
        

    # finite difference method: (2.4) ~ (2.15)
    def finite_difference_method(self):
        # === chapter section: 2-1
        print('chapter - section [ 2-1 ]')
        print('')
        
        #
        self.f_i_diff1_fw_j = (self.f_ip1_j - self.f_i_j)   / ( (self.x + self.dx) - self.x )
        self.f_i_diff1_bw_j = (self.f_i_j - self.f_im1_j)   / ( self.x - (self.x - self.dx) )
        self.f_i_diff1_ct_j = (self.f_ip1_j - self.f_im1_j) / ( (self.x + self.dx) - (self.x - self.dx) )

        #
        self.f_i_diff2_ct_j = (self.f_i_diff1_fw_j - self.f_i_diff1_bw_j) / ( (self.x + self.dx/2) - (self.x - self.dx/2) )
        self.f_i_diff2_ct_j = self.f_i_diff2_ct_j.expand().simplify()
        
        #
        self.f_i_j_diff1_fw = (self.f_i_jp1 - self.f_i_j)   / ( (self.t + self.dt) - self.t )

        #
        print('forward  1st difference formula (ex):', self.f_i_diff1_fw_j)
        print('backward 1st difference formula (ex):', self.f_i_diff1_bw_j)
        print('central  1st difference formula (ex):', self.f_i_diff1_ct_j)
        print('central  2nd difference formula (ex):', self.f_i_diff2_ct_j)
        print('forward  1st difference formula (j):',  self.f_i_j_diff1_fw)

        #
        self.heat_eq_ex_LHS = self.f_i_j_diff1_fw * self.dt + self.f_i_j
        self.heat_eq_ex_RHS = self.f_i_diff2_ct_j * self.dt + self.f_i_j
        self.heat_eq_ex_RHS = self.heat_eq_ex_RHS.expand().subs(self.dt / self.dx**2, self.R).simplify()
        self.heat_eq_ex_RHS_im1 = self.heat_eq_ex_RHS.coeff(self.f_im1_j)
        self.heat_eq_ex_RHS_i   = self.heat_eq_ex_RHS.coeff(self.f_i_j)
        self.heat_eq_ex_RHS_ip1 = self.heat_eq_ex_RHS.coeff(self.f_ip1_j)
        self.heat_eq_ex = sp.Eq(self.heat_eq_ex_LHS, self.heat_eq_ex_RHS)

        #
        print('Heat equation (ex):', self.heat_eq_ex)
        print(' ROW > [', self.heat_eq_ex_RHS_im1, ',', self.heat_eq_ex_RHS_i, ',', self.heat_eq_ex_RHS_ip1, ']')
        print('')

        #
        self.f_i_diff1_fw_jp1 = (self.f_ip1_jp1 - self.f_i_jp1)   / ( (self.x + self.dx) - self.x )
        self.f_i_diff1_bw_jp1 = (self.f_i_jp1 - self.f_im1_jp1)   / ( self.x - (self.x - self.dx) )
        self.f_i_diff1_ct_jp1 = (self.f_ip1_jp1 - self.f_im1_jp1) / ( (self.x + self.dx) - (self.x - self.dx) )

        #
        self.f_i_diff2_ct_jp1 = (self.f_i_diff1_fw_jp1 - self.f_i_diff1_bw_jp1) / ( (self.x + self.dx/2) - (self.x - self.dx/2) )
        self.f_i_diff2_ct_jp1 = self.f_i_diff2_ct_jp1.expand().simplify()

        #
        print('forward  1st difference formula (im):', self.f_i_diff1_fw_jp1)
        print('backward 1st difference formula (im):', self.f_i_diff1_bw_jp1)
        print('central  1st difference formula (im):', self.f_i_diff1_ct_jp1)
        print('central  2nd difference formula (im):', self.f_i_diff2_ct_jp1)
        print('forward  1st difference formula (j):',  self.f_i_j_diff1_fw)

        #
        self.heat_eq_im_LHS = -self.f_i_j_diff1_fw * self.dt + self.f_i_jp1
        self.heat_eq_im_RHS = -self.f_i_diff2_ct_jp1 * self.dt + self.f_i_jp1
        self.heat_eq_im_RHS = self.heat_eq_im_RHS.expand().subs(self.dt / self.dx**2, self.R).simplify()
        self.heat_eq_im_RHS_im1 = self.heat_eq_im_RHS.coeff(self.f_im1_jp1)
        self.heat_eq_im_RHS_i   = self.heat_eq_im_RHS.coeff(self.f_i_jp1)
        self.heat_eq_im_RHS_ip1 = self.heat_eq_im_RHS.coeff(self.f_ip1_jp1)
        self.heat_eq_im = sp.Eq(self.heat_eq_im_LHS, self.heat_eq_im_RHS)

        #
        print('Heat equation (im):', self.heat_eq_im)
        print(' ROW > [', self.heat_eq_im_RHS_im1, ',', self.heat_eq_im_RHS_i, ',', self.heat_eq_im_RHS_ip1, ']')
        print('')
        

    # stability, convergence, consistency: (2.16) ~ (2.26)
    def stability_convergence_consistency(self):
        # === chapter section: 2-2
        print('chapter - section [ 2-2 ]')
        print('')
        
        # FD scheme is stable if
        #   the computed solution remain finite |u_i^j| < oo (xi <= 1)
        #   and does not oscillate unnecessarily as dx, dt -> 0.

        # von Neumann stability analysis: the solution is a finite Fourier series
        self.f_i_jp1_sta   = self.xi**(self.q+1)*sp.exp(sp.I*self.beta*self.p*self.h)
        self.f_i_j_sta     = self.xi**(self.q)*sp.exp(sp.I*self.beta*self.p*self.h)
        self.f_im1_j_sta   = self.xi**(self.q)*sp.exp(sp.I*self.beta*(self.p-1)*self.h)
        self.f_ip1_j_sta   = self.xi**(self.q)*sp.exp(sp.I*self.beta*(self.p+1)*self.h)
        self.f_im1_jp1_sta = self.xi**(self.q+1)*sp.exp(sp.I*self.beta*(self.p-1)*self.h)
        self.f_ip1_jp1_sta = self.xi**(self.q+1)*sp.exp(sp.I*self.beta*(self.p+1)*self.h)

        # Heat equation explicit scheme
        self.heat_eq_ex_stability = self.heat_eq_ex.subs(self.f_i_jp1, self.f_i_jp1_sta)
        self.heat_eq_ex_stability = self.heat_eq_ex_stability.subs(self.f_i_j, self.f_i_j_sta)
        self.heat_eq_ex_stability = self.heat_eq_ex_stability.subs(self.f_im1_j, self.f_im1_j_sta)
        self.heat_eq_ex_stability = self.heat_eq_ex_stability.subs(self.f_ip1_j, self.f_ip1_j_sta)
        print('Heat equation (ex):', self.heat_eq_ex)
        print('Heat equation (ex) stability:', self.heat_eq_ex_stability)

        # stability check LHS
        self.heat_eq_ex_stability_LHS = self.heat_eq_ex_stability.lhs / self.xi**(self.q) / sp.exp(sp.I*self.beta*self.p*self.h)
        self.heat_eq_ex_stability_LHS = self.heat_eq_ex_stability_LHS.simplify()
        print('LHS:', self.heat_eq_ex_stability_LHS)

        # stability check RHS
        self.heat_eq_ex_stability_RHS = self.heat_eq_ex_stability.rhs / self.xi**(self.q) / sp.exp(sp.I*self.beta*self.p*self.h)
        self.heat_eq_ex_stability_RHS = self.heat_eq_ex_stability_RHS.simplify().rewrite(sp.cos).simplify()
        print('RHS:', self.heat_eq_ex_stability_RHS)

        # stability check equation, check R range
        self.heat_eq_ex_stability_EQ = sp.Eq(self.heat_eq_ex_stability_LHS, self.heat_eq_ex_stability_RHS)
        print('Stability check:', self.heat_eq_ex_stability_EQ)
        print('')

        # Heat equation explicit scheme
        self.heat_eq_im_stability = self.heat_eq_im.subs(self.f_i_jp1, self.f_i_jp1_sta)
        self.heat_eq_im_stability = self.heat_eq_im_stability.subs(self.f_i_j, self.f_i_j_sta)
        self.heat_eq_im_stability = self.heat_eq_im_stability.subs(self.f_im1_jp1, self.f_im1_jp1_sta)
        self.heat_eq_im_stability = self.heat_eq_im_stability.subs(self.f_ip1_jp1, self.f_ip1_jp1_sta)
        print('Heat equation (im):', self.heat_eq_im)
        print('Heat equation (im) stability:', self.heat_eq_im_stability)

        # stability check LHS
        self.heat_eq_im_stability_LHS = self.heat_eq_im_stability.lhs / self.xi**(self.q) / sp.exp(sp.I*self.beta*self.p*self.h)
        self.heat_eq_im_stability_LHS = self.heat_eq_im_stability_LHS.simplify()
        print('LHS:', self.heat_eq_im_stability_LHS)

        # stability check RHS
        self.heat_eq_im_stability_RHS = self.heat_eq_im_stability.rhs / self.xi**(self.q) / sp.exp(sp.I*self.beta*self.p*self.h)
        self.heat_eq_im_stability_RHS = self.heat_eq_im_stability_RHS.simplify().rewrite(sp.cos).simplify()
        print('RHS:', self.heat_eq_im_stability_RHS)

        # stability check equation, check R range
        self.heat_eq_im_stability_EQ = sp.Eq(self.heat_eq_im_stability_LHS, self.heat_eq_im_stability_RHS)
        print('Stability check:', self.heat_eq_im_stability_EQ)
        print('')


    # drift diffusion model: (2.27) ~ (2.33)
    def drift_diffusion_model(self):
        # === chapter section: 2-3
        print('chapter - section [ 2-3 ]')
        print('')
        
        # === FTFS time evolution 1
        self.ddm_im_fdm_ftfs_LHS = (self.f_i_jp1 - self.f_i_j)   / ( (self.t + self.dt) - self.t )
        print('DDM FTFS LHS:', self.ddm_im_fdm_ftfs_LHS)
        
        # FTFS diffusion current 1
        self.f_i_diff1_fw_jp1 = (self.f_ip1_jp1 - self.f_i_jp1)   / ( (self.x + self.dx) - self.x )
        self.f_i_diff1_bw_jp1 = (self.f_i_jp1 - self.f_im1_jp1)   / ( (self.x + self.dx) - self.x )
        self.f_i_diff2_ct_jp1 = (self.f_i_diff1_fw_jp1 - self.f_i_diff1_bw_jp1) / ( (self.x + self.dx/2) - (self.x - self.dx/2) )
        self.ddm_im_fdm_ftfs_RHS_diffusion = self.D * self.f_i_diff2_ct_jp1.expand().simplify()
        print('DDM FTFS RHS diffusion:', self.ddm_im_fdm_ftfs_RHS_diffusion)
        
        # FTFS drift current 1
        self.ddm_im_fdm_ftfs_RHS_drift = -self.v * self.f_i_diff1_fw_jp1
        print('DDM FTFS RHS drift:', self.ddm_im_fdm_ftfs_RHS_drift)
        print('')

        # === FTFS time evolution 2
        self.ddm_im_fdm_ftfs_LHS = -self.ddm_im_fdm_ftfs_LHS * self.dt
        print('DDM FTFS LHS:', self.ddm_im_fdm_ftfs_LHS)
        
        # FTFS diffusion current 2
        self.ddm_im_fdm_ftfs_RHS_diffusion = -self.ddm_im_fdm_ftfs_RHS_diffusion * self.dt
        self.ddm_im_fdm_ftfs_RHS_diffusion = self.ddm_im_fdm_ftfs_RHS_diffusion.expand()
        self.ddm_im_fdm_ftfs_RHS_diffusion = self.ddm_im_fdm_ftfs_RHS_diffusion.subs(self.dt/self.dx**2, self.R2)
        self.ddm_im_fdm_ftfs_RHS_diffusion = self.ddm_im_fdm_ftfs_RHS_diffusion.simplify()
        print('DDM FTFS RHS diffusion:', self.ddm_im_fdm_ftfs_RHS_diffusion)
        
        # FTFS drift current 2
        self.ddm_im_fdm_ftfs_RHS_drift = -self.ddm_im_fdm_ftfs_RHS_drift * self.dt
        self.ddm_im_fdm_ftfs_RHS_drift = self.ddm_im_fdm_ftfs_RHS_drift.expand()
        self.ddm_im_fdm_ftfs_RHS_drift = self.ddm_im_fdm_ftfs_RHS_drift.subs(self.dt/self.dx, self.R1)
        self.ddm_im_fdm_ftfs_RHS_drift = self.ddm_im_fdm_ftfs_RHS_drift.simplify()
        print('DDM FTFS RHS drift:', self.ddm_im_fdm_ftfs_RHS_drift)
        print('')

        # === FTFS time evolution 3
        self.ddm_im_fdm_ftfs_LHS = self.ddm_im_fdm_ftfs_LHS + self.f_i_jp1
        print('DDM FTFS LHS:', self.ddm_im_fdm_ftfs_LHS)
        
        # FTFS diffusion current 3
        self.ddm_im_fdm_ftfs_RHS_diffusion = self.ddm_im_fdm_ftfs_RHS_diffusion + self.f_i_jp1
        self.ddm_im_fdm_ftfs_RHS_diffusion = self.ddm_im_fdm_ftfs_RHS_diffusion.expand()
        print('DDM FTFS RHS diffusion:', self.ddm_im_fdm_ftfs_RHS_diffusion)
        
        # FTFS drift current 3
        self.ddm_im_fdm_ftfs_RHS_drift = self.ddm_im_fdm_ftfs_RHS_drift
        self.ddm_im_fdm_ftfs_RHS_drift = self.ddm_im_fdm_ftfs_RHS_drift.expand()
        
        print('DDM FTFS RHS drift:', self.ddm_im_fdm_ftfs_RHS_drift)
        print('')

        # === FTFS FDM equation
        self.ddm_im_fdm_ftfs_EQ = sp.Eq(self.ddm_im_fdm_ftfs_LHS, self.ddm_im_fdm_ftfs_RHS_diffusion + self.ddm_im_fdm_ftfs_RHS_drift)
        self.ddm_im_fdm_ftfs_EQ_RHS_f_ip1_jp1_coeff = self.ddm_im_fdm_ftfs_EQ.rhs.coeff(self.f_ip1_jp1)
        self.ddm_im_fdm_ftfs_EQ_RHS_f_i_jp1_coeff = self.ddm_im_fdm_ftfs_EQ.rhs.coeff(self.f_i_jp1)
        self.ddm_im_fdm_ftfs_EQ_RHS_f_im1_jp1_coeff = self.ddm_im_fdm_ftfs_EQ.rhs.coeff(self.f_im1_jp1)
        print('DDM FTFS EQ:', self.ddm_im_fdm_ftfs_EQ)
        print('DDM FTFS RHS coeff of ip1_jp1 (lambda2):', self.ddm_im_fdm_ftfs_EQ_RHS_f_ip1_jp1_coeff)
        print('DDM FTFS RHS coeff of i_jp1 (lambda1)  :', self.ddm_im_fdm_ftfs_EQ_RHS_f_i_jp1_coeff)
        print('DDM FTFS RHS coeff of im1_jp1 (lambda3):', self.ddm_im_fdm_ftfs_EQ_RHS_f_im1_jp1_coeff)
        print('')

        # === FTBS time evolution 1
        self.ddm_im_fdm_ftbs_LHS = (self.f_i_jp1 - self.f_i_j)   / ( (self.t + self.dt) - self.t )
        print('DDM FTBS LHS:', self.ddm_im_fdm_ftbs_LHS)
        
        # FTBS diffusion current 1
        self.f_i_diff1_fw_jp1 = (self.f_ip1_jp1 - self.f_i_jp1)   / ( (self.x + self.dx) - self.x )
        self.f_i_diff1_bw_jp1 = (self.f_i_jp1 - self.f_im1_jp1)   / ( (self.x + self.dx) - self.x )
        self.f_i_diff2_ct_jp1 = (self.f_i_diff1_fw_jp1 - self.f_i_diff1_bw_jp1) / ( (self.x + self.dx/2) - (self.x - self.dx/2) )
        self.ddm_im_fdm_ftbs_RHS_diffusion = self.D * self.f_i_diff2_ct_jp1.expand().simplify()
        print('DDM FTBS RHS diffusion:', self.ddm_im_fdm_ftbs_RHS_diffusion)
        
        # FTBS drift current 1-1
        self.ddm_im_fdm_ftbs_RHS_drift = -self.v * self.f_i_diff1_bw_jp1
        print('DDM FTBS RHS drift:', self.ddm_im_fdm_ftbs_RHS_drift)
        print('')

        # === FTBS time evolution 2
        self.ddm_im_fdm_ftbs_LHS = -self.ddm_im_fdm_ftbs_LHS * self.dt
        print('DDM FTBS LHS:', self.ddm_im_fdm_ftbs_LHS)
        
        # FTBS diffusion current 2
        self.ddm_im_fdm_ftbs_RHS_diffusion = -self.ddm_im_fdm_ftbs_RHS_diffusion * self.dt
        self.ddm_im_fdm_ftbs_RHS_diffusion = self.ddm_im_fdm_ftbs_RHS_diffusion.expand()
        self.ddm_im_fdm_ftbs_RHS_diffusion = self.ddm_im_fdm_ftbs_RHS_diffusion.subs(self.dt/self.dx**2, self.R2)
        self.ddm_im_fdm_ftbs_RHS_diffusion = self.ddm_im_fdm_ftbs_RHS_diffusion.simplify()
        print('DDM FTBS RHS diffusion:', self.ddm_im_fdm_ftbs_RHS_diffusion)
        
        # FTBS drift current 2
        self.ddm_im_fdm_ftbs_RHS_drift = -self.ddm_im_fdm_ftbs_RHS_drift * self.dt
        self.ddm_im_fdm_ftbs_RHS_drift = self.ddm_im_fdm_ftbs_RHS_drift.expand()
        self.ddm_im_fdm_ftbs_RHS_drift = self.ddm_im_fdm_ftbs_RHS_drift.subs(self.dt/self.dx, self.R1)
        self.ddm_im_fdm_ftbs_RHS_drift = self.ddm_im_fdm_ftbs_RHS_drift.simplify()
        print('DDM FTBS RHS drift:', self.ddm_im_fdm_ftbs_RHS_drift)
        print('')

        # === FTBS time evolution 3
        self.ddm_im_fdm_ftbs_LHS = self.ddm_im_fdm_ftbs_LHS + self.f_i_jp1
        print('DDM FTBS LHS:', self.ddm_im_fdm_ftbs_LHS)
        
        # FTBS diffusion current 3
        self.ddm_im_fdm_ftbs_RHS_diffusion = self.ddm_im_fdm_ftbs_RHS_diffusion + self.f_i_jp1
        self.ddm_im_fdm_ftbs_RHS_diffusion = self.ddm_im_fdm_ftbs_RHS_diffusion.expand()
        print('DDM FTBS RHS diffusion:', self.ddm_im_fdm_ftbs_RHS_diffusion)
        
        # FTBS drift current 3
        self.ddm_im_fdm_ftbs_RHS_drift = self.ddm_im_fdm_ftbs_RHS_drift
        self.ddm_im_fdm_ftbs_RHS_drift = self.ddm_im_fdm_ftbs_RHS_drift.expand()
        
        print('DDM FTBS RHS drift:', self.ddm_im_fdm_ftbs_RHS_drift)
        print('')

        # === FTBS FDM equation
        self.ddm_im_fdm_ftbs_EQ = sp.Eq(self.ddm_im_fdm_ftbs_LHS, self.ddm_im_fdm_ftbs_RHS_diffusion + self.ddm_im_fdm_ftbs_RHS_drift)
        self.ddm_im_fdm_ftbs_EQ_RHS_f_ip1_jp1_coeff = self.ddm_im_fdm_ftbs_EQ.rhs.coeff(self.f_ip1_jp1)
        self.ddm_im_fdm_ftbs_EQ_RHS_f_i_jp1_coeff = self.ddm_im_fdm_ftbs_EQ.rhs.coeff(self.f_i_jp1)
        self.ddm_im_fdm_ftbs_EQ_RHS_f_im1_jp1_coeff = self.ddm_im_fdm_ftbs_EQ.rhs.coeff(self.f_im1_jp1)
        print('DDM FTBS EQ:', self.ddm_im_fdm_ftbs_EQ)
        print('DDM FTBS RHS coeff of ip1_jp1 (lambda2):', self.ddm_im_fdm_ftbs_EQ_RHS_f_ip1_jp1_coeff)
        print('DDM FTBS RHS coeff of i_jp1 (lambda1)  :', self.ddm_im_fdm_ftbs_EQ_RHS_f_i_jp1_coeff)
        print('DDM FTBS RHS coeff of im1_jp1 (lambda3):', self.ddm_im_fdm_ftbs_EQ_RHS_f_im1_jp1_coeff)
        print('')

        # === implicit drift-diffusion equation solver: FTBS with PBC

        ftbs_dt = 0.1
        ftbs_dx = 0.1
        ftbs_x_div = 101
        ftbs_t_step = 200

        ftbs_R2 = ftbs_dt / ftbs_dx**2
        ftbs_R1 = ftbs_dt / ftbs_dx

        ftbs_D = 0.001
        ftbs_v = -2.0

        ftbs_x = np.linspace(0.0, ftbs_dx*(ftbs_x_div-1), ftbs_x_div)

        ftbs_u_ics = np.exp( -(ftbs_x - 5.0)**2 )

        ftbs_A = np.zeros([ftbs_x_div, ftbs_x_div], dtype=float)
        ftbs_row = np.zeros(3, dtype=float)
        ftbs_row[0] = -ftbs_D * ftbs_R2 - ftbs_v * ftbs_R1
        ftbs_row[1] = 2.0 * ftbs_D * ftbs_R2 + ftbs_v * ftbs_R1 + 1.0
        ftbs_row[2] = -ftbs_D * ftbs_R2

        ftfs_A = np.zeros([ftbs_x_div, ftbs_x_div], dtype=float)
        ftfs_row = np.zeros(3, dtype=float)
        ftfs_row[0] = -ftbs_D * ftbs_R2
        ftfs_row[1] = 2.0 * ftbs_D * ftbs_R2 - ftbs_v * ftbs_R1 + 1.0
        ftfs_row[2] = -ftbs_D * ftbs_R2 + ftbs_v * ftbs_R1
        
        for row_cnt in range(ftbs_x_div):
            if row_cnt == 0:
                ftbs_A[row_cnt, 0:2] = ftbs_row[1:]
                ftbs_A[row_cnt, (row_cnt-1)] = ftbs_row[0]              # PBC
                ftfs_A[row_cnt, 0:2] = ftfs_row[1:]
                ftfs_A[row_cnt, (row_cnt-1)] = ftfs_row[0]              # PBC
            elif row_cnt == (ftbs_x_div-1):
                ftbs_A[row_cnt, (row_cnt-1):] = ftbs_row[:2]
                ftbs_A[row_cnt, 0] = ftbs_row[2]                        # PBC
                ftfs_A[row_cnt, (row_cnt-1):] = ftfs_row[:2]
                ftfs_A[row_cnt, 0] = ftfs_row[2]                        # PBC
            else:
                ftbs_A[row_cnt, (row_cnt-1):(row_cnt+2)] = ftbs_row
                ftfs_A[row_cnt, (row_cnt-1):(row_cnt+2)] = ftfs_row

        print(ftbs_A)       # FTBS A debugging
        print(ftbs_row)     # FTBS row debugging
        print(ftfs_A)       # FTFS A debugging
        print(ftfs_row)     # FTFS row debugging
        print('')

        ftbs_u1 = np.linalg.solve(ftbs_A, ftbs_u_ics)       # first solution
        ftfs_u1 = np.linalg.solve(ftfs_A, ftbs_u_ics)       # first solution
        
        for t_cnt in range(ftbs_t_step):
            if t_cnt == 0:                                  # second solution
                ftbs_u = np.linalg.solve(ftbs_A, ftbs_u1)
                ftfs_u = np.linalg.solve(ftfs_A, ftfs_u1)
            else:                                           # others
                ftbs_u = np.linalg.solve(ftbs_A, ftbs_u)
                ftfs_u = np.linalg.solve(ftfs_A, ftfs_u)
        
        # visualization
        fig, ax = plt.subplots(1, 1)
        ax.plot(ftbs_x, ftbs_u_ics, 'o-')
        ax.plot(ftbs_x, ftbs_u1, 'o-')
        ax.plot(ftbs_x, ftbs_u, 'o-')
        ax.plot(ftbs_x, ftfs_u1, '-')
        ax.plot(ftbs_x, ftfs_u, '-')
        ax.grid(ls=':')
        plt.show()
        plt.close()


    # Crank-Nicholson nethod: (2.34)
    def crank_nicholson_method(self):
        # === chapter section: 2-4
        print('chapter - section [ 2-4 ]')
        print('')

        # explicit scheme -> implicit scheme
        # implicit scheme -> FTFS method, FTBS method

        # Crank Nicholson finite difference method (1947)
        # unconditionally stable, convergent, small truncation error
        # makes use of the midpoint between two time steps
        
        # === CN method
        self.ddm_im_cnm_LHS =  (self.f_i_jp1 - self.f_i_j)   / ( (self.t + self.dt) - self.t )
        self.ddm_im_cnm_LHS =  self.ddm_im_cnm_LHS * self.dt
        print('DDM CNM LHS:', self.ddm_im_cnm_LHS)

        # finite difference method
        self.f_i_diff1_fw_j   = (self.f_ip1_j - self.f_i_j)   / ( (self.x + self.dx) - self.x )
        self.f_i_diff1_bw_j   = (self.f_i_j - self.f_im1_j)   / ( (self.x + self.dx) - self.x )
        self.f_i_diff2_ct_j   = (self.f_i_diff1_fw_j - self.f_i_diff1_bw_j) / ( (self.x + self.dx/2) - (self.x - self.dx/2) )
        self.f_i_diff1_fw_jp1 = (self.f_ip1_jp1 - self.f_i_jp1)   / ( (self.x + self.dx) - self.x )
        self.f_i_diff1_bw_jp1 = (self.f_i_jp1 - self.f_im1_jp1)   / ( (self.x + self.dx) - self.x )
        self.f_i_diff2_ct_jp1 = (self.f_i_diff1_fw_jp1 - self.f_i_diff1_bw_jp1) / ( (self.x + self.dx/2) - (self.x - self.dx/2) )

        # CN method diffusion current
        self.ddm_im_cnm_RHS_diffusion = self.D * self.f_i_diff2_ct_jp1 / 2 + self.D * self.f_i_diff2_ct_j / 2
        self.ddm_im_cnm_RHS_diffusion = self.ddm_im_cnm_RHS_diffusion * self.dt
        self.ddm_im_cnm_RHS_diffusion = self.ddm_im_cnm_RHS_diffusion.expand()
        self.ddm_im_cnm_RHS_diffusion = self.ddm_im_cnm_RHS_diffusion.subs(self.dt/self.dx**2, self.R2)
        print('DDM CNM RHS diffusion:', self.ddm_im_cnm_RHS_diffusion)

        # CN method drift current
        self.ddm_im_cnm_RHS_drift = -self.v * self.f_i_diff1_fw_jp1 / 2 - self.v * self.f_i_diff1_fw_j / 2
        self.ddm_im_cnm_RHS_drift = self.ddm_im_cnm_RHS_drift * self.dt
        self.ddm_im_cnm_RHS_drift = self.ddm_im_cnm_RHS_drift.expand()
        self.ddm_im_cnm_RHS_drift = self.ddm_im_cnm_RHS_drift.subs(self.dt/self.dx, self.R1)
        print('DDM CNM RHS drift:', self.ddm_im_cnm_RHS_drift)
        print('')

        # CN FDM equation 1
        self.ddm_im_cnm_EQ = sp.Eq(self.ddm_im_cnm_LHS, self.ddm_im_cnm_RHS_diffusion + self.ddm_im_cnm_RHS_drift)
        print('DDM CNM EQ:', self.ddm_im_cnm_EQ)
        print('')

        # CN FDM equation 2
        self.ddm_im_cnm_EQ_LHS = self.ddm_im_cnm_EQ.lhs + self.f_i_j
        self.ddm_im_cnm_EQ_RHS = self.ddm_im_cnm_EQ.rhs + self.f_i_j
        self.ddm_im_cnm_EQ_LHS = self.ddm_im_cnm_EQ_LHS - self.D * self.R2 * self.f_ip1_jp1 / 2
        self.ddm_im_cnm_EQ_RHS = self.ddm_im_cnm_EQ_RHS - self.D * self.R2 * self.f_ip1_jp1 / 2
        self.ddm_im_cnm_EQ_LHS = self.ddm_im_cnm_EQ_LHS - self.D * self.R2 * self.f_im1_jp1 / 2
        self.ddm_im_cnm_EQ_RHS = self.ddm_im_cnm_EQ_RHS - self.D * self.R2 * self.f_im1_jp1 / 2
        self.ddm_im_cnm_EQ_LHS = self.ddm_im_cnm_EQ_LHS + self.D * self.R2 * self.f_i_jp1
        self.ddm_im_cnm_EQ_RHS = self.ddm_im_cnm_EQ_RHS + self.D * self.R2 * self.f_i_jp1
        self.ddm_im_cnm_EQ_LHS = self.ddm_im_cnm_EQ_LHS + self.v * self.R1 * self.f_ip1_jp1 / 2
        self.ddm_im_cnm_EQ_RHS = self.ddm_im_cnm_EQ_RHS + self.v * self.R1 * self.f_ip1_jp1 / 2
        self.ddm_im_cnm_EQ_LHS = self.ddm_im_cnm_EQ_LHS - self.v * self.R1 * self.f_i_jp1 / 2
        self.ddm_im_cnm_EQ_RHS = self.ddm_im_cnm_EQ_RHS - self.v * self.R1 * self.f_i_jp1 / 2
        print('DDM CNM EQ LHS:', self.ddm_im_cnm_EQ_LHS)
        print('DDM CNM EQ RHS:', self.ddm_im_cnm_EQ_RHS)
        print('')

        # CN FDM equation 2-1: matrix A
        self.ddm_im_cnm_EQ_LHS_f_ip1_jp1_coeff = self.ddm_im_cnm_EQ_LHS.coeff(self.f_ip1_jp1)
        self.ddm_im_cnm_EQ_LHS_f_i_jp1_coeff   = self.ddm_im_cnm_EQ_LHS.coeff(self.f_i_jp1)
        self.ddm_im_cnm_EQ_LHS_f_im1_jp1_coeff = self.ddm_im_cnm_EQ_LHS.coeff(self.f_im1_jp1)
        print('DDM CNM EQ LHS coeff of f_ip1_jp1 (lambda2):', self.ddm_im_cnm_EQ_LHS_f_ip1_jp1_coeff)
        print('DDM CNM EQ LHS coeff of f_i_jp1   (lambda1):', self.ddm_im_cnm_EQ_LHS_f_i_jp1_coeff)
        print('DDM CNM EQ LHS coeff of f_im1_jp1 (lambda3):', self.ddm_im_cnm_EQ_LHS_f_im1_jp1_coeff)

        # CN FDM equation 2-2: matrix B
        self.ddm_im_cnm_EQ_RHS_f_ip1_j_coeff   = self.ddm_im_cnm_EQ_RHS.coeff(self.f_ip1_j)
        self.ddm_im_cnm_EQ_RHS_f_i_j_coeff     = self.ddm_im_cnm_EQ_RHS.coeff(self.f_i_j)
        self.ddm_im_cnm_EQ_RHS_f_im1_j_coeff   = self.ddm_im_cnm_EQ_RHS.coeff(self.f_im1_j)
        print('DDM CNM EQ RHS coeff of f_ip1_j   (lambda2):', self.ddm_im_cnm_EQ_RHS_f_ip1_j_coeff)
        print('DDM CNM EQ RHS coeff of f_i_j     (lambda1):', self.ddm_im_cnm_EQ_RHS_f_i_j_coeff)
        print('DDM CNM EQ RHS coeff of f_im1_j   (lambda3):', self.ddm_im_cnm_EQ_RHS_f_im1_j_coeff)

        # === implicit drift-diffusion equation solver: CN FDM with PBC

        cnm_ftfs_dt = 0.1
        cnm_ftfs_dx = 0.1
        cnm_ftfs_x_div = 101
        cnm_ftfs_t_step = 200

        cnm_ftfs_R2 = cnm_ftfs_dt / cnm_ftfs_dx**2
        cnm_ftfs_R1 = cnm_ftfs_dt / cnm_ftfs_dx

        cnm_ftfs_D = 0.001
        cnm_ftfs_v = -2.0

        cnm_ftfs_x = np.linspace(0.0, cnm_ftfs_dx*(cnm_ftfs_x_div-1), cnm_ftfs_x_div)

        cnm_ftfs_u_ics = np.exp( -(cnm_ftfs_x - 5.0)**2 )

        cnm_ftfs_A = np.zeros([cnm_ftfs_x_div, cnm_ftfs_x_div], dtype=float)
        cnm_ftfs_A_row = np.zeros(3, dtype=float)
        cnm_ftfs_A_row[0] = -cnm_ftfs_D * cnm_ftfs_R2 / 2
        cnm_ftfs_A_row[1] = cnm_ftfs_D * cnm_ftfs_R2 - cnm_ftfs_v * cnm_ftfs_R1 / 2 + 1.0
        cnm_ftfs_A_row[2] = -cnm_ftfs_D * cnm_ftfs_R2 / 2 + cnm_ftfs_v * cnm_ftfs_R1 / 2

        cnm_ftfs_B = np.zeros([cnm_ftfs_x_div, cnm_ftfs_x_div], dtype=float)
        cnm_ftfs_B_row = np.zeros(3, dtype=float)
        cnm_ftfs_B_row[0] = cnm_ftfs_D * cnm_ftfs_R2 / 2
        cnm_ftfs_B_row[1] = -cnm_ftfs_D * cnm_ftfs_R2 + cnm_ftfs_v * cnm_ftfs_R1 / 2 + 1.0
        cnm_ftfs_B_row[2] = cnm_ftfs_D * cnm_ftfs_R2 / 2 - cnm_ftfs_v * cnm_ftfs_R1 / 2

        for row_cnt in range(cnm_ftfs_x_div):
            if row_cnt == 0:
                cnm_ftfs_A[row_cnt, 0:2] = cnm_ftfs_A_row[1:]
                cnm_ftfs_A[row_cnt, (row_cnt-1)] = cnm_ftfs_A_row[0]              # PBC
                cnm_ftfs_B[row_cnt, 0:2] = cnm_ftfs_B_row[1:]
                cnm_ftfs_B[row_cnt, (row_cnt-1)] = cnm_ftfs_B_row[0]              # PBC
            elif row_cnt == (cnm_ftfs_x_div-1):
                cnm_ftfs_A[row_cnt, (row_cnt-1):] = cnm_ftfs_A_row[:2]
                cnm_ftfs_A[row_cnt, 0] = cnm_ftfs_A_row[2]                        # PBC
                cnm_ftfs_B[row_cnt, (row_cnt-1):] = cnm_ftfs_B_row[:2]
                cnm_ftfs_B[row_cnt, 0] = cnm_ftfs_B_row[2]                        # PBC
            else:
                cnm_ftfs_A[row_cnt, (row_cnt-1):(row_cnt+2)] = cnm_ftfs_A_row
                cnm_ftfs_B[row_cnt, (row_cnt-1):(row_cnt+2)] = cnm_ftfs_B_row

        print(cnm_ftfs_A)           # CNM FTFS A debugging
        print(cnm_ftfs_A_row)       # CNM FTFS A row debugging
        print(cnm_ftfs_B)           # CNM FTFS B debugging
        print(cnm_ftfs_B_row)       # CNM FTFS B row debugging
        print('')

        cnm_ftfs_u1 = np.linalg.solve(cnm_ftfs_A, np.matmul(cnm_ftfs_B, cnm_ftfs_u_ics))       # first solution

        for t_cnt in range(cnm_ftfs_t_step):
            if t_cnt == 0:                                  # second solution
                cnm_ftfs_u = np.linalg.solve(cnm_ftfs_A, np.matmul(cnm_ftfs_B, cnm_ftfs_u1))
            else:                                           # others
                cnm_ftfs_u = np.linalg.solve(cnm_ftfs_A, np.matmul(cnm_ftfs_B, cnm_ftfs_u))

        # visualization
        fig, ax = plt.subplots(1, 1)
        ax.plot(cnm_ftfs_x, cnm_ftfs_u_ics, 'o-')
        ax.plot(cnm_ftfs_x, cnm_ftfs_u1, 'o-')
        ax.plot(cnm_ftfs_x, cnm_ftfs_u, 'o-')
        ax.grid(ls=':')
        plt.show()
        plt.close()


    # drift diffusion model stability: (2.35) ~ (2.40)
    def drift_diffusion_model_stability(self):
        # === chapter section: 2-5
        print('chapter - section [ 2-5 ]')
        print('')

        # von Neumann stability analysis: the solution is a finite Fourier series
        self.f_i_jp1_sta   = self.xi**(self.q+1)*sp.exp(sp.I*self.beta*self.p    *self.h)
        self.f_i_j_sta     = self.xi**(self.q)  *sp.exp(sp.I*self.beta*self.p    *self.h)
        self.f_im1_j_sta   = self.xi**(self.q)  *sp.exp(sp.I*self.beta*(self.p-1)*self.h)
        self.f_ip1_j_sta   = self.xi**(self.q)  *sp.exp(sp.I*self.beta*(self.p+1)*self.h)
        self.f_im1_jp1_sta = self.xi**(self.q+1)*sp.exp(sp.I*self.beta*(self.p-1)*self.h)
        self.f_ip1_jp1_sta = self.xi**(self.q+1)*sp.exp(sp.I*self.beta*(self.p+1)*self.h)

        # === FTFS time evolution 1
        self.ddm_im_fdm_ftfs_LHS = (self.f_i_jp1 - self.f_i_j)   / ( (self.t + self.dt) - self.t )

        self.f_i_diff1_fw_jp1 = (self.f_ip1_jp1 - self.f_i_jp1)   / ( (self.x + self.dx) - self.x )
        self.f_i_diff1_bw_jp1 = (self.f_i_jp1 - self.f_im1_jp1)   / ( (self.x + self.dx) - self.x )
        self.f_i_diff2_ct_jp1 = (self.f_i_diff1_fw_jp1 - self.f_i_diff1_bw_jp1) / ( (self.x + self.dx/2) - (self.x - self.dx/2) )
        self.ddm_im_fdm_ftfs_RHS_diffusion = self.D * self.f_i_diff2_ct_jp1.expand().simplify()
        self.ddm_im_fdm_ftfs_RHS_drift = -self.v * self.f_i_diff1_fw_jp1

        # === FTFS time evolution 2
        self.ddm_im_fdm_ftfs_LHS = self.ddm_im_fdm_ftfs_LHS * self.dt

        self.ddm_im_fdm_ftfs_RHS_diffusion = self.ddm_im_fdm_ftfs_RHS_diffusion * self.dt
        self.ddm_im_fdm_ftfs_RHS_diffusion = self.ddm_im_fdm_ftfs_RHS_diffusion.expand()
        self.ddm_im_fdm_ftfs_RHS_diffusion = self.ddm_im_fdm_ftfs_RHS_diffusion.subs(self.dt/self.dx**2, self.R2)
        
        self.ddm_im_fdm_ftfs_RHS_drift = self.ddm_im_fdm_ftfs_RHS_drift * self.dt
        self.ddm_im_fdm_ftfs_RHS_drift = self.ddm_im_fdm_ftfs_RHS_drift.expand()
        self.ddm_im_fdm_ftfs_RHS_drift = self.ddm_im_fdm_ftfs_RHS_drift.subs(self.dt/self.dx, self.R1)

        # === FTFS time evolution 3
        self.ddm_im_fdm_ftfs_LHS = self.ddm_im_fdm_ftfs_LHS
        self.ddm_im_fdm_ftfs_RHS = self.ddm_im_fdm_ftfs_RHS_diffusion + self.ddm_im_fdm_ftfs_RHS_drift
        
        # === FTFS FDM equation stability check
        self.ddm_im_fdm_ftfs_EQ = sp.Eq(self.ddm_im_fdm_ftfs_LHS, self.ddm_im_fdm_ftfs_RHS)
        print('DDM FTFS EQ:', self.ddm_im_fdm_ftfs_EQ)
        print('')
        
        self.ddm_im_fdm_ftfs_EQ_LHS_sta = self.ddm_im_fdm_ftfs_EQ.lhs.subs(self.f_i_jp1, self.f_i_jp1_sta)
        self.ddm_im_fdm_ftfs_EQ_LHS_sta = self.ddm_im_fdm_ftfs_EQ_LHS_sta.subs(self.f_i_j, self.f_i_j_sta)
        print('DDM FTFS EQ LHS:', self.ddm_im_fdm_ftfs_EQ_LHS_sta)
        print('')

        self.ddm_im_fdm_ftfs_EQ_RHS_sta = self.ddm_im_fdm_ftfs_EQ.rhs.subs(self.f_ip1_jp1, self.f_ip1_jp1_sta)
        self.ddm_im_fdm_ftfs_EQ_RHS_sta = self.ddm_im_fdm_ftfs_EQ_RHS_sta.subs(self.f_im1_jp1, self.f_im1_jp1_sta)
        self.ddm_im_fdm_ftfs_EQ_RHS_sta = self.ddm_im_fdm_ftfs_EQ_RHS_sta.subs(self.f_i_jp1, self.f_i_jp1_sta)
        print('DDM FTFS EQ RHS:', self.ddm_im_fdm_ftfs_EQ_RHS_sta)
        print('')

        self.ddm_im_fdm_ftfs_EQ_LHS_sta = self.ddm_im_fdm_ftfs_EQ_LHS_sta / self.xi**(self.q+1) / sp.exp(sp.I*self.beta*self.p*self.h)
        self.ddm_im_fdm_ftfs_EQ_RHS_sta = self.ddm_im_fdm_ftfs_EQ_RHS_sta / self.xi**(self.q+1) / sp.exp(sp.I*self.beta*self.p*self.h)
        self.ddm_im_fdm_ftfs_EQ_LHS_sta = -self.ddm_im_fdm_ftfs_EQ_LHS_sta * self.xi + self.xi
        self.ddm_im_fdm_ftfs_EQ_RHS_sta = -self.ddm_im_fdm_ftfs_EQ_RHS_sta * self.xi + self.xi
        self.ddm_im_fdm_ftfs_EQ_LHS_sta = self.ddm_im_fdm_ftfs_EQ_LHS_sta.expand().simplify()
        self.ddm_im_fdm_ftfs_EQ_RHS_sta = self.ddm_im_fdm_ftfs_EQ_RHS_sta.simplify().rewrite(sp.sin).simplify()
        self.ddm_im_fdm_ftfs_EQ_RHS_sta = self.ddm_im_fdm_ftfs_EQ_RHS_sta.subs(sp.cos(self.beta*self.h), 1-sp.sin(self.beta*self.h/2)**2)
        self.ddm_im_fdm_ftfs_EQ_RHS_sta = self.ddm_im_fdm_ftfs_EQ_RHS_sta.expand().simplify()
        print('DDM FTFS EQ LHS stability:', self.ddm_im_fdm_ftfs_EQ_LHS_sta)
        print('DDM FTFS EQ RHS stability:', self.ddm_im_fdm_ftfs_EQ_RHS_sta)
        print('')

        # === FTBS time evolution 1
        self.ddm_im_fdm_ftbs_LHS = (self.f_i_jp1 - self.f_i_j)   / ( (self.t + self.dt) - self.t )

        self.f_i_diff1_fw_jp1 = (self.f_ip1_jp1 - self.f_i_jp1)   / ( (self.x + self.dx) - self.x )
        self.f_i_diff1_bw_jp1 = (self.f_i_jp1 - self.f_im1_jp1)   / ( (self.x + self.dx) - self.x )
        self.f_i_diff2_ct_jp1 = (self.f_i_diff1_fw_jp1 - self.f_i_diff1_bw_jp1) / ( (self.x + self.dx/2) - (self.x - self.dx/2) )
        self.ddm_im_fdm_ftbs_RHS_diffusion = self.D * self.f_i_diff2_ct_jp1.expand().simplify()
        self.ddm_im_fdm_ftbs_RHS_drift = -self.v * self.f_i_diff1_bw_jp1

        # === FTBS time evolution 2
        self.ddm_im_fdm_ftbs_LHS = self.ddm_im_fdm_ftbs_LHS * self.dt

        self.ddm_im_fdm_ftbs_RHS_diffusion = self.ddm_im_fdm_ftbs_RHS_diffusion * self.dt
        self.ddm_im_fdm_ftbs_RHS_diffusion = self.ddm_im_fdm_ftbs_RHS_diffusion.expand()
        self.ddm_im_fdm_ftbs_RHS_diffusion = self.ddm_im_fdm_ftbs_RHS_diffusion.subs(self.dt/self.dx**2, self.R2)
        
        self.ddm_im_fdm_ftbs_RHS_drift = self.ddm_im_fdm_ftbs_RHS_drift * self.dt
        self.ddm_im_fdm_ftbs_RHS_drift = self.ddm_im_fdm_ftbs_RHS_drift.expand()
        self.ddm_im_fdm_ftbs_RHS_drift = self.ddm_im_fdm_ftbs_RHS_drift.subs(self.dt/self.dx, self.R1)

        # === FTBS time evolution 3
        self.ddm_im_fdm_ftbs_LHS = self.ddm_im_fdm_ftbs_LHS
        self.ddm_im_fdm_ftbs_RHS = self.ddm_im_fdm_ftbs_RHS_diffusion + self.ddm_im_fdm_ftbs_RHS_drift
        
        # === FTBS FDM equation stability check
        self.ddm_im_fdm_ftbs_EQ = sp.Eq(self.ddm_im_fdm_ftbs_LHS, self.ddm_im_fdm_ftbs_RHS)
        print('DDM FTBS EQ:', self.ddm_im_fdm_ftbs_EQ)
        print('')
        
        self.ddm_im_fdm_ftbs_EQ_LHS_sta = self.ddm_im_fdm_ftbs_EQ.lhs.subs(self.f_i_jp1, self.f_i_jp1_sta)
        self.ddm_im_fdm_ftbs_EQ_LHS_sta = self.ddm_im_fdm_ftbs_EQ_LHS_sta.subs(self.f_i_j, self.f_i_j_sta)
        print('DDM FTBS EQ LHS:', self.ddm_im_fdm_ftbs_EQ_LHS_sta)
        print('')

        self.ddm_im_fdm_ftbs_EQ_RHS_sta = self.ddm_im_fdm_ftbs_EQ.rhs.subs(self.f_ip1_jp1, self.f_ip1_jp1_sta)
        self.ddm_im_fdm_ftbs_EQ_RHS_sta = self.ddm_im_fdm_ftbs_EQ_RHS_sta.subs(self.f_im1_jp1, self.f_im1_jp1_sta)
        self.ddm_im_fdm_ftbs_EQ_RHS_sta = self.ddm_im_fdm_ftbs_EQ_RHS_sta.subs(self.f_i_jp1, self.f_i_jp1_sta)
        print('DDM FTBS EQ RHS:', self.ddm_im_fdm_ftbs_EQ_RHS_sta)
        print('')

        self.ddm_im_fdm_ftbs_EQ_LHS_sta = self.ddm_im_fdm_ftbs_EQ_LHS_sta / self.xi**(self.q+1) / sp.exp(sp.I*self.beta*self.p*self.h)
        self.ddm_im_fdm_ftbs_EQ_RHS_sta = self.ddm_im_fdm_ftbs_EQ_RHS_sta / self.xi**(self.q+1) / sp.exp(sp.I*self.beta*self.p*self.h)
        self.ddm_im_fdm_ftbs_EQ_LHS_sta = -self.ddm_im_fdm_ftbs_EQ_LHS_sta * self.xi + self.xi
        self.ddm_im_fdm_ftbs_EQ_RHS_sta = -self.ddm_im_fdm_ftbs_EQ_RHS_sta * self.xi + self.xi
        self.ddm_im_fdm_ftbs_EQ_LHS_sta = self.ddm_im_fdm_ftbs_EQ_LHS_sta.expand().simplify()
        self.ddm_im_fdm_ftbs_EQ_RHS_sta = self.ddm_im_fdm_ftbs_EQ_RHS_sta.simplify().rewrite(sp.sin).simplify()
        self.ddm_im_fdm_ftbs_EQ_RHS_sta = self.ddm_im_fdm_ftbs_EQ_RHS_sta.subs(sp.cos(self.beta*self.h), 1-sp.sin(self.beta*self.h/2)**2)
        self.ddm_im_fdm_ftbs_EQ_RHS_sta = self.ddm_im_fdm_ftbs_EQ_RHS_sta.expand().simplify()
        print('DDM FTBS EQ LHS stability:', self.ddm_im_fdm_ftbs_EQ_LHS_sta)
        print('DDM FTBS EQ RHS stability:', self.ddm_im_fdm_ftbs_EQ_RHS_sta)
        print('')

        # for D>0, R1>0, R2>0,
        # FTFS finite difference scheme is stable when v < 0
        # FTBS finite difference scheme is stable when v > 0
        #   -> upwinding scheme

        # upwinding scheme can be replaced by the more commonly used
        # Scharfetter-Gummel finite scheme


#=============================================================
# CLASS: the van Roosbroeck system (3.1) ~ (3.48)
#=============================================================

class VRS:

    # constructor
    def __init__(self):

        # function
        self.n = sp.symbols('n', cls=sp.Function, real=True)
        self.p = sp.symbols('p', cls=sp.Function, real=True)
        self.V = sp.symbols('V', cls=sp.Function, real=True)
        self.Jn = sp.symbols('Jn', cls=sp.Function, real=True)
        self.Jp = sp.symbols('Jp', cls=sp.Function, real=True)
        self.E = sp.symbols('E', cls=sp.Function, real=True)
        
        # independent variables
        self.x  = sp.symbols('x',  real=True)
        self.dx = sp.symbols('dx', real=True)
        self.t  = sp.symbols('t',  real=True)
        self.dt = sp.symbols('dt', real=True)

        # electron density
        self.e_im1_j = sp.symbols('e_{i-1}^{j}', real=True)
        self.e_i_j   = sp.symbols('e_{i}^{j}',   real=True)
        self.e_ip1_j = sp.symbols('e_{i+1}^{j}', real=True)
        self.e_im1_jp1 = sp.symbols('e_{i-1}^{j+1}', real=True)
        self.e_i_jp1   = sp.symbols('e_{i}^{j+1}',   real=True)
        self.e_ip1_jp1 = sp.symbols('e_{i+1}^{j+1}', real=True)

        # hole density
        self.h_im1_j = sp.symbols('h_{i-1}^{j}', real=True)
        self.h_i_j   = sp.symbols('h_{i}^{j}',   real=True)
        self.h_ip1_j = sp.symbols('h_{i+1}^{j}', real=True)
        self.h_im1_jp1 = sp.symbols('h_{i-1}^{j+1}', real=True)
        self.h_i_jp1   = sp.symbols('h_{i}^{j+1}',   real=True)
        self.h_ip1_jp1 = sp.symbols('h_{i+1}^{j+1}', real=True)
        
        # constant
        self.q = sp.symbols('q', real=True)
        self.kb = sp.symbols('kb', real=True)

        # mobility
        self.mu_n = sp.symbols('mu_n', real=True)
        self.mu_p = sp.symbols('mu_p', real=True)

        # diffusion constant
        self.D_n = sp.symbols('D_n', real=True)
        self.D_p = sp.symbols('D_p', real=True)

        # electric permittivity
        self.ep = sp.symbols('ep', real=True)

        # reaction
        self.R = sp.symbols('R', real=True)

        # doping concentration
        self.C = sp.symbols('C', real=True)

        # intrinsic concentration, initial potential
        self.n_int = sp.symbols('n_int', real=True)
        self.V0 = sp.symbols('V0', real=True)
        self.delta_V = sp.symbols('delta_V', real=True)

        # constant current
        self.Jn0 = sp.symbols('Jn0', real=True)
        self.Jp0 = sp.symbols('Jp0', real=True)

        # constant electic field
        self.E0 = sp.symbols('E0', real=True)

        # temperature, kelvin
        self.T = sp.symbols('T', real=True)

        # thermal voltage
        self.Vther = sp.symbols('Vther', real=True)


    def van_roosbroeck_system(self):
        #
        vrs_Jn_LHS =  self.Jn(self.x, self.t)
        vrs_Jn_RHS = (-self.q) * self.mu_n * self.n(self.x, self.t) * (-self.E(self.x, self.t)) + \
                     (-self.q) * self.D_n * (-self.n(self.x, self.t).diff(self.x))
        print('VRS Jn LHS:', vrs_Jn_LHS)
        print('VRS Jn RHS:', vrs_Jn_RHS)
        print('')
        #
        vrs_Jp_LHS =  self.Jp(self.x, self.t)
        vrs_Jp_RHS = (+self.q) * self.mu_p * self.p(self.x, self.t) * (+self.E(self.x, self.t)) + \
                     (+self.q) * self.D_p * (-self.p(self.x, self.t).diff(self.x))
        print('VRS Jp LHS:', vrs_Jp_LHS)
        print('VRS Jp RHS:', vrs_Jp_RHS)
        print('')
        #
        vrs_n_LHS =  self.n(self.x, self.t).diff(self.t)
        vrs_n_RHS = 1 / (-self.q) * (-self.Jn(self.x, self.t).diff(self.x)) + self.R
        print('VRS n LHS:', vrs_n_LHS)
        print('VRS n RHS:', vrs_n_RHS)
        print('')
        #
        vrs_p_LHS =  self.p(self.x, self.t).diff(self.t)
        vrs_p_RHS = 1 / (+self.q) * (-self.Jp(self.x, self.t).diff(self.x)) + self.R
        print('VRS p LHS:', vrs_p_LHS)
        print('VRS p RHS:', vrs_p_RHS)
        print('')
        #
        vrs_V_LHS = ( self.ep * (-self.V(self.x, self.t).diff(self.x)) ).diff(self.x)
        vrs_V_RHS = self.q * (self.p(self.x, self.t) - self.n(self.x, self.t) + self.C)
        print('VRS V LHS:', vrs_V_LHS)
        print('VRS V RHS:', vrs_V_RHS)
        print('')


    def no_current_van_roosbroeck_system(self):
        # n current equation
        n_current_eq_LHS = self.Jn(self.x)
        n_current_eq_RHS = self.q * self.mu_n * self.n(self.x) * -self.V(self.x).diff(self.x) + \
                           self.q * self.D_n * self.n(self.x).diff(self.x)
        n_current_eq = sp.Eq(n_current_eq_LHS, n_current_eq_RHS)
        print('n current EQ:', n_current_eq)
        n_current_eq = n_current_eq.subs(self.D_n, self.mu_n * self.kb * self.T / self.q)
        print('n current EQ w/ Einstein relation:', n_current_eq)
        print('')

        # p current equation
        p_current_eq_LHS = self.Jp(self.x)
        p_current_eq_RHS = self.q * self.mu_p * self.p(self.x) * -self.V(self.x).diff(self.x) - \
                           self.q * self.D_p * self.p(self.x).diff(self.x)
        p_current_eq = sp.Eq(p_current_eq_LHS, p_current_eq_RHS)
        print('p current EQ:', p_current_eq)
        p_current_eq = p_current_eq.subs(self.D_p, self.mu_p * self.kb * self.T / self.q)
        print('p current EQ w/ Einstein relation:', p_current_eq)
        print('')

        # n current equation w/ no current
        n_current_eq_no_current = n_current_eq.subs(self.Jn(self.x), 0)
        n_current_eq_no_current = n_current_eq_no_current.subs(self.V(self.x).diff(self.x), 1)
        print('n current EQ w/ no current:', n_current_eq_no_current)
        n_current_eq_no_current_SOL = sp.dsolve(n_current_eq_no_current, ics={self.n(0):self.n_int})
        n_current_eq_no_current_SOL = n_current_eq_no_current_SOL.subs(self.x, self.delta_V)
        print('n current EQ w/ no current SOL:', n_current_eq_no_current_SOL, ' delta_V = V - Vo')
        print('')

        # p current equation w/ no current
        p_current_eq_no_current = p_current_eq.subs(self.Jp(self.x), 0)
        p_current_eq_no_current = p_current_eq_no_current.subs(self.V(self.x).diff(self.x), 1)
        print('p current EQ w/ no current:', p_current_eq_no_current)
        p_current_eq_no_current_SOL = sp.dsolve(p_current_eq_no_current, ics={self.p(0):self.n_int})
        p_current_eq_no_current_SOL = p_current_eq_no_current_SOL.subs(self.x, self.delta_V)
        print('p current EQ w/ no current SOL:', p_current_eq_no_current_SOL, ' delta_V = V - Vo')
        print('')


    def constant_current_van_roosbroeck_system(self):
        # n current equation
        n_current_eq_LHS = self.Jn(self.x)
        n_current_eq_RHS = self.q * self.mu_n * self.n(self.x) * -self.V(self.x).diff(self.x) + \
                           self.q * self.D_n * self.n(self.x).diff(self.x)
        n_current_eq = sp.Eq(n_current_eq_LHS, n_current_eq_RHS)
        n_current_eq = n_current_eq.subs(self.D_n, self.mu_n * self.kb * self.T / self.q)

        # p current equation
        p_current_eq_LHS = self.Jp(self.x)
        p_current_eq_RHS = self.q * self.mu_p * self.p(self.x) * -self.V(self.x).diff(self.x) - \
                           self.q * self.D_p * self.p(self.x).diff(self.x)
        p_current_eq = sp.Eq(p_current_eq_LHS, p_current_eq_RHS)
        p_current_eq = p_current_eq.subs(self.D_p, self.mu_p * self.kb * self.T / self.q)
        
        # n current equation w/ constant current
        n_current_eq_const_current = n_current_eq.subs(self.Jn(self.x), self.Jn0)
        n_current_eq_const_current_LHS = (n_current_eq_const_current.lhs / self.T / self.kb / self.mu_n).simplify()
        n_current_eq_const_current_RHS = (n_current_eq_const_current.rhs / self.T / self.kb / self.mu_n).simplify()
        n_current_eq_const_current = sp.Eq(n_current_eq_const_current_LHS, n_current_eq_const_current_RHS)
        print('n current EQ w/ const current:', n_current_eq_const_current)

        # p current equation w/ constant current
        p_current_eq_const_current = p_current_eq.subs(self.Jp(self.x), self.Jp0)
        p_current_eq_const_current_LHS = (p_current_eq_const_current.lhs / self.T / self.kb / self.mu_p).simplify()
        p_current_eq_const_current_RHS = (p_current_eq_const_current.rhs / self.T / self.kb / self.mu_p).simplify()
        p_current_eq_const_current = sp.Eq(p_current_eq_const_current_LHS, p_current_eq_const_current_RHS)
        print('p current EQ w/ const current:', p_current_eq_const_current)
        print('')


    def constant_current_linear_potential_van_roosbroeck_system(self):
        # n current equation
        n_current_eq_LHS = self.Jn(self.x)
        n_current_eq_RHS = self.q * self.mu_n * self.n(self.x) * -self.V(self.x).diff(self.x) + \
                           self.q * self.D_n * self.n(self.x).diff(self.x)
        n_current_eq = sp.Eq(n_current_eq_LHS, n_current_eq_RHS)
        n_current_eq = n_current_eq.subs(self.D_n, self.mu_n * self.kb * self.T / self.q)

        # p current equation
        p_current_eq_LHS = self.Jp(self.x)
        p_current_eq_RHS = self.q * self.mu_p * self.p(self.x) * -self.V(self.x).diff(self.x) - \
                           self.q * self.D_p * self.p(self.x).diff(self.x)
        p_current_eq = sp.Eq(p_current_eq_LHS, p_current_eq_RHS)
        p_current_eq = p_current_eq.subs(self.D_p, self.mu_p * self.kb * self.T / self.q)
        
        # n current equation w/ constant current, linear potential
        n_current_eq_const_current_linear_potential = n_current_eq.subs(self.Jn(self.x), self.Jn0)
        n_current_eq_const_current_linear_potential = n_current_eq_const_current_linear_potential.subs(self.V(self.x).diff(self.x), -self.E0)
        n_current_eq_const_current_linear_potential_LHS = (n_current_eq_const_current_linear_potential.lhs / self.T / self.kb / self.mu_n).simplify()
        n_current_eq_const_current_linear_potential_RHS = (n_current_eq_const_current_linear_potential.rhs / self.T / self.kb / self.mu_n).simplify()
        n_current_eq_const_current_linear_potential = sp.Eq(n_current_eq_const_current_linear_potential_LHS, n_current_eq_const_current_linear_potential_RHS)
        print('n current EQ w/ const current linear potential:', n_current_eq_const_current_linear_potential)
        
        # p current equation w/ constant current
        p_current_eq_const_current_linear_potential = p_current_eq.subs(self.Jp(self.x), self.Jp0)
        p_current_eq_const_current_linear_potential = p_current_eq_const_current_linear_potential.subs(self.V(self.x).diff(self.x), -self.E0)
        p_current_eq_const_current_linear_potential_LHS = (p_current_eq_const_current_linear_potential.lhs / self.T / self.kb / self.mu_p).simplify()
        p_current_eq_const_current_linear_potential_RHS = (p_current_eq_const_current_linear_potential.rhs / self.T / self.kb / self.mu_p).simplify()
        p_current_eq_const_current_linear_potential = sp.Eq(p_current_eq_const_current_linear_potential_LHS, p_current_eq_const_current_linear_potential_RHS)
        print('p current EQ w/ const current linear potential:', p_current_eq_const_current_linear_potential)
        print('')


    def gummel_iteration_damping(self):
        # Gummel iteration
        # 1. solving the coupled set of drift-diffusion equations and Poisson equation
        #    using a decoupling procedure
        # 2. utilizing Slotboom variables instead of the Quasi-Fermi level variables
        # 3. initial conditions V0 -> n0=n_int*Exp(-V0/(kb*t)), p0=n_int*Exp(V0/(kb*t))) -> solve poisson equation
        # 4. V_k -> finding n_k, p_k (Slotboom variables) -> finding V_(k+1) -> finding n_(k+1), p_(k+1) -> ...  

        # Damping
        # V_(k+1) = alpha * V_(k+1) + ( 1 - alpha ) * V_k -> linear combination
        #   -> V_(k+1): newly computed potential, V_k: current potential
        #   -> the same can be done for the concentration of electrons and holes
        # 
        
        pass


    def drift_diffusion_in_van_roosbroeck_system(self):
        # n drift diffusion equation
        vrs_Jn_LHS =  self.Jn(self.x, self.t)
        vrs_Jn_RHS = (-self.q) * self.mu_n * self.n(self.x, self.t) * (-self.E(self.x, self.t)) + \
                     (-self.q) * self.D_n * (-self.n(self.x, self.t).diff(self.x))
        vrs_n_LHS =  self.n(self.x, self.t).diff(self.t)
        vrs_n_RHS = 1 / (-self.q) * (-self.Jn(self.x, self.t).diff(self.x)) + self.R

        vrs_Jn_RHS = vrs_Jn_RHS.subs(self.E(self.x, self.t), self.E0)                       # linear potential
        vrs_Jn_RHS = vrs_Jn_RHS.subs(self.D_n, self.mu_n * self.kb * self.T / self.q)       # Einstein relation
        vrs_n_RHS  = vrs_n_RHS.subs(self.R, 0)                                              # no reaction

        vrs_Jn_RHS = vrs_Jn_RHS.subs(self.E(self.x, self.t), -self.V(self.x, self.t).diff(self.x))      # electric potential
        vrs_n_RHS  = vrs_n_RHS.subs(self.Jn(self.x, self.t), vrs_Jn_RHS).simplify().expand()            # substitution
        vrs_n_RHS  = vrs_n_RHS.subs(self.kb * self.T / self.q, self.Vther)                              # thermal volate

        vrs_n_EQ = sp.Eq(vrs_n_LHS, vrs_n_RHS)
        print('n drift diffusion:', vrs_n_EQ)

        # p drift diffusion equation
        vrs_Jp_LHS =  self.Jp(self.x, self.t)
        vrs_Jp_RHS = (+self.q) * self.mu_p * self.p(self.x, self.t) * (+self.E(self.x, self.t)) + \
                     (+self.q) * self.D_p * (-self.p(self.x, self.t).diff(self.x))
        vrs_p_LHS =  self.p(self.x, self.t).diff(self.t)
        vrs_p_RHS = 1 / (+self.q) * (-self.Jp(self.x, self.t).diff(self.x)) + self.R

        vrs_Jp_RHS = vrs_Jp_RHS.subs(self.E(self.x, self.t), self.E0)                       # linear potential
        vrs_Jp_RHS = vrs_Jp_RHS.subs(self.D_p, self.mu_p * self.kb * self.T / self.q)       # Einstein relation
        vrs_p_RHS  = vrs_p_RHS.subs(self.R, 0)                                              # no reaction
        
        vrs_Jp_RHS = vrs_Jp_RHS.subs(self.E(self.x, self.t), -self.V(self.x, self.t).diff(self.x))      # electric potential
        vrs_p_RHS  = vrs_p_RHS.subs(self.Jp(self.x, self.t), vrs_Jp_RHS).simplify().expand()            # substitution
        vrs_p_RHS  = vrs_p_RHS.subs(self.kb * self.T / self.q, self.Vther)                              # thermal volate

        vrs_p_EQ = sp.Eq(vrs_p_LHS, vrs_p_RHS)
        print('p drift diffusion:', vrs_p_EQ)
        print('')
        
        

#=============================================================
# MAIN
#=============================================================

if False:
    model = HEAT_EQ()
    model.solving_equation()

if False:
    fdm = FDM()
    fdm.finite_difference_method()
    fdm.stability_convergence_consistency()
    fdm.drift_diffusion_model()
    fdm.crank_nicholson_method()
    fdm.drift_diffusion_model_stability()

if True:
    vrs = VRS()
    vrs.van_roosbroeck_system()
    vrs.no_current_van_roosbroeck_system()
    vrs.constant_current_van_roosbroeck_system()
    vrs.constant_current_linear_potential_van_roosbroeck_system()
    vrs.gummel_iteration_damping()
    vrs.drift_diffusion_in_van_roosbroeck_system()




