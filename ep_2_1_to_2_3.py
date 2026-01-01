#
# TITLE: van Roosbroeck system
# AUTHOR: Hyunseung Yoo
# PURPOSE: 
# REVISION: 
# REFERENCE: a numerical study of the van Roosbroeck system for semiconductor (SJSU, 2018) 
#


import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


#
# CLASS: Heat Equation (2.1) ~ (2.3)
#

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

        # PDE x t base
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
        
        # finding Cn
        SOL_x_t_base_Cn_RHS = SOL_x_t_base_IC_EQ_RHS_integral * 2 / self.L
        SOL_x_t_base_Cn_LHS = SOL_x_t_base_IC_EQ_LHS_integral * 2 / self.L
        SOL_x_t_base_Cn = sp.Eq(SOL_x_t_base_Cn_LHS, SOL_x_t_base_Cn_RHS)
        print('calculated Cn:', SOL_x_t_base_Cn)
        print('')
        
        

#
# MAIN
#

model = HEAT_EQ()
model.solving_equation()
