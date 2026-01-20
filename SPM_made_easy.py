#
# TITLE: 
# AUTHOR: Hyunseung Yoo
# PURPOSE: 
# REVISION: 
# REFERENCE: semiconductor physics modeling made easy (2025)
#

import sys
import time
import numpy as np
import scipy as sc
import sympy as sy
import matplotlib.pyplot as plt

#
# CLASS
#

class CH01_DRUDE_MODEL:

    # fundamental constants of free electron
    q = 1.602e-19           # [C]
    m0_e = 9.109e-31        # [kg]

    def __init__(self, electric_field, relaxation_time, carrier_density, mobility):
        # calculation: average velocity
        self.average_velocity = (self.q * electric_field / self.m0_e) * relaxation_time 

        # calculation: conductivity
        self.conductivity = self.q * carrier_density * mobility   

        # calculation: current density
        self.current_density = self.conductivity * electric_field
        
        #
        output_string  = ' electric field = %.2e [V/m] \n' % (electric_field)
        output_string += ' relaxation time = %.2e [s] \n' % (relaxation_time)
        output_string += ' average velocity = %.2e [m/s] \n' % (self.average_velocity)
        output_string += '\n'
        output_string += ' carrier density = %.2e [m^-3] \n' % (carrier_density)
        output_string += ' mobility = %.2e [m^2/V*s] \n' % (mobility)
        output_string += ' conductivity = %.2e [S/m] \n' % (self.conductivity)
        output_string += '\n'
        output_string += ' current density = %.2e [A/m^2] \n' % (self.current_density)
        
        #
        print(output_string)

#
# MAIN
#

if False:
    electric_field = 1000.0     # [V] / [m]
    relaxation_time = 1e-14     # [s]
    carrier_density = 1e28      # [m]^-3
    mobility = 0.14             # [m]^2 / [V] [s]

    ch01_drude_model = CH01_DRUDE_MODEL(electric_field, relaxation_time, carrier_density, mobility)


#
# CLASS
#

class CH02_OHMS_LAW:

    # fundamental constants
    resistivity = 1.68e-8       # [ohm] [m], copper
    alpha = 0.0039              # temperature coefficient, coper

    # constructor
    def __init__(self, voltage, resistance, length, cross_section_area, R0, delta_T):
        # calculation: current
        cal_current = voltage / resistance

        # calculation: resistance
        cal_resistance = self.resistivity * length / cross_section_area

        # calculation:
        cal_resistance_change = R0 * self.alpha * delta_T

        #
        output_string  = ' voltage = %.1e [V] \n' % (voltage)
        output_string += ' resistance = %1.e [ohm] \n' % (resistance)
        output_string += ' calculated current = %1.e [A] \n\n' % (cal_current)
        output_string += ' resistivity = %.1e [ohm] [m] \n' % (self.resistivity)
        output_string += ' length = %.1e [m] \n' % (length)
        output_string += ' cross section area = %.1e [m]^2 \n' % (cross_section_area)
        output_string += ' calculated resistance = %.1e [ohm] \n\n' % (cal_resistance)
        output_string += ' resistance at ref temp = %1.e [ohm] \n' % (R0)
        output_string += ' temperature coefficient = %.1e \n' % (self.alpha)
        output_string += ' change in temperature = %.1e [K] \n' % (delta_T)
        output_string += ' change in resistance = %.1e [ohm] \n\n' % (cal_resistance_change)

        #
        print(output_string)
        

#
# MAIN
#

if True:
    voltage = 10.0                  # [V]
    resistance = 5.0                # [ohm]
    length = 2.0                    # [m]
    cross_section_area = 1.0e-6     # [m]^2
    R0 = 5.0                        # [ohm] at reference temperature
    delta_T = 25.0                  # [K] change in temperature
    
    #
    ch02_ohms_law = CH02_OHMS_LAW(voltage, resistance, length, cross_section_area, R0, delta_T)


