import SUAVE
import copy
import math
from SUAVE.Core import Units, Data
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_mass
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_max_lift_coeff  
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_induced_velocity_matrix import compute_induced_velocity_matrix
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_vortex_distribution import compute_vortex_distribution
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_RHS_matrix  import compute_RHS_matrix

import numpy as np
import pylab as plt
from copy import deepcopy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    #-----------------------------------------------------------------
    # Specify the analysis settings for the problem:
    #-----------------------------------------------------------------
    analysis_settings                          = Data()
    analysis_settings.case                     = 'disturbed_freestream' #'uniform_freestream' #
    analysis_settings.rotation                 = 'ccw'
    analysis_settings.wake_type                = 'viscous' # 'inviscid' # 
    analysis_settings.use_Blade_Element_Theory = True
    
    #-----------------------------------------------------------------
    # Specify the vehicle operating conditions:
    #-----------------------------------------------------------------    
    cruise_speed = 30 #np.array([25.,30., 35., 40., 45., 50., 55., 60.]) #50 # 80 gives CL of 0.270, 50 gives CLof 0.69, 45 gives CL of 0.854
    wing_aoa     = np.array([[3*Units.deg]])#np.linspace(-2,10,11)*Units.deg
    altitude     = 4000 #np.array([50., 100., 200., 500., 800., 1000., 1500., 2500., 3000., 4000.]) #4000
    
    conditions   = cruise_conditions(cruise_speed, altitude, wing_aoa) 
    
    #-----------------------------------------------------------------
    # Setup the vehicle configuration (one-time deal):
    #-----------------------------------------------------------------
    prop_type    = 'small_tip_mount' #   'larger_half_mount' #
    vehicle      = vehicle_setup(conditions, prop_type)
    
    #-----------------------------------------------------------------
    # Test off-design condtions:
    #-----------------------------------------------------------------   
    conditions.aerodynamics.angle_of_attack        = np.array([[5*Units.deg]])
    conditions.altitude                            = 50    
    vehicle      = wing_effect(vehicle,conditions)
    vehicle.propulsors.prop_net.propeller.analysis_settings = analysis_settings
    
    drag_percent = 0.2
    
    
    
    #------------------------------------------------------------------------------------------
    # Step 1: Determine U_inf required for trimmed flight for the pusher config (Secant Method)
    #------------------------------------------------------------------------------------------
    diff = 1
    tol = 1e-5
    f_old = vehicle.propulsors.prop_net.propeller.design_thrust
    
    # Initial Guess:
    V0 = 62 # max stall speed for given conditions
    vehicle.cruise_speed = V0
    conditions.frames.inertial.velocity_vector[0][0] = V0
    F, Q, P, Cp , outputs , etap = vehicle.propulsors.prop_net.propeller.spin(conditions,vehicle)
    V_inf = vehicle.cruise_speed
    CL_wing  = vehicle.CL_design
    AR       = vehicle.wings.main_wing.aspect_ratio
    e        = 0.7
    CD_wing  = 0.012 + CL_wing**2/(np.pi*AR*e)      
    Drag0     = CD_wing*0.5*conditions.freestream.density*V0**2*vehicle.reference_area
    f0 = (drag_percent*Drag0 - 2*F)    
    
    # Second Guess:
    V1 = 60 # starting velocity
    vehicle.cruise_speed = V1
    conditions.frames.inertial.velocity_vector[0][0] = V1
    F, Q, P, Cp , outputs , etap = vehicle.propulsors.prop_net.propeller.spin(conditions,vehicle)
    V_inf = vehicle.cruise_speed
    Drag1     = Drag0 * V1**2 / (V0**2) #CD_wing*0.5*conditions.freestream.density*V1**2*vehicle.reference_area    
    f1 = (drag_percent*Drag1 - 2*F)    
    
    while abs(diff)>tol:
        
        # Update freestream velocity to try to reach convergence:
        V2                   = V1 - f1*((V1-V0)/(f1-f0))
        vehicle.cruise_speed = V2
        conditions.frames.inertial.velocity_vector[0][0] = V2
               
        #------------------------------------------------------------
        # COMPUTE VEHICLE DRAG:
        #------------------------------------------------------------
        
        V_inf = vehicle.cruise_speed
        CL_wing  = vehicle.CL_design
        AR       = vehicle.wings.main_wing.aspect_ratio
        e        = 0.7
        CD_wing  = 0.012 + CL_wing**2/(np.pi*AR*e)        
        Drag     = CD_wing*0.5*conditions.freestream.density*V_inf**2*vehicle.reference_area
        
        
        
        # Run propeller model at the given conditions:
        F, Q, P, Cp , outputs , etap = vehicle.propulsors.prop_net.propeller.spin(conditions,vehicle)
        diff = (drag_percent*Drag - 2*F)
        
        f0 = f1
        f1 = diff
        V0 = V1
        V1 = V2         
        
    
    #-----------------------------------------------   
    # Step 1b: Plot Results for Distrubed Propeller
    #-----------------------------------------------
    psi   = outputs.azimuthal_distribution_2d[0,:,:]
    r     = outputs.blade_radial_distribution_normalized_2d[0,:,:]    
    
    #plt.figure(0)
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, -np.ones_like(outputs.axial_velocity_distribution_2d[0]) + outputs.axial_velocity_distribution_2d[0]/outputs.velocity[0][0],100,cmap=plt.cm.jet)    
    cbar0 = plt.colorbar(CS_0, ax=axis0)
    cbar0.ax.set_ylabel('$\dfrac{V_a-V_\infty}{V_\infty}$, m/s')
    axis0.set_title('Axial Inflow to Propeller')  
    
    #plt.figure(1)
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, outputs.thrust_distribution_2d[0],100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    # -np.pi+psi turns it 
    cbar0 = plt.colorbar(CS_0, ax=axis0)
    cbar0.ax.set_ylabel('Thrust (N)')
    axis0.set_title('Thrust Distribution of Propeller')  
    
    #plt.figure(2)
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, outputs.torque_distribution_2d[0],100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    # -np.pi+psi turns it 
    cbar0 = plt.colorbar(CS_0, ax=axis0)
    cbar0.ax.set_ylabel('Torque (Nm)')
    axis0.set_title('Torque Distribution of Propeller') 
    
    plt.show()    
    
    #---------------------------------------------------------------------------------------
    # Step 2: Now do the same for the isolated propeller
    #---------------------------------------------------------------------------------------
    analysis_settings.case = 'uniform_freestream'
    vehicle.propulsors.prop_net.propeller.analysis_settings = analysis_settings
    diff_iso = 1
    tol = 1e-5
    f_old = vehicle.propulsors.prop_net.propeller.design_thrust
    
    # Initial Guess:
    V0 = 62 # max stall speed for given conditions
    vehicle.cruise_speed = V0
    conditions.frames.inertial.velocity_vector[0][0] = V0
    F_iso, Q_iso, P_iso, Cp_iso , outputs_iso , etap_iso = vehicle.propulsors.prop_net.propeller.spin(conditions,vehicle)
    V_inf_iso = vehicle.cruise_speed
    CL_wing  = vehicle.CL_design
    AR       = vehicle.wings.main_wing.aspect_ratio
    e        = 0.7
    CD_wing  = 0.012 + CL_wing**2/(np.pi*AR*e)      
    Drag0     = CD_wing*0.5*conditions.freestream.density*V0**2*vehicle.reference_area
    f0 = (drag_percent*Drag0 - 2*F_iso)    
    
    # Second Guess:
    V1 = 60 # starting velocity
    vehicle.cruise_speed = V1
    conditions.frames.inertial.velocity_vector[0][0] = V1
    F_iso, Q_iso, P_iso, Cp_iso , outputs_iso , etap_iso = vehicle.propulsors.prop_net.propeller.spin(conditions,vehicle)
    V_inf_iso = vehicle.cruise_speed
    Drag1     = Drag0 * V1**2 / (V0**2) #CD_wing*0.5*conditions.freestream.density*V1**2*vehicle.reference_area    
    f1 = (drag_percent*Drag1 - 2*F_iso)    
    
    while abs(diff_iso)>tol:
        
        # Update freestream velocity to try to reach convergence:
        V2_iso                   = V1 - f1*((V1-V0)/(f1-f0))
        vehicle.cruise_speed = V2_iso
        conditions.frames.inertial.velocity_vector[0][0] = V2_iso
               
        #------------------------------------------------------------
        # COMPUTE VEHICLE DRAG:
        #------------------------------------------------------------
        
        V_inf_iso = vehicle.cruise_speed
        CL_wing  = vehicle.CL_design
        AR       = vehicle.wings.main_wing.aspect_ratio
        e        = 0.7
        CD_wing  = 0.012 + CL_wing**2/(np.pi*AR*e)        
        Drag_iso     = CD_wing*0.5*conditions.freestream.density*V_inf_iso**2*vehicle.reference_area
        
        
        
        # Run propeller model at the given conditions:
        F_iso, Q_iso, P_iso, Cp_iso , outputs_iso , etap_iso = vehicle.propulsors.prop_net.propeller.spin(conditions,vehicle)
        diff_iso = (drag_percent*Drag_iso - 2*F_iso)
        
        f0 = f1
        f1 = diff_iso
        V0 = V1
        V1 = V2_iso       
    
    #----------------------------------------------   
    # Step 2b: Plot Results for Isolated Propeller
    #----------------------------------------------
    psi   = outputs_iso.azimuthal_distribution_2d[0,:,:]
    r     = outputs_iso.blade_radial_distribution_normalized_2d[0,:,:]    
    
    ##plt.figure(0)
    #fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_0 = axis0.contourf(psi, r, -np.ones_like(outputs_iso.axial_velocity_distribution_2d[0]) + outputs_iso.axial_velocity_distribution_2d[0]/outputs_iso.velocity[0][0],100,cmap=plt.cm.jet)    
    #cbar0 = plt.colorbar(CS_0, ax=axis0)
    #cbar0.ax.set_ylabel('$\dfrac{V_a-V_\infty}{V_\infty}$, m/s')
    #axis0.set_title('Axial Inflow to Propeller')  
    
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, vehicle.propulsors.prop_net.propeller.disturbed_w,100,cmap=plt.cm.jet)    
    cbar0 = plt.colorbar(CS_0, ax=axis0)
    cbar0.ax.set_ylabel('$\dfrac{V_w}{V_\infty}$, m/s')
    axis0.set_title('Downwash at Propeller')     
    
    ##plt.figure(1)
    #fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_0 = axis0.contourf(psi, r, outputs_iso.thrust_distribution_2d[0],100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    # -np.pi+psi turns it 
    #cbar0 = plt.colorbar(CS_0, ax=axis0)
    #cbar0.ax.set_ylabel('Thrust (N)')
    #axis0.set_title('Thrust Distribution of Propeller')  
    
    ##plt.figure(2)
    #fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_0 = axis0.contourf(psi, r, outputs_iso.torque_distribution_2d[0],100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    # -np.pi+psi turns it 
    #cbar0 = plt.colorbar(CS_0, ax=axis0)
    #cbar0.ax.set_ylabel('Torque (Nm)')
    #axis0.set_title('Torque Distribution of Propeller') 
    
    plt.show()
    

    return

def cruise_conditions(cruise_speed, alt, aoa):
    # --------------------------------------------------------------------------------    
    #          Cruise conditions  
    # --------------------------------------------------------------------------------
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data  = atmosphere.compute_values(altitude = alt) 
        
    rho = atmo_data.density[0,0]
    mu  = atmo_data.dynamic_viscosity[0,0]
    T   = atmo_data.temperature[0,0]
    P   = atmo_data.pressure[0,0]
    a   = atmo_data.speed_of_sound[0,0] 
        
    velocity_freestream  = np.array([[cruise_speed]]) #m/s this is the velocity at which the propeller is designed for based on the CLwing value being near 0.7
    mach                 = velocity_freestream/a #np.array([[0.8]]) #
    re                   = rho*a*mach/mu
    N   = 361
    
    conditions                                     = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics() 
    conditions.altitude                            = alt
    conditions.freestream.velocity                 = velocity_freestream
    conditions.freestream.mach_number              = mach
    conditions.freestream.density                  = rho
    conditions.freestream.dynamic_viscosity        = mu
    conditions.freestream.temperature              = T
    conditions.freestream.pressure                 = P
    conditions.freestream.reynolds_number          = re
    conditions.freestream.speed_of_sound           = a
    conditions.N = N
    conditions.aerodynamics.angle_of_attack        = aoa
    
    velocity_vector                                      = np.array([[mach[0][0]*a, 0. ,0.]])
    conditions.frames.inertial.velocity_vector     = np.tile(velocity_vector,(1,1))  
    conditions.frames.body.transform_to_inertial   = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])   
    return conditions

def vehicle_setup(conditions,prop_type):
    #-----------------------------------------------------------------    
    # Vehicle Initialization:
    #-----------------------------------------------------------------
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'simple_Cessna'
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    vehicle.mass_properties.takeoff    = 2550. *Units.lbf #2550. * Units.lbf #gross weight of Cessna 172
    vehicle.reference_area             = 16  
    vehicle.passengers                 = 4
    vehicle.cruise_speed               = conditions.freestream.velocity[0][0] # 62.78 is optimal for Cessna 172
    vehicle.CL_design = vehicle.mass_properties.takeoff / (0.5*conditions.freestream.density*vehicle.cruise_speed**2*vehicle.reference_area) #Design CL is 0.708    
    
    # ------------------------------------------------------------------
    #   Wing Properties
    # ------------------------------------------------------------------       
    wing = flat_plate_wing()
    vehicle.append_component(wing) 
    
    # ------------------------------------------------------------------
    #   Propulsion Properties
    # ------------------------------------------------------------------
    net                          = SUAVE.Components.Energy.Networks.Battery_Propeller()
    net.tag                      = 'prop_net'
    net.number_of_engines        = 2
    vehicle.append_component(net)
    
    # Design the propeller once for the given cruise condition:
    if prop_type == 'small_tip_mount':
        prop      = small_prop_1(vehicle,conditions) 
    elif prop_type == 'larger_half_mount':
        prop      = prop_1(vehicle,conditions)
        
    vehicle.propulsors.prop_net.propeller  = prop    
    
    return vehicle

def prop_1(vehicle, conditions):
    # Designs the propeller to operate at specified vehicle flight conditions
    V_design = vehicle.cruise_speed
    
    # We want thrust=drag; so to specify thrust first find drag: profile drag and drag due to lift; use this for design thrust
    CL_wing = vehicle.CL_design
    AR      = vehicle.wings.main_wing.aspect_ratio
    e       = 0.7
    CD_wing = 0.012 + CL_wing**2/(np.pi*AR*e)
    Drag    = CD_wing*0.5*conditions.freestream.density*V_design**2*vehicle.reference_area 
    
    prop                            = SUAVE.Components.Energy.Converters.Propeller()
    prop.tag                        = 'Cessna_Prop' 
    
    prop.tip_radius                 = 2.8 * Units.feet #0.684 # 2.8
    prop.hub_radius                 = 0.6 * Units.feet
    prop.number_blades              = 2
    prop.disc_area                  = np.pi*(prop.tip_radius**2)
    prop.induced_hover_velocity     = 0 
    prop.freestream_velocity        = V_design*np.cos(conditions.aerodynamics.angle_of_attack[0][0])
    prop.angular_velocity           = 2200. * Units['rpm']
    prop.design_Cl                  = 0.7
    prop.design_altitude            = conditions.altitude
   
    prop.design_thrust              = Drag/vehicle.propulsors.prop_net.number_of_engines #(vehicle.mass_properties.takeoff/vehicle.net.number_of_engines)# *contingency_factor
    prop.design_power               = 0.0
    prop.thrust_angle               = 0. * Units.degrees
    prop.inputs.omega               = np.ones((1,1)) *  prop.angular_velocity
    
    prop.prop_loc                   = [2.2, -2.0, 0]
    
    #prop.airfoil_geometry          = ['NACA_4412_geo.txt','Clark_y.txt']
    #prop.airfoil_polars            = [['NACA_4412_polar_Re_50000.txt','NACA_4412_polar_Re_100000.txt',
                                      #'NACA_4412_polar_Re_200000.txt','NACA_4412_polar_Re_500000.txt',
                                      #'NACA_4412_polar_Re_1000000.txt'],
                                      #['Clark_y_polar_Re_50000.txt','Clark_y_polar_Re_100000.txt',
                                      #'Clark_y_polar_Re_200000.txt','Clark_y_polar_Re_500000.txt',
                                      #'Clark_y_polar_Re_1000000.txt']] # airfoil polars for at different reynolds numbers 
    
    ## 0 represents the first airfoil, 1 represents the second airfoil etc. 
    #prop.airfoil_polar_stations    = [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]     

    N = conditions.N
    prop                            = propeller_design(prop,N)    
    return prop


def small_prop_1(vehicle, conditions):
    # Designs the propeller to operate at specified vehicle flight conditions
    V_design = vehicle.cruise_speed
    
    # We want thrust=drag; so to specify thrust first find drag: profile drag and drag due to lift; use this for design thrust
    CL_wing = vehicle.CL_design
    AR      = vehicle.wings.main_wing.aspect_ratio
    e       = 0.7
    CD_wing = 0.012 + CL_wing**2/(np.pi*AR*e)
    Drag    = CD_wing*0.5*conditions.freestream.density*V_design**2*vehicle.reference_area 
    
    prop                            = SUAVE.Components.Energy.Converters.Propeller()
    prop.tag                        = 'Cessna_Prop' 
    
    prop.tip_radius                 = 1.1*Units.feet #2.8 * Units.feet #0.684 #
    prop.hub_radius                 = 0.2*Units.feet #0.6 * Units.feet
    prop.number_blades              = 2
    prop.disc_area                  = np.pi*(prop.tip_radius**2)
    prop.induced_hover_velocity     = 0 
    prop.freestream_velocity        = V_design*np.cos(conditions.aerodynamics.angle_of_attack[0][0])
    prop.angular_velocity           = 3500. * Units['rpm']
    prop.design_Cl                  = 0.7
    prop.design_altitude            = conditions.altitude
   
    prop.design_thrust              = (0.2/2)*Drag/vehicle.propulsors.prop_net.number_of_engines #(vehicle.mass_properties.takeoff/vehicle.net.number_of_engines)# *contingency_factor
    prop.design_power               = 0.0
    prop.thrust_angle               = 0. * Units.degrees
    prop.inputs.omega               = np.ones((1,1)) *  prop.angular_velocity
    
    prop.prop_loc                   = [2.2, -4.0, 0]
    
    #prop.airfoil_geometry          = ['NACA_4412_geo.txt'] #,'Clark_y.txt']
    #prop.airfoil_polars            = [['NACA_4412_polar_Re_50000.txt','NACA_4412_polar_Re_100000.txt',
                                      #'NACA_4412_polar_Re_200000.txt','NACA_4412_polar_Re_500000.txt',
                                      #'NACA_4412_polar_Re_1000000.txt'],
                                      #['Clark_y_polar_Re_50000.txt','Clark_y_polar_Re_100000.txt',
                                      #'Clark_y_polar_Re_200000.txt','Clark_y_polar_Re_500000.txt',
                                      #'Clark_y_polar_Re_1000000.txt']] # airfoil polars for at different reynolds numbers 
    
    ## 0 represents the first airfoil, 1 represents the second airfoil etc. 
    #prop.airfoil_polar_stations    = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]     
    
    N = conditions.N
    prop                            = propeller_design(prop,N)    
    return prop


def flat_plate_wing():
    #-------------------------------------------------------------------------
    #          Variables:
    #-------------------------------------------------------------------------    
    span         = 8
    croot        = 2
    ctip         = 2
    sweep_le     = 0.0
    sref         = span*(croot+ctip)/2    
    twist_root   = 0.0 * Units.degrees 
    twist_tip    = 0.0 * Units.degrees   
    dihedral     = 0.0 * Units.degrees
    AR           = span**2/sref
    # ------------------------------------------------------------------
    # Initialize the Main Wing
    # ------------------------------------------------------------------             
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'        
    
    wing.aspect_ratio            = AR
    wing.spans.projected         = span
    wing.chords.root             = croot
    wing.chords.tip              = ctip
    wing.areas.reference         = sref
    wing.twists.root             = twist_root
    wing.twists.tip              = twist_tip
    wing.sweeps.leading_edge     = sweep_le #45. * Units.degrees
    wing.dihedral                = dihedral #0. * Units.degrees
    wing.span_efficiency         = 0.8 
    wing.origin                  = [0.,0.,0.]
    wing.vertical                = False 
    wing.symmetric               = True
    
    # Determining Lift Curve Slope of the Wing (Anderson, Page 442)
    a0 = 2*np.pi
    if (AR>=4 and sweep_le ==0):
        a_finite_wing = a0*(AR/(2+AR))
    elif (AR<4 and sweep_le ==0):
        a_finite_wing = a0/(np.sqrt(1+((a0/(np.pi*AR))**2)) + (a0/(np.pi*AR)))
    elif (AR<4 and sweep_le>0):
        a_finite_wing = a0*np.cos(sweep_le)/(np.sqrt(1+((a0*np.cos(sweep_le))/(np.pi*AR))**2) + (a0*np.cos(sweep_le))/(np.pi*AR))
    wing.lift_curve_slope = a_finite_wing

    return wing







def baseline_propeller(vehicle,conditions):
    vehicle.propulsors.prop_net.propeller.analysis_settings.case  = 'uniform_freestream' #NOT CURRENTLY BEING PUT INTO EFFECT WHEN RUNNING THIS; NEED TO INCLUDE IN PROP.SPIN SOMEHOW
    vehicle.propulsors.prop_net.propeller.disturbed_u = np.zeros_like(vehicle.propulsors.prop_net.propeller.disturbed_u)
    vehicle.propulsors.prop_net.propeller.disturbed_w = np.zeros_like(vehicle.propulsors.prop_net.propeller.disturbed_w)    
    
    F, Q, P, Cp, outputs, etap = vehicle.propulsors.prop_net.propeller.spin(conditions,vehicle)
        
    return F, Q, P, Cp, outputs, etap

def plots_v_prop_loc(vehicle, conditions,rotation):
    prop = vehicle.propulsors.prop_net.propeller
    case = 'disturbed_freestream'
    # Determine wing influence on propeller for each point along the wing:
    prop_y_locs = np.linspace(0.25,4,30)

    thrust_vals, thrust_vals_2, thrust_vals_3  = np.zeros_like(prop_y_locs), np.zeros_like(prop_y_locs), np.zeros_like(prop_y_locs)
    torque_vals, torque_vals_2, torque_vals_3  = np.zeros_like(prop_y_locs), np.zeros_like(prop_y_locs), np.zeros_like(prop_y_locs)
    power_vals,  power_vals_2,  power_vals_3   = np.zeros_like(prop_y_locs), np.zeros_like(prop_y_locs), np.zeros_like(prop_y_locs)
    Cp_vals,     Cp_vals_2,     Cp_vals_3      = np.zeros_like(prop_y_locs), np.zeros_like(prop_y_locs), np.zeros_like(prop_y_locs)
    etap_vals,   etap_vals_2,   etap_vals_3    = np.zeros_like(prop_y_locs), np.zeros_like(prop_y_locs), np.zeros_like(prop_y_locs)
    
    V1 = 30
    V2 = 20
    V3 = 40
    
    # Adjust wing influence on propeller for each possible placement on the wing. Generate results:
    for i in range(len(prop_y_locs)):
        prop_loc         = [2.5,prop_y_locs[i],0]
        ua_wing, ut_wing = wing_effect(vehicle,prop_loc)
        prop             = include_prop_config(prop,case,ua_wing,ut_wing,rotation)   
        
        # Run Propeller model for Disturbed Freestream:
        conditions.frames.inertial.velocity_vector[0]= V1
        F, Q, P, Cp , outputs , etap = prop.spin(conditions,vehicle) #spin_simple_pusher(conditions) #prop.spin(conditions) # 
        conditions.frames.inertial.velocity_vector[0] = V2
        F_2, Q_2, P_2, Cp_2 , outputs , etap_2 = prop.spin(conditions,vehicle)#_simple_pusher(conditions) #prop.spin(conditions) #        
        conditions.frames.inertial.velocity_vector[0] = V3
        F_3, Q_3, P_3, Cp_3 , outputs , etap_3 = prop.spin(conditions,vehicle) #_simple_pusher(conditions) #prop.spin(conditions) #
        
        thrust_vals[i],   torque_vals[i],   power_vals[i],   Cp_vals[i],   etap_vals[i]   = F,   Q,   P,   Cp,   etap
        thrust_vals_2[i], torque_vals_2[i], power_vals_2[i], Cp_vals_2[i], etap_vals_2[i] = F_2, Q_2, P_2, Cp_2, etap_2
        thrust_vals_3[i], torque_vals_3[i], power_vals_3[i], Cp_vals_3[i], etap_vals_3[i] = F_3, Q_3, P_3, Cp_3, etap_3
    
    # Results from case with Uniform Flow are used as Baseline:
    conditions.frames.inertial.velocity_vector[0] = V1
    T_base, Q_base, P_base, Cp_base, outputs, etap_base = baseline_propeller(prop, conditions,vehicle)    
    T_baseline = np.ones_like(thrust_vals)*T_base[0][0] #2591.77561215
    Q_baseline = np.ones_like(thrust_vals)*Q_base[0][0] #287.4145557
    P_baseline = np.ones_like(thrust_vals)*P_base[0][0] #93303.74385
    Cp_baseline = np.ones_like(thrust_vals)*Cp_base[0][0] #0.03811682
    etap_baseline = np.ones_like(thrust_vals)*etap_base[0][0]
    
    conditions.frames.inertial.velocity_vector[0] = V2
    T_base2, Q_base2, P_base2, Cp_base2, outputs, etap_base2 = baseline_propeller(prop, conditions,vehicle)    
    T_baseline_2 = np.ones_like(thrust_vals_2)*T_base2[0][0] #2591.77561215
    Q_baseline_2 = np.ones_like(thrust_vals_2)*Q_base2[0][0] #287.4145557
    P_baseline_2 = np.ones_like(thrust_vals_2)*P_base2[0][0] #93303.74385
    Cp_baseline_2 = np.ones_like(thrust_vals_2)*Cp_base2[0][0] #0.03811682  
    etap_baseline_2 = np.ones_like(thrust_vals)*etap_base2[0][0]
    
    conditions.frames.inertial.velocity_vector[0] = V3
    T_base3, Q_base3, P_base3, Cp_base3, outputs, etap_base3 = baseline_propeller(prop, conditions,vehicle)    
    T_baseline_3 = np.ones_like(thrust_vals_3)*T_base3[0][0] #2591.77561215
    Q_baseline_3 = np.ones_like(thrust_vals_3)*Q_base3[0][0] #287.4145557
    P_baseline_3 = np.ones_like(thrust_vals_3)*P_base3[0][0] #93303.74385
    Cp_baseline_3 = np.ones_like(thrust_vals_3)*Cp_base3[0][0] #0.03811682   
    etap_baseline_3 = np.ones_like(thrust_vals)*etap_base3[0][0]
      
    
    # Parameters for generating coefficients:
    n    = prop.angular_velocity/(2*np.pi)
    D    = 2*prop.tip_radius
    rho  = conditions.freestream.density[0][0]
    
    
    half_span = 4
    #Plot of thrust v propeller spanwise location (y-locs):
    figA = plt.figure()
    axisA = figA.add_subplot(1,1,1)
    axisA.plot(prop_y_locs/half_span,thrust_vals,label='V1_disturbed')#/(rho*n**2*D**4))
    axisA.plot(prop_y_locs/half_span,T_baseline, label='V1_baseline')#/(rho*n**2*D**4))
    axisA.plot(prop_y_locs/half_span,thrust_vals_2,label='V2_disturbed')#/(rho*n**2*D**4))
    axisA.plot(prop_y_locs/half_span,T_baseline_2, label='V2_baseline')#/(rho*n**2*D**4))   
    axisA.plot(prop_y_locs/half_span,thrust_vals_3,label='V3_disturbed')#/(rho*n**2*D**4))
    axisA.plot(prop_y_locs/half_span,T_baseline_3, label='V3_baseline')#/(rho*n**2*D**4))     
    axisA.set_ylabel('CT')
    axisA.set_xlabel('Spanwise Center Propeller Location (y/0.5b)')
    axisA.set_title('Propeller Placement Effect on Thrust')
    plt.legend()
    
    figB = plt.figure()
    axisB = figB.add_subplot(1,1,1)
    axisB.plot(prop_y_locs/half_span,torque_vals, label='V1_disturbed')#/(rho*n**2*D**5))
    axisB.plot(prop_y_locs/half_span,Q_baseline, label='V1_baseline')#/(rho*n**2*D**5))
    axisB.plot(prop_y_locs/half_span,torque_vals_2, label='V2_disturbed')#/(rho*n**2*D**5))
    axisB.plot(prop_y_locs/half_span,Q_baseline_2, label='V2_baseline')#/(rho*n**2*D**5))
    axisB.plot(prop_y_locs/half_span,torque_vals_3, label='V3_disturbed')#/(rho*n**2*D**5))
    axisB.plot(prop_y_locs/half_span,Q_baseline_3, label='V3_baseline')#/(rho*n**2*D**5))    
    axisB.set_ylabel('CQ')
    axisB.set_xlabel('Spanwise Center Propeller Location (y/0.5b)')
    axisB.set_title('Propeller Placement Effect on Torque')  
    
    figC = plt.figure()
    axisC = figC.add_subplot(1,1,1)
    axisC.plot(prop_y_locs/half_span,etap_vals, label='V1_disturbed')#/(rho*n**2*D**5))
    axisC.plot(prop_y_locs/half_span,etap_baseline, label='V1_baseline')#/(rho*n**2*D**5))
    axisC.plot(prop_y_locs/half_span,etap_vals_2, label='V2_disturbed')#/(rho*n**2*D**5))
    axisC.plot(prop_y_locs/half_span,etap_baseline_2, label='V2_baseline')#/(rho*n**2*D**5))
    axisC.plot(prop_y_locs/half_span,etap_vals_3, label='V3_disturbed')#/(rho*n**2*D**5))
    axisC.plot(prop_y_locs/half_span,etap_baseline_3, label='V3_baseline')#/(rho*n**2*D**5))    
    axisC.set_ylabel('Propeller Efficiency')
    axisC.set_xlabel('Spanwise Center Propeller Location (y/0.5b)')
    axisC.set_title('Propeller Placement Effect on eta')  
    
    
    plt.legend()
    plt.show()
    
    return

def run_plots_prop_disk(conditions, vehicle):
    
    ua_wing = vehicle.propulsors.prop_net.propeller.disturbed_u
    ut_wing = vehicle.propulsors.prop_net.propeller.disturbed_w
    prop_loc = vehicle.propulsors.prop_net.propeller.prop_loc
    
    # Run Propeller model 
    F, Q, P, Cp , outputs , etap = vehicle.propulsors.prop_net.propeller.spin(conditions,vehicle) #_simple_pusher(conditions) #prop.spin(conditions) #   
    
    F_iso, Q_iso, P_iso, Cp_iso , outputs_iso , etap_iso = baseline_propeller(vehicle,conditions)
          
    
    
    psi   = outputs.azimuthal_distribution_2d[0,:,:]
    r     = outputs.blade_radial_distribution_normalized_2d[0,:,:] 

    
    # Isolated Propeller Plots:
    
    #figT=plt.figure()
    #axisT = figT.add_subplot(1,1,1)
    #axisT.plot(r[0], outputs_iso.blade_T_distribution[0], linewidth=4, label='isolated')
    #for i in range(10):
        #psi_pt = psi[i*2][0]/Units.deg
        #axisT.plot(r[0],outputs.blade_T_distribution_2d[0][i*2], label ='disturbed, $\psi=%1.0f\degree$' %psi_pt)
        
    #axisT.set_ylabel('Thrust (N)')
    #axisT.set_xlabel('Normalized Radius (r/R)')
    #axisT.set_title('Thrust Distribution along Propeller Blade')
    #plt.legend()
    #plt.grid()    
    
    #figT=plt.figure()
    #axisT = figT.add_subplot(1,1,1)
    #axisT.plot(r[0], outputs_iso.blade_Q_distribution[0], label='isolated')
    #axisT.set_ylabel('Torque (Nm)')
    #axisT.set_xlabel('Normalized Radius (r/R)')
    #axisT.set_title('Torque Distribution along Propeller Blade')
    #plt.legend()
    #plt.grid()    
    
    #figT=plt.figure()
    #axisT = figT.add_subplot(1,1,1)
    #for i in range(10):
        #psi_pt = psi[i*2][0]/Units.deg
        #axisT.plot(r[0], outputs.blade_T_distribution_2d[0][i*2]-outputs_iso.blade_T_distribution_2d[0][i*2], label ='disturbed, $\psi=%1.0f\degree$' %psi_pt)#label='loc=[%1.1f' %prop_loc[0] +',%1.1f' %prop_loc[1] + ', %1.1f]' %prop_loc[2] )
    #axisT.set_ylabel('$\Delta T$')
    #axisT.set_xlabel('Normalized Radius (r/R)')
    #axisT.set_title('Thrust Difference from Isolated Propeller')
    #plt.legend()
    #plt.grid()        

    
    
    
    #plt.show()
    
    
    
    # ----------------------------------------------------------------------------
    # DISC PLOTS   
    # ----------------------------------------------------------------------------
    Radius = vehicle.propulsors.prop_net.propeller.tip_radius
    
    # perpendicular velocity, up Plot 
    #figT=plt.figure()
    #axisT = figT.add_subplot(1,1,1)
    #axisT.plot(r[0], outputs.thrust_distribution[0], label='disturbed, loc=[%1.1f' %prop_loc[0] +',%1.1f' %prop_loc[1] + ', %1.1f]' %prop_loc[2])
    #axisT.plot(r[0], outputs_iso.thrust_distribution[0], label='isolated')
    #axisT.set_ylabel('Thrust, T (N)')
    #axisT.set_xlabel('Normalized Radius (r/R)')
    #axisT.set_title('Thrust Distribution along Propeller Blade')
    #plt.legend()
    #plt.grid()

    
    plt.figure(1)
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, -np.ones_like(outputs.axial_velocity_distribution_2d[0]) + outputs.axial_velocity_distribution_2d[0]/outputs.velocity[0][0],100,cmap=plt.cm.jet)    
    cbar0 = plt.colorbar(CS_0, ax=axis0)
    cbar0.ax.set_ylabel('$\dfrac{V_a-V_\infty}{V_\infty}$, m/s')
    axis0.set_title('Axial Inflow to Propeller')    
    
    ##plt.figure(2)
    ##fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    ##CS_0 = axis0.contourf(psi, r, (ua_wing+outputs.velocity[0][0])/outputs.velocity[0][0],100)#,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    
    ##cbar0 = plt.colorbar(CS_0, ax=axis0)
    ##cbar0.ax.set_ylabel('$\dfrac{V_a-V_\infty}{V_\infty}$, m/s')
    ##axis0.set_title('Axial Velocity at Propeller (Inviscid)')
    
    #plt.figure(3)
    #fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_0 = axis0.contourf(psi, r, (ut_wing),100,cmap=plt.cm.jet)#(np.array(ut_wing)+2200*Units.feet)/(2200*Units.feet),100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    # -np.pi+psi turns it 
    #cbar0 = plt.colorbar(CS_0, ax=axis0)
    #cbar0.ax.set_ylabel('$\dfrac{V_t}{V_\infty}$, m/s')
    #axis0.set_title('Downwash Velocity at Propeller')    
    
    plt.figure(2)
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, outputs.thrust_distribution_2d[0],100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    # -np.pi+psi turns it 
    cbar0 = plt.colorbar(CS_0, ax=axis0)
    cbar0.ax.set_ylabel('Thrust (N)')
    axis0.set_title('Thrust Distribution of Propeller')  
    
    plt.figure(3)
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, outputs.torque_distribution_2d[0],100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    # -np.pi+psi turns it 
    cbar0 = plt.colorbar(CS_0, ax=axis0)
    cbar0.ax.set_ylabel('Torque (Nm)')
    axis0.set_title('Torque Distribution of Propeller') 
    
    #plt.figure(6)
    #fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_0 = axis0.contourf(psi, r, (outputs.thrust_distribution_2d[0]-outputs_iso.thrust_distribution_2d[0])/outputs_iso.thrust_distribution_2d[0],100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    # -np.pi+psi turns it 
    #cbar0 = plt.colorbar(CS_0, ax=axis0)
    #cbar0.ax.set_ylabel('Thrust (N)')
    #axis0.set_title('Percent Change in Thrust')  
    
    #plt.figure(7)
    #fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_0 = axis0.contourf(psi, r, (outputs.torque_distribution_2d[0]-outputs_iso.torque_distribution_2d[0])/outputs_iso.torque_distribution_2d[0],100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    # -np.pi+psi turns it 
    #cbar0 = plt.colorbar(CS_0, ax=axis0)
    #cbar0.ax.set_ylabel('Torque (Nm)')
    #axis0.set_title('Percent Change in Torque')     
    
    
    #N=20
    #psi          = np.linspace(0,2*np.pi,N)
    #psi_2d       = np.tile(np.atleast_2d(psi).T,(1,N))  
    

    
    #Cpplot = 1-(1+outputs.va_2d[0,:,:])**2#1-((outputs.va_2d[0,:,:]-outputs.velocity[0][0])/outputs.velocity[0][0])**2
    
    #plt.figure(9)
    #fig8, axis8 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_8 = axis8.contourf(psi, r, Cpplot,100)#,cmap=plt.cm.jet)    
    #cbar8 = plt.colorbar(CS_8, ax=axis8)
    #cbar8.ax.set_ylabel('Pressure (Axial)')
    #axis8.set_title('Pressure Distribution on Propeller')
    
    #z = (1-((outputs.vt_2d[0,:,:])/U[0,:,:])**2)
    #plt.figure(10)
    #fig9, axis9 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_9 = axis9.contourf(psi, r, z,100)#,cmap=plt.cm.jet)    
    #cbar9 = plt.colorbar(CS_9, ax=axis9)
    #cbar9.ax.set_ylabel('Pressure Calculated') 
    
    
    #plt.show() 
    
    return F, F_iso, Q, Q_iso, etap, etap_iso


def wing_effect(vehicle,conditions):
    #-------------------------------------------------------------------------
    #          Extracting variables:
    #-------------------------------------------------------------------------
    R_tip    = vehicle.propulsors.prop_net.propeller.tip_radius
    r_hub    = vehicle.propulsors.prop_net.propeller.hub_radius    
    prop_loc = vehicle.propulsors.prop_net.propeller.prop_loc
    N        = conditions.N

    # --------------------------------------------------------------------------------
    #          Settings and Calling for VLM:  
    # --------------------------------------------------------------------------------
    vortices            = 10
    VLM_settings        = Data()
    VLM_settings.number_panels_spanwise   = 1#vortices **2
    VLM_settings.number_panels_chordwise  = vortices

    VLM_outputs   = wing_VLM(vehicle, conditions, VLM_settings)
    VD            = copy.deepcopy(VLM_outputs.VD)

    #------------------------------------------------------------------------------------
    #         Computing velocity = velocity(radius, blade angle):
    #------------------------------------------------------------------------------------
    r_R_min = r_hub/R_tip
    phi     = np.linspace(0,2*np.pi,N)
    r2d     = np.linspace(r_R_min, 0.96, N)*R_tip
    
    prop_x_center = np.array([vehicle.wings.main_wing.origin[0] + prop_loc[0]])
    prop_y_center = np.array([prop_loc[1]])
    prop_z_center = np.array([prop_loc[2]])

    u_pts =  [[0 for j in range(len(r2d))] for x in range(len(phi))]
    v_pts =  [[0 for j in range(len(r2d))] for x in range(len(phi))]
    w_pts =  [[0 for j in range(len(r2d))] for x in range(len(phi))]
    u_pts2 = [[0 for j in range(len(r2d))] for x in range(len(phi))]
    v_pts2 = [[0 for j in range(len(r2d))] for x in range(len(phi))]
    w_pts2 = [[0 for j in range(len(r2d))] for x in range(len(phi))]

    for i in range(len(phi)):                                        
        # Outputs from VLM_velocity_sweep are ua_wing[[u(r1,phi1), ... u(rn,phi1)], ... [u(r1,phin), ... u(rn,phin)]]]
        for k in range(len(r2d)):
            yloc = np.array([prop_y_center + r2d[k]*np.cos(phi[i])])
            zloc = np.array([prop_z_center + r2d[k]*np.sin(phi[i])])
            be_locx = [prop_x_center, yloc, zloc]
            Cmnx, uk, vk, wk, propvalk = VLM_velocity_sweep(conditions,VLM_outputs,be_locx,VLM_settings)

            if wk>1 or wk<-1 or uk>1 or uk<-1:
                w_pts[i][k] = w_pts[i][k-1]
                u_pts[i][k] = u_pts[i][k-1]
                v_pts[i][k] = v_pts[i][k-1]

            else:
                u_pts[i][k] = uk[0]
                v_pts[i][k] = vk[0]
                w_pts[i][k] = wk[0]   
    vehicle.propulsors.prop_net.propeller.disturbed_u = u_pts
    vehicle.propulsors.prop_net.propeller.disturbed_w = w_pts
                           
    return vehicle




def wing_VLM(vehicle,conditions, VLM_settings):
    # --------------------------------------------------------------------------------
    #          Get Vehicle Geometry and Unpaack Settings:  
    # --------------------------------------------------------------------------------    
    Sref         = vehicle.wings.main_wing.areas.reference
    n_sw         = VLM_settings.number_panels_spanwise
    n_cw         = VLM_settings.number_panels_chordwise

    aoa          = conditions.aerodynamics.angle_of_attack   # angle of attack  
    mach         = conditions.freestream.mach_number         # mach number
    
    ones         = np.atleast_2d(np.ones_like(aoa)) 
    
    # --------------------------------------------------------------------------------
    #          Generate Vortex Distribution and Build Induced Velocity Matrix:
    # --------------------------------------------------------------------------------     
    VD                          = compute_vortex_distribution(vehicle,VLM_settings)
    vehicle.vortex_distribution = VD
    C_mn, DW_mn                 = compute_induced_velocity_matrix(VD,n_sw,n_cw,aoa,mach)
    MCM                         = VD.MCM 
    
    # --------------------------------------------------------------------------------
    #          Compute Flow Tangency Conditions:
    # --------------------------------------------------------------------------------     
    inv_root_beta           = np.zeros_like(mach)
    inv_root_beta[mach<1]   = 1/np.sqrt(1-mach[mach<1]**2)     
    inv_root_beta[mach>1]   = 1/np.sqrt(mach[mach>1]**2-1) 
    if np.any(mach==1):
        raise('Mach of 1 cannot be used in building compressibiliy corrections.')
    inv_root_beta = np.atleast_2d(inv_root_beta)
    
    phi      = np.arctan((VD.ZBC - VD.ZAC)/(VD.YBC - VD.YAC))*ones          # dihedral angle 
    delta    = np.arctan((VD.ZC - VD.ZCH)/((VD.XC - VD.XCH)*inv_root_beta)) # mean camber surface angle 
   
    # --------------------------------------------------------------------------------
    #          Build Aerodynamic Influence Coefficient Matrix
    # --------------------------------------------------------------------------------    
    A =   np.multiply(C_mn[:,:,:,0],np.atleast_3d(np.sin(delta)*np.cos(phi))) \
        + np.multiply(C_mn[:,:,:,1],np.atleast_3d(np.cos(delta)*np.sin(phi))) \
        - np.multiply(C_mn[:,:,:,2],np.atleast_3d(np.cos(phi)*np.cos(delta)))   # valdiated from book eqn 7.42 
    
    B =   np.multiply(DW_mn[:,:,:,0],np.atleast_3d(np.sin(delta)*np.cos(phi))) \
        + np.multiply(DW_mn[:,:,:,1],np.atleast_3d(np.cos(delta)*np.sin(phi))) \
        - np.multiply(DW_mn[:,:,:,2],np.atleast_3d(np.cos(phi)*np.cos(delta)))   # valdiated from book eqn 7.42     
   
    # Build the RHS vector
    RHS = compute_RHS_matrix(n_sw,n_cw,delta,phi,conditions,vehicle,True,False)

    # Compute vortex strength  
    n_cp  = VD.n_cp  
    gamma = np.linalg.solve(A,RHS)
    GAMMA = np.repeat(np.atleast_3d(gamma), n_cp ,axis = 2 )
    u = np.sum(C_mn[:,:,:,0]*MCM[:,:,:,0]*GAMMA, axis = 2) 
    v = np.sum(C_mn[:,:,:,1]*MCM[:,:,:,1]*GAMMA, axis = 2) 
    w = np.sum(C_mn[:,:,:,2]*MCM[:,:,:,2]*GAMMA, axis = 2) 
    w_ind = -np.sum(B*MCM[:,:,:,2]*GAMMA, axis = 2) 
     
    # ---------------------------------------------------------------------------------------
    #           Compute aerodynamic coefficients 
    # ---------------------------------------------------------------------------------------  
    n_cppw     = n_sw*n_cw
    n_w        = VD.n_w
    CS         = VD.CS*ones
    wing_areas = VD.wing_areas
    CL_wing    = np.zeros(n_w)
    CDi_wing   = np.zeros(n_w)
    
    Del_Y = np.abs(VD.YB1 - VD.YA1)*ones
    
    # Linspace out where breaks are
    wing_space = np.linspace(0,n_cppw*n_w,n_w+1)
    
    # Use split to divide u, w, gamma, and Del_y into more arrays
    u_n_w        = np.array(np.array_split(u,n_w,axis=1))
    u_n_w_sw     = np.array(np.array_split(u,n_w*n_sw,axis=1))
    w_n_w        = np.array(np.array_split(w,n_w,axis=1))
    w_n_w_sw     = np.array(np.array_split(w,n_w*n_sw,axis=1))
    w_ind_n_w    = np.array(np.array_split(w_ind,n_w,axis=1))
    w_ind_n_w_sw = np.array(np.array_split(w_ind,n_w*n_sw,axis=1))    
    gamma_n_w    = np.array(np.array_split(gamma,n_w,axis=1))
    gamma_n_w_sw = np.array(np.array_split(gamma,n_w*n_sw,axis=1))
    Del_Y_n_w    = np.array(np.array_split(Del_Y,n_w,axis=1))
    Del_Y_n_w_sw = np.array(np.array_split(Del_Y,n_w*n_sw,axis=1))
    
    # Calculate the Coefficients on each wing individually
    L_wing   = np.sum(np.multiply(u_n_w+1,(gamma_n_w*Del_Y_n_w)),axis=2).T
    CL_wing  = L_wing/wing_areas
    Di_wing  = np.sum(np.multiply(-w_ind_n_w,(gamma_n_w*Del_Y_n_w)),axis=2).T
    CDi_wing = Di_wing/wing_areas
    
    # Calculate each spanwise set of Cls and Cds
    cl_y = np.sum(np.multiply(u_n_w_sw +1,(gamma_n_w_sw*Del_Y_n_w_sw)),axis=2).T/CS
    cdi_y = np.sum(np.multiply(-w_ind_n_w_sw,(gamma_n_w_sw*Del_Y_n_w_sw)),axis=2).T/CS
    
    # Split the Cls and Cds for each wing
    Cl_wings = np.array(np.split(cl_y,n_w,axis=1))
    Cd_wings = np.array(np.split(cdi_y,n_w,axis=1))
            
    # total lift and lift coefficient
    L  = np.atleast_2d(np.sum(np.multiply((1+u),gamma*Del_Y),axis=1)).T 
    CL = L/(0.5*Sref)           # validated form page 402-404, aerodynamics for engineers # supersonic lift off by 2^3 
    CL[mach>1] = CL[mach>1]*2*4 # supersonic lift off by a factor of 4 
    
    # total drag and drag coefficient
    D  =   -np.atleast_2d(np.sum(np.multiply(w_ind,gamma*Del_Y),axis=1)).T   
    CDi = D/(0.5*Sref)  
    CDi[mach>1] = CDi[mach>1]*2
    
    # pressure coefficient
    U_tot = np.sqrt((1+u)*(1+u) + v*v + w*w)
    CP = 1 - (U_tot)*(U_tot)
    
    # delete MCM from VD data structure since it consumes memory
    delattr(VD, 'MCM')
    # ---------------------------------------------------------------------------------------
    #          Comparisons to Expectations:
    # ---------------------------------------------------------------------------------------
    AR                   = vehicle.wings.main_wing.spans.projected/vehicle.wings.main_wing.chords.root
    CL_flat_plate        = 2*np.pi*aoa*(AR/(2+AR))
    
    elliptical_downwash  = CL/(np.pi*AR)
    actual_downwash      = np.arctan(w[0][0]/(conditions.freestream.velocity))*180/np.pi
    
    
    # ---------------------------------------------------------------------------------------
    #  Specify function outputs: 
    # ---------------------------------------------------------------------------------------
    VLM_outputs       = Data()
    VLM_outputs.VD    = VD
    VLM_outputs.MCM   = MCM
    VLM_outputs.C_mn  = C_mn
    VLM_outputs.gamma = GAMMA
    VLM_outputs.cl_y  = cl_y
    VLM_outputs.CL    = CL
    VLM_outputs.CDi   = CDi

    return VLM_outputs

def CL_downwash_validation(vehicle, state):
    #-------------------------------------------------------------------------------------------
    #          Range of Angle of Attack:
    #------------------------------------------------------------------------------------------- 
    aoa_vec         = np.linspace(-2,8,20) * Units.deg
    N               = 16
    prop_x          = np.array([2.0])
    prop_y          = np.linspace(0,4,N)
    prop_y2          = np.linspace(0.01,3.4,N)
    prop_z          = np.array([0.0])
    
    vortices   = 10
    n_sw       = 1#vortices **2
    n_cw       = vortices

    # --------------------------------------------------------------------------------
    #          Settings and Calling for VLM:  
    # --------------------------------------------------------------------------------
    VLM_settings        = Data()
    VLM_settings.number_panels_spanwise   = n_sw
    VLM_settings.number_panels_chordwise   = n_cw    
    #-------------------------------------------------------------------------------------------
    #          CL Plots for Validating VLM:
    #------------------------------------------------------------------------------------------- 
    AR              = vehicle.wings.main_wing.aspect_ratio
    e               = 0.7
    CL_vec          = np.zeros((len(aoa_vec),len(prop_y)))
    CDi_vec         = np.zeros((len(aoa_vec),len(prop_y)))
    w_vlm           = np.zeros((len(aoa_vec),len(prop_y2)))
    CL_flat_plate   = 2*np.pi*aoa_vec*(AR/(2+AR))
    CDi_flat_plate  = CL_flat_plate**2/(np.pi*e*AR)
    state_new       = state
    for i in range(len(aoa_vec)):
        for a in range(len(prop_y)):
            prop_loc = [prop_x, np.array([prop_y[a]]),prop_z]
            # Set the angle of attack for the new set of state conditions:
            aoa_val                                           = np.array([[aoa_vec[i]]])
            state_new.conditions.aerodynamics.angle_of_attack = aoa_val
            
            # Generate the VLM results for this state condition:
            VD, MCM, C_mn, gammaT, cl_y_out, CL_out, CDi_out  = wing_VLM(vehicle,state_new,VLM_settings)
            CL_vec[i,a]                                       = CL_out
            CDi_vec[i,a]                                      = CDi_out
            
        for b in range(len(prop_y2)):
            # Determine the induced velocities at the selected location for this state condition:
            prop_loc = [prop_x, np.array([prop_y2[b]]),prop_z]
            C_mn_2, u,v,w,prop_val  = VLM_velocity_sweep(VD,prop_loc,VLM_settings.n_sw,VLM_settings.n_cw,aoa_val,state.conditions.freestream.mach_number,C_mn,MCM,gammaT)            
            w_vlm[i,b]              = w
        
    w_momentum_theory = (2*CL_flat_plate*vehicle.wings.main_wing.areas.reference)/(np.pi*vehicle.wings.main_wing.spans.projected**2)
    aoa_vec           = aoa_vec*180/np.pi
    w_vlm_avg         = np.mean(-w_vlm, axis=1)
    CL_vec            = np.mean(CL_vec, axis=1) #CL_vec[:,0]
    CDi_vec           = np.mean(CDi_vec, axis=1)
    w_momentum_theory = (2*CL_vec*vehicle.wings.main_wing.areas.reference)/(np.pi*vehicle.wings.main_wing.spans.projected**2)
    
    
    #Plotting CL distribution of VLM v. Theory:
    fig_val1  = plt.figure()
    axes_val1 = fig_val1.add_subplot(1,1,1)
    axes_val1.plot(aoa_vec, CL_flat_plate,'--', linewidth=4, label="Flat Plate Theory")
    axes_val1.plot(aoa_vec, CL_vec,'r', label="Flat Plate VLM Result")
    axes_val1.set_xlabel("Angle of Attack (deg)")
    axes_val1.set_ylabel("CL")
    axes_val1.set_title("CL v Alpha, Flat Plate")
    plt.legend()
    plt.grid()
    
    
    #Plotting downwash of VLM v. Momentum Theory: 
    fig_val2 = plt.figure()
    axes_val2 = fig_val2.add_subplot(1,1,1)
    axes_val2.plot(aoa_vec, w_momentum_theory, '--', linewidth=4, label="Momentum Theory")
    axes_val2.plot(aoa_vec, w_vlm_avg,'r', label="Averaged VLM Result")
    axes_val2.set_xlabel("Angle of Attack (deg)")
    axes_val2.set_ylabel("Downwash Velocity")
    axes_val2.set_title("Downwash v Alpha, Flat Plate")  
    
    plt.legend()
    plt.grid()    
    
    
    fig_val3  = plt.figure()
    axes_val3 = fig_val3.add_subplot(1,1,1)
    axes_val3.plot(aoa_vec, CDi_flat_plate,'--', linewidth=4, label="Flat Plate Theory")
    axes_val3.plot(aoa_vec, CDi_vec,'r', label="Flat Plate VLM Result")
    axes_val3.set_xlabel("Angle of Attack (deg)")
    axes_val3.set_ylabel("CDi")
    axes_val3.set_title("CDi v Alpha, Flat Plate")
    plt.legend()
    plt.grid()  
    
    plt.show() 
    
    return



def VLM_velocity_sweep(conditions, VLM_outputs, prop_location, VLM_settings):
    VD     = VLM_outputs.VD
    gammaT = VLM_outputs.gamma
    MCM    = VLM_outputs.MCM
    C_mn   = VLM_outputs.C_mn
    
    aoa    = conditions.aerodynamics.angle_of_attack
    mach   = conditions.freestream.mach_number
    n_sw   = VLM_settings.number_panels_spanwise
    n_cw   = VLM_settings.number_panels_chordwise
    
    # prop_location is a vector of vectors representing x, y, and z propeller locations
    # we want a pairing of all possible (x,y,z) points to be represented 
    cp_x = np.zeros(len(prop_location[0])*len(prop_location[1])*len(prop_location[2]))
    cp_y = np.zeros(len(cp_x))
    cp_z = np.zeros(len(cp_x))
    prop_val = np.array([[0.0 for i in range(3)] for j in range(len(cp_x))])
    count = 0
    for x in range(len(prop_location[0])):
        for y in range(len(prop_location[1])):
            for z in range(len(prop_location[2])):
                cp_x[count] = prop_location[0][x]
                cp_y[count] = prop_location[1][y]
                cp_z[count] = prop_location[2][z]
                prop_val[count] = [cp_x[count],cp_y[count],cp_z[count]]
                count = count + 1
                
    max_val_per_loop    = len(C_mn[0])
    num_pts_of_interest = len(cp_x)
    
    u  = np.zeros(num_pts_of_interest)
    v  = np.zeros(num_pts_of_interest)
    w  = np.zeros(num_pts_of_interest)
    
    count  = 0
    num_loops_required = math.ceil(num_pts_of_interest/max_val_per_loop)
    remainder = num_pts_of_interest%max_val_per_loop
    for i in range(num_loops_required):
        #if i == num_loops_required-1:
            #max_val_per_loop = remainder        
        # Loop through 250 of the points and keep track of the count
        VD.XC = cp_x[count:count+max_val_per_loop]
        VD.YC = cp_y[count:count+max_val_per_loop]
        VD.ZC = cp_z[count:count+max_val_per_loop]
        # Build new induced velocity matrix, C_mn
        C_mn, DW_mn = compute_induced_velocity_matrix(VD,n_sw,n_cw,aoa,mach)
        MCM = VD.MCM
        # Compute induced velocities at control point from all panel influences
        u[count:count+max_val_per_loop] = (C_mn[:,:,:,0]*MCM[:,:,:,0]@gammaT)[:,:,0]
        v[count:count+max_val_per_loop] = (C_mn[:,:,:,1]*MCM[:,:,:,1]@gammaT)[:,:,0]
        w[count:count+max_val_per_loop] = (C_mn[:,:,:,2]*MCM[:,:,:,2]@gammaT)[:,:,0]  
        
        count += max_val_per_loop
    check = (np.linspace(1,len(u)-1,len(u)-1))
    for val in check:
        j = int(val)
        if u[j] >= 1 or u[j] <= -1:
            u[j] = u[j-1]
        if v[j] >= 1 or v[j] <= -1:
            v[j] = v[j-1]       
        if w[j] >= 1 or w[j] <= -1:
            w[j] = w[j-1]        
            
    return C_mn, u, v, w, prop_val #C_mn, u[0,0], v[0,0], w[0,0]


if __name__ == '__main__': 
    main()    
    plt.show()   