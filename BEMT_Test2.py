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
    '''
    In this test case, input data (propeller and flight conditions) is obtained from drone acoustic tests by Christian in "Auralization of tonal noise .."
    '''
    #-----------------------------------------------------------------
    # Specify the propeller attributes and setup the vehicle:
    #-----------------------------------------------------------------
    case         = 'disturbed_freestream' #'uniform_freestream' # 'disturbed_freestream' #'uniform_freestream'
    rotation     = 'ccw'
    cruise_speed = 50 # 80 gives CL of 0.270, 45 gives CL of 0.854
    state        = cruise_conditions(cruise_speed)  # N is in here as 30
    V_design     = state.conditions.freestream.velocity # 40
    state.conditions.test_BET  = True

    vehicle      = vehicle_setup(state.conditions)
   
    #-----------------------------------------------------------------
    # Compute and include wing influence at prop location in prop definition:
    #-----------------------------------------------------------------
    prop_loc         = [2.25,-4.0,0]
    
    ua_wing, ut_wing = wing_effect(vehicle, prop_loc)  
    prop             = include_prop_config(vehicle.propulsors.prop_net.propeller,case,ua_wing,ut_wing,rotation)  
    vehicle.propulsors.prop_net.propeller    = prop
    vehicle.propulsors.prop_net.propeller.prop_loc = prop_loc
    
    #-------------------------------------------------------------------    
    # Generate plots of thrust, torque, etc. on propeller disk:
    #-------------------------------------------------------------------           
    run_plots_prop_disk(case, rotation, state.conditions, vehicle)
    
    ##-------------------------------------------------------------------
    ## Generate 2D plots of thrust, torque, power, etc. versus propeller y-location: ( Isolated Propeller Case )
    ##-------------------------------------------------------------------
    #plots_v_prop_loc(vehicle,state.conditions,rotation)

    return

def cruise_conditions(cruise_speed):
    # --------------------------------------------------------------------------------    
    #          Cruise conditions  
    # --------------------------------------------------------------------------------
    # Calculated for altitude of 4000 m (13000ft), typical altitude of a Cessna (max is 14000ft)
    rho                  = np.array([[0.81934611]])
    mu                   = np.array([[1.66118986e-05]])
    T                    = np.array([[262.16631373]])
    a                    = np.array([[np.sqrt(1.4*287.058*T[0][0])]])
    velocity_freestream  = np.array([[cruise_speed]]) #m/s this is the velocity at which the propeller is designed for based on the CLwing value being near 0.7
    mach                 = velocity_freestream/a #np.array([[0.8]]) #
    pressure             = np.array([[61660.37789389]])
    re                   = rho*a*mach/mu
    N  = 20
    
    state                                                = SUAVE.Analyses.Mission.Segments.Conditions.State()
    state.conditions                                     = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()  
    state.conditions.freestream.velocity                 = velocity_freestream
    state.conditions.freestream.mach_number              = mach
    state.conditions.freestream.density                  = rho
    state.conditions.freestream.dynamic_viscosity        = mu
    state.conditions.freestream.temperature              = T
    state.conditions.freestream.pressure                 = pressure
    state.conditions.freestream.reynolds_number          = re
    state.conditions.freestream.speed_of_sound           = a
    state.conditions.N = N
    
    velocity_vector                                      = np.array([[mach[0][0]*a[0][0], 0. ,0.]])
    state.conditions.frames.inertial.velocity_vector     = np.tile(velocity_vector,(1,1))  
    state.conditions.frames.body.transform_to_inertial   = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])   
    return state

def vehicle_setup(conditions):
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
    
    prop                                   = prop_1(vehicle,conditions) # Designs the propeller once for the given cruise condition:
    vehicle.propulsors.prop_net.propeller  = prop    
    
    return vehicle
    




def baseline_propeller(prop, conditions):
    case  = 'uniform_freestream'
    prop  = include_prop_config(prop,case,prop.prop_configs.wing_ua,prop.prop_configs.wing_ut,prop.prop_configs.rotation) 
    
    F, Q, P, Cp, outputs, etap = prop.spin(conditions) #spin_simple_pusher(conditions)
        
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
        F, Q, P, Cp , outputs , etap = prop.spin(conditions) #spin_simple_pusher(conditions) #prop.spin(conditions) # 
        conditions.frames.inertial.velocity_vector[0] = V2
        F_2, Q_2, P_2, Cp_2 , outputs , etap_2 = prop.spin(conditions)#_simple_pusher(conditions) #prop.spin(conditions) #        
        conditions.frames.inertial.velocity_vector[0] = V3
        F_3, Q_3, P_3, Cp_3 , outputs , etap_3 = prop.spin(conditions) #_simple_pusher(conditions) #prop.spin(conditions) #
        
        thrust_vals[i],   torque_vals[i],   power_vals[i],   Cp_vals[i],   etap_vals[i]   = F,   Q,   P,   Cp,   etap
        thrust_vals_2[i], torque_vals_2[i], power_vals_2[i], Cp_vals_2[i], etap_vals_2[i] = F_2, Q_2, P_2, Cp_2, etap_2
        thrust_vals_3[i], torque_vals_3[i], power_vals_3[i], Cp_vals_3[i], etap_vals_3[i] = F_3, Q_3, P_3, Cp_3, etap_3
    
    # Results from case with Uniform Flow are used as Baseline:
    conditions.frames.inertial.velocity_vector[0] = V1
    T_base, Q_base, P_base, Cp_base, outputs, etap_base = baseline_propeller(prop, conditions)    
    T_baseline = np.ones_like(thrust_vals)*T_base[0][0] #2591.77561215
    Q_baseline = np.ones_like(thrust_vals)*Q_base[0][0] #287.4145557
    P_baseline = np.ones_like(thrust_vals)*P_base[0][0] #93303.74385
    Cp_baseline = np.ones_like(thrust_vals)*Cp_base[0][0] #0.03811682
    etap_baseline = np.ones_like(thrust_vals)*etap_base[0][0]
    
    conditions.frames.inertial.velocity_vector[0] = V2
    T_base2, Q_base2, P_base2, Cp_base2, outputs, etap_base2 = baseline_propeller(prop, conditions)    
    T_baseline_2 = np.ones_like(thrust_vals_2)*T_base2[0][0] #2591.77561215
    Q_baseline_2 = np.ones_like(thrust_vals_2)*Q_base2[0][0] #287.4145557
    P_baseline_2 = np.ones_like(thrust_vals_2)*P_base2[0][0] #93303.74385
    Cp_baseline_2 = np.ones_like(thrust_vals_2)*Cp_base2[0][0] #0.03811682  
    etap_baseline_2 = np.ones_like(thrust_vals)*etap_base2[0][0]
    
    conditions.frames.inertial.velocity_vector[0] = V3
    T_base3, Q_base3, P_base3, Cp_base3, outputs, etap_base3 = baseline_propeller(prop, conditions)    
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

def run_plots_prop_disk(case, rotation, conditions, vehicle):
    ua_wing = vehicle.propulsors.prop_net.propeller.prop_configs.wing_ua
    ut_wing = vehicle.propulsors.prop_net.propeller.prop_configs.wing_ut
    prop_loc = vehicle.propulsors.prop_net.propeller.prop_loc
    # Run Propeller model 
    F, Q, P, Cp , outputs , etap = vehicle.propulsors.prop_net.propeller.spin(conditions) #_simple_pusher(conditions) #prop.spin(conditions) #   
    

    vehicle.propulsors.prop_net.propeller = include_prop_config(vehicle.propulsors.prop_net.propeller,'uniform_freestream', ua_wing,ut_wing,'ccw')
    F_iso, Q_iso, P_iso, Cp_iso , outputs_iso , etap_iso = vehicle.propulsors.prop_net.propeller.spin(conditions) #_simple_pusher(conditions) #prop.spin(conditions) #      
    
    
    psi   = outputs.azimuthal_distribution_2d[0,:,:]
    r     = outputs.radius_distribution_2d[0,:,:] 

    
    # Isolated Propeller Plots:
    
    figT=plt.figure()
    axisT = figT.add_subplot(1,1,1)
    axisT.plot(r[0], outputs_iso.blade_T_distribution[0], linewidth=4, label='isolated')
    for i in range(10):
        psi_pt = psi[i*2][0]/Units.deg
        axisT.plot(r[0],outputs.blade_T_distribution_2d[0][i*2], label ='disturbed, $\psi=%1.0f\degree$' %psi_pt)
        
    axisT.set_ylabel('Thrust (N)')
    axisT.set_xlabel('Normalized Radius (r/R)')
    axisT.set_title('Thrust Distribution along Propeller Blade')
    plt.legend()
    plt.grid()    
    
    figT=plt.figure()
    axisT = figT.add_subplot(1,1,1)
    axisT.plot(r[0], outputs_iso.blade_Q_distribution[0], label='isolated')
    axisT.set_ylabel('Torque (Nm)')
    axisT.set_xlabel('Normalized Radius (r/R)')
    axisT.set_title('Torque Distribution along Propeller Blade')
    plt.legend()
    plt.grid()    
    
    figT=plt.figure()
    axisT = figT.add_subplot(1,1,1)
    for i in range(10):
        psi_pt = psi[i*2][0]/Units.deg
        axisT.plot(r[0], outputs.blade_T_distribution_2d[0][i*2]-outputs_iso.blade_T_distribution_2d[0][i*2], label ='disturbed, $\psi=%1.0f\degree$' %psi_pt)#label='loc=[%1.1f' %prop_loc[0] +',%1.1f' %prop_loc[1] + ', %1.1f]' %prop_loc[2] )
    axisT.set_ylabel('$\Delta T$')
    axisT.set_xlabel('Normalized Radius (r/R)')
    axisT.set_title('Thrust Difference from Isolated Propeller')
    plt.legend()
    plt.grid()        

    
    
    
    plt.show()
    
    
    
    # ----------------------------------------------------------------------------
    # DISC PLOTS   
    # ----------------------------------------------------------------------------
    Radius = vehicle.propulsors.prop_net.propeller.tip_radius
    
    # perpendicular velocity, up Plot 
    figT=plt.figure()
    axisT = figT.add_subplot(1,1,1)
    axisT.plot(r[0], outputs.blade_T_distribution[0], label='disturbed, loc=[%1.1f' %prop_loc[0] +',%1.1f' %prop_loc[1] + ', %1.1f]' %prop_loc[2])
    axisT.plot(r[0], outputs_iso.blade_T_distribution[0], label='isolated')
    axisT.set_ylabel('Thrust, T (N)')
    axisT.set_xlabel('Normalized Radius (r/R)')
    axisT.set_title('Thrust Distribution along Propeller Blade')
    plt.legend()
    plt.grid()

    
    f2 = plt.figure(1)
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, (outputs.va_2d[0]-outputs.velocity[0,0])/outputs.velocity[0,0],100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    
    cbar0 = plt.colorbar(CS_0, ax=axis0)
    cbar0.ax.set_ylabel('Ua_wing, m/s')
    axis0.set_title('Axial Inflow to Propeller')    
    
    plt.figure(2)
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, ua_wing,100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    
    cbar0 = plt.colorbar(CS_0, ax=axis0)
    cbar0.ax.set_ylabel('$\dfrac{V_a-V_\infty}{V_\infty}$, m/s')
    axis0.set_title('Axial Velocity at Propeller')
    
    plt.figure(3)
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, np.array(ut_wing),100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    # -np.pi+psi turns it 
    cbar0 = plt.colorbar(CS_0, ax=axis0)
    cbar0.ax.set_ylabel('$\dfrac{V_t}{V_\infty}$, m/s')
    axis0.set_title('Tangential Velocity at Propeller')    
    
    u_wing_tot = np.sqrt(np.square(ua_wing+np.ones_like(ua_wing))+np.square(ut_wing))-np.ones_like(ua_wing)
    plt.figure(4)
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, u_wing_tot,100)#,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    
    cbar0 = plt.colorbar(CS_0, ax=axis0)
    cbar0.ax.set_ylabel('$\dfrac{V-V_\infty}{V_\infty}$, m/s')
    axis0.set_title('Velocity at Propeller, $\dfrac{y}{b}=-\dfrac{1}{2}$ ')      
    
    plt.figure(3)
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(psi, r, outputs.blade_T_distribution_2d[0],100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    # -np.pi+psi turns it 
    cbar0 = plt.colorbar(CS_0, ax=axis0)
    cbar0.ax.set_ylabel('Thrust (N)')
    axis0.set_title('Thrust Distribution of Propeller')      
    
    
    #plt.figure(1)
    #fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_0 = axis0.contourf(psi, r, (outputs.va_2d[0,:,:]-outputs.velocity[0][0])/outputs.velocity[0][0],100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)    
    #cbar0 = plt.colorbar(CS_0, ax=axis0)
    #cbar0.ax.set_ylabel('(Ua-Vinf)/Vinf , m/s')
    #axis0.set_title('Axial Inflow to Propeller')    
    
    
    ## tangentIal velocity, ut Plot  
    #N=20
    #psi          = np.linspace(0,2*np.pi,N)
    #psi_2d       = np.tile(np.atleast_2d(psi).T,(1,N))  
    
    #utang_wing = np.array(ut_wing)*np.cos(psi_2d)    
    #omega_2d_y
    
    #ut_dist_pts = (utang_wing+outputs.omega_2d_y[0])/outputs.velocity[0][0]
    #ut_dist_avg = np.average((utang_wing+outputs.omega_2d_y[0])/outputs.velocity[0][0],axis=0)
    #relative_ut = ut_dist_pts/ut_dist_avg    
    
    
    
    #plt.figure(2)
    #fig1, axis1 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_1 = axis1.contourf(psi, r, relative_ut,100,cmap=plt.cm.jet) #(utang_wing+outputs.omega_2d_y[0,:,:])/outputs.omega_2d_y[0],100) #outputs.velocity[0][0],100)#,cmap=plt.cm.jet)    
    #cbar1 = plt.colorbar(CS_1, ax=axis1)
    #cbar1.ax.set_ylabel('Ut/Vinf , m/s')
    #axis1.set_title('Tangential Velocity at Propeller')
    
    ## total velocity, U plot
    #plt.figure(3)
    #U = np.sqrt(outputs.va_2d**2 + outputs.vt_2d**2)
    #fig2, axis2 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_2 = axis2.contourf(psi, r, U[0,:,:]/outputs.velocity[0][0],100,cmap=plt.cm.jet)#,cmap=plt.cm.jet)  
    #cbar2 = plt.colorbar(CS_2, ax=axis2)
    #cbar2.ax.set_ylabel('Total U /Vinf, m/s')
    #axis2.set_title('Total Velocity at Propeller')
    
    ## thrust distribution, blade_T_distribution_2d plot
    #T_dist_pts = outputs.blade_T_distribution_2d[0,:,:]
    #T_dist_avg = np.average(outputs.blade_T_distribution_2d[0,:,:],axis=0)
    #relative_T = T_dist_pts/T_dist_avg
    
    #plt.figure(4)
    #fig3, axis3 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_3 = axis3.contourf(psi, r, relative_T,100,cmap=plt.cm.jet)  
    #cbar3 = plt.colorbar(CS_3, ax=axis3)
    #cbar3.ax.set_ylabel('Thrust')
    #axis3.set_title('Blade Thrust Distribution, Relative to Time Averaged')
    
    ## torque distribution, blade_Q_distribution_2d plot
    #Q_dist_pts = outputs.blade_Q_distribution_2d[0,:,:]
    #Q_dist_avg = np.average(outputs.blade_Q_distribution_2d[0,:,:],axis=0)
    #relative_Q = Q_dist_pts/Q_dist_avg    
    #plt.figure(5)
    #fig4, axis4 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_4 = axis4.contourf(psi, r, relative_Q,100,cmap=plt.cm.jet) #outputs.blade_Q_distribution_2d[0,:,:],100)#,cmap=plt.cm.jet)  
    #cbar4 = plt.colorbar(CS_4, ax=axis4)
    #cbar4.ax.set_ylabel('Torque')
    #axis4.set_title('Blade Torque Distribution')
    
    #plt.figure(6)
    #fig5, axis5 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_5 = axis5.contourf(psi, r, ua_wing,100,cmap=plt.cm.jet)  
    #cbar5 = plt.colorbar(CS_5, ax=axis5)
    #cbar5.ax.set_ylabel('Ua_wing Distribution') 
    #axis5.set_title('Axial Wing Influence on Propeller')
    
    #N=20
    #psi          = np.linspace(0,2*np.pi,N)
    #psi_2d       = np.tile(np.atleast_2d(psi).T,(1,N))  
    
    #utang_wing = np.array(ut_wing)*np.cos(psi_2d)
    #ur_wing = np.array(ut_wing)*np.sin(psi_2d)
    
    #plt.figure(7)
    #fig6, axis6 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_6 = axis6.contourf(psi, r, utang_wing,100,cmap=plt.cm.jet)#(utang_wing+outputs.velocity[0][0])/outputs.velocity[0][0],100)#,cmap=plt.cm.jet)  
    #cbar6 = plt.colorbar(CS_6, ax=axis6)
    #cbar6.ax.set_ylabel('Ut_wing Distribution') 
    #axis6.set_title('Tangential Wing Influence on Propeller')
    
    #plt.figure(8)
    #fig7, axis7 = plt.subplots(subplot_kw=dict(projection='polar'))
    #CS_7 = axis7.contourf(psi, r, ur_wing,100)#,cmap=plt.cm.jet)  
    #cbar7 = plt.colorbar(CS_7, ax=axis7)
    #cbar7.ax.set_ylabel('Ut_wing Distribution') 
    #axis7.set_title('Radial Wing Influence on Propeller')    
    
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
    
    plt.show() 
    
    return



def large_prop_1(vehicle, conditions):
    # Designs the propeller to operate at specified vehicle flight conditions
    V_design = vehicle.cruise_speed
    
    # We want thrust=drag; so to specify thrust first find drag: profile drag and drag due to lift; use this for design thrust
    CL_wing = vehicle.CL_design[0][0]
    AR      = vehicle.wings.main_wing.aspect_ratio
    e       = 0.7
    CD_wing = 0.012 + CL_wing**2/(np.pi*AR*e)
    Drag    = CD_wing*0.5*conditions.freestream.density[0][0]*V_design**2*vehicle.reference_area 
    
    prop                            = SUAVE.Components.Energy.Converters.Propeller()
    prop.tag                        = 'Cessna_Prop' 
    
    prop.tip_radius                 = 2.8 * Units.feet #0.684 #
    prop.hub_radius                 = 0.6 * Units.feet
    prop.number_blades              = 2
    prop.disc_area                  = np.pi*(prop.tip_radius**2)
    prop.induced_hover_velocity     = 0 
    prop.design_freestream_velocity = V_design
    prop.angular_velocity           = 2200. * Units['rpm']
    prop.design_Cl                  = 0.7
    prop.design_altitude            = 4000 #20 * Units.feet
   
    prop.design_thrust              = Drag/vehicle.propulsors.prop_net.number_of_engines #(vehicle.mass_properties.takeoff/vehicle.net.number_of_engines)# *contingency_factor
    prop.design_power               = 0.0
    prop.thrust_angle               = 0. * Units.degrees
    prop.inputs.omega               = np.ones((1,1)) *  prop.angular_velocity
    
    N = conditions.N
    prop                            = propeller_design(prop,N)    
    return prop
def prop_1(vehicle, conditions):
    # Designs the propeller to operate at specified vehicle flight conditions
    V_design = vehicle.cruise_speed
    
    # We want thrust=drag; so to specify thrust first find drag: profile drag and drag due to lift; use this for design thrust
    CL_wing = vehicle.CL_design[0][0]
    AR      = vehicle.wings.main_wing.aspect_ratio
    e       = 0.7
    CD_wing = 0.012 + CL_wing**2/(np.pi*AR*e)
    Drag    = CD_wing*0.5*conditions.freestream.density[0][0]*V_design**2*vehicle.reference_area 
    
    prop                            = SUAVE.Components.Energy.Converters.Propeller()
    prop.tag                        = 'Cessna_Prop' 
    
    prop.tip_radius                 = 0.8*Units.feet #2.8 * Units.feet #0.684 #
    prop.hub_radius                 = 0.2*Units.feet #0.6 * Units.feet
    prop.number_blades              = 2
    prop.disc_area                  = np.pi*(prop.tip_radius**2)
    prop.induced_hover_velocity     = 0 
    prop.design_freestream_velocity = V_design
    prop.angular_velocity           = 2200. * Units['rpm']
    prop.design_Cl                  = 0.7
    prop.design_altitude            = 4000 #20 * Units.feet
   
    prop.design_thrust              = 0.3*Drag/vehicle.propulsors.prop_net.number_of_engines #(vehicle.mass_properties.takeoff/vehicle.net.number_of_engines)# *contingency_factor
    prop.design_power               = 0.0
    prop.thrust_angle               = 0. * Units.degrees
    prop.inputs.omega               = np.ones((1,1)) *  prop.angular_velocity
    
    N = conditions.N
    prop                            = propeller_design(prop,N)    
    return prop

def include_prop_config(prop,case,ua_wing,ut_wing,rotation):
    prop.prop_configs                    = Data()
    prop.prop_configs.config             = 'pusher'
    prop.prop_configs.freestream_case    = case
    prop.prop_configs.rotation           = rotation
    
    if case =='disturbed_freestream':
        # Wing influence at propeller location:
        prop.prop_configs.wing_ua        = ua_wing
        prop.prop_configs.wing_ut        = ut_wing
    else:
        prop.prop_configs.wing_ua        = np.zeros_like(ua_wing)
        prop.prop_configs.wing_ut        = np.zeros_like(ut_wing) 
        
    return prop


def wing_effect(vehicle,prop_loc):
    #-------------------------------------------------------------------------
    #          Choose wing setup:
    #-------------------------------------------------------------------------
    aoa                                            = np.array([[3 * Units.deg]])
    state                                          = cruise_conditions(vehicle.cruise_speed)    
    state.conditions.aerodynamics.angle_of_attack  = aoa
    mach                                           = state.conditions.freestream.mach_number
    N = state.conditions.N

    vortices   = 10
    n_sw       = 1#vortices **2
    n_cw       = vortices

    # --------------------------------------------------------------------------------
    #          Settings and Calling for VLM:  
    # --------------------------------------------------------------------------------
    VLM_settings        = Data()
    VLM_settings.n_sw   = n_sw
    VLM_settings.n_cw   = n_cw

    VD_geom, MCM, C_mn, gammaT, cl_y, CL, CDi   = wing_VLM(vehicle, state, VLM_settings)
    VD                                    = copy.deepcopy(VD_geom)

    #------------------------------------------------------------------------------------
    #         Plot of velocity at blade element versus blade element angle:
    #------------------------------------------------------------------------------------
    #prop_x_desired = 2.5 #2.5944615384615393
    R_tip   = vehicle.propulsors.prop_net.propeller.tip_radius
    r_hub   = vehicle.propulsors.prop_net.propeller.hub_radius
    r_R_min = r_hub/R_tip
    theta   = np.linspace(0,2*np.pi,N)
    r2d     = np.linspace(r_R_min, 0.96, N)*R_tip
    
    #r2d = np.array([0.21428571, 0.25357143, 0.29285714, 0.33214286, 0.37142857,
                    #0.41071429, 0.45      , 0.48928571, 0.52857143, 0.56785714,
         #0.60714286, 0.64642857, 0.68571429, 0.725     , 0.76428571,
         #0.80357143, 0.84285714, 0.88214286, 0.92142857, 0.96071429])*vehicle.propulsors.prop_net.propeller.tip_radius
    #Output is the u- and w- velocity components corresponding to the sweep of r2d and theta: This will become input to propeller.py
    u_pts,w_pts = Blade_Element_Rotating_Velocity_Plot(vehicle, prop_loc,theta, r2d, VD, n_sw, n_cw, aoa, mach, C_mn, MCM, gammaT)
    return u_pts, w_pts

def Blade_Element_Rotating_Velocity_Plot(vehicle, prop_loc,theta, radius, VD, n_sw, n_cw, aoa, mach, C_mn, MCM, gammaT):
    prop_x_center = np.array([vehicle.wings.main_wing.origin[0] + prop_loc[0]])
    prop_y_center = np.array([prop_loc[1]])
    prop_z_center = np.array([prop_loc[2]])

    #theta = np.linspace(0,6*np.pi,24*3)
    u_pts =  [[0 for j in range(len(radius))] for x in range(len(theta))]
    v_pts =  [[0 for j in range(len(radius))] for x in range(len(theta))]
    w_pts =  [[0 for j in range(len(radius))] for x in range(len(theta))]
    u_pts2 = [[0 for j in range(len(radius))] for x in range(len(theta))]
    v_pts2 = [[0 for j in range(len(radius))] for x in range(len(theta))]
    w_pts2 = [[0 for j in range(len(radius))] for x in range(len(theta))]

    for i in range(len(theta)):                                        
        # Outputs from VLM_velocity_sweep are ua_wing[[u(r1,phi1), ... u(rn,phi1)], ... [u(r1,phin), ... u(rn,phin)]]]
        for k in range(len(radius)):
            yloc = np.array([prop_y_center + radius[k]*np.cos(theta[i])])
            zloc = np.array([prop_z_center + radius[k]*np.sin(theta[i])])
            be_locx = [prop_x_center, yloc, zloc]
            Cmnx, uk, vk, wk, propvalk = VLM_velocity_sweep(VD,be_locx,n_sw,n_cw,aoa,mach,C_mn,MCM,gammaT)

            if wk>1 or wk<-1 or uk>1 or uk<-1:
                w_pts[i][k] = w_pts[i][k-1]
                u_pts[i][k] = u_pts[i][k-1]
                v_pts[i][k] = v_pts[i][k-1]

            else:
                u_pts[i][k] = uk[0]
                v_pts[i][k] = vk[0]
                w_pts[i][k] = wk[0]

    return u_pts,w_pts

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


def wing_VLM(vehicle,state, VLM_settings):
    # --------------------------------------------------------------------------------
    #          Get Vehicle Geometry and Unpaack Settings:  
    # --------------------------------------------------------------------------------    
    #vehicle     = vehicle_setup(wing_parameters)
    Sref         = vehicle.wings.main_wing.areas.reference
    n_sw         = VLM_settings.n_sw
    n_cw         = VLM_settings.n_cw

    aoa          = state.conditions.aerodynamics.angle_of_attack   # angle of attack  
    mach         = state.conditions.freestream.mach_number         # mach number
    
    ones         = np.atleast_2d(np.ones_like(aoa)) 
    
    # --------------------------------------------------------------------------------
    #          Generate Vortex Distribution and Build Induced Velocity Matrix:
    # --------------------------------------------------------------------------------     
    VD           = compute_vortex_distribution(vehicle,VLM_settings)
    C_mn, DW_mn  = compute_induced_velocity_matrix(VD,n_sw,n_cw,aoa,mach)
    MCM          = VD.MCM 
    
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
   
    # Build the vector
    RHS = compute_RHS_matrix(VD,n_sw,n_cw,delta,phi,state.conditions,vehicle,True)

    # Compute vortex strength  
    n_cp  = VD.n_cp  
    gamma = np.linalg.solve(A,RHS)
    GAMMA = np.repeat(np.atleast_3d(gamma), n_cp ,axis = 2 )
    u = np.sum(C_mn[:,:,:,0]*MCM[:,:,:,0]*GAMMA, axis = 2) 
    v = np.sum(C_mn[:,:,:,1]*MCM[:,:,:,1]*GAMMA, axis = 2) 
    w = np.sum(C_mn[:,:,:,2]*MCM[:,:,:,2]*GAMMA, axis = 2) 
    w_ind = -np.sum(B*MCM[:,:,:,2]*GAMMA, axis = 2) 
     
    # ---------------------------------------------------------------------------------------
    # STEP 10: Compute aerodynamic coefficients 
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
    actual_downwash      = np.arctan(w[0][0]/(state.conditions.freestream.velocity))*180/np.pi
    
    
    ## ---------------------------------------------------------------------------------------
    ##          Plot Resulting Lift Distribution: 
    ## ---------------------------------------------------------------------------------------
    #dim = len(cl_y[0])
    #half_dim = int(dim/2)
    #fig = plt.figure()
    #axes = fig.add_subplot(1,1,1)
    #x1 = np.linspace(0, (wing_parameters.span/2),half_dim)
    #x2 = np.linspace(0, -(wing_parameters.span/2),half_dim)
    #axes.plot(x1, cl_y[0][0:half_dim],'b', label="Rectangular Distribution")
    #axes.plot(x2, cl_y[0][half_dim:dim],'b')
    #axes.set_xlabel("Spanwise Location (y)")
    #axes.set_ylabel("Cl")
    #axes.set_title("Cl Distribution")
    #plt.legend()
    #plt.show()

    return VD, MCM, C_mn, GAMMA, cl_y, CL, CDi




def VLM_velocity_sweep(VD,prop_location,n_sw,n_cw,aoa,mach,C_mn,MCM,gammaT):
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