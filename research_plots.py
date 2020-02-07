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
    # Specify the propeller attributes:
    #-----------------------------------------------------------------
    case        = 'disturbed_freestream' #'uniform_freestream'
    rotation    = 'ccw'
    state       = cruise_conditions() 

    #-----------------------------------------------------------------    
    # Design the propeller once for the given cruise condition:
    #-----------------------------------------------------------------
    prop        = prop_1(state.conditions.freestream.velocity[0][0])
    prop_radius = prop.tip_radius

    #-----------------------------------------------------------------
    # Observe/include wing influence at prop location in prop definition:
    #-----------------------------------------------------------------
    prop_loc         = [2.5,3.0,0.0]
    ua_wing, ut_wing = wing_effect(prop_loc,prop_radius)    
    prop             = include_prop_config(prop,case,ua_wing,ut_wing,rotation)   
    
    #-------------------------------------------------------------------
    # Generate 2D plots of thrust, torque, power versus propeller y-location:
    #-------------------------------------------------------------------
    plots_v_prop_loc(prop,state.conditions)

    ##-------------------------------------------------------------------    
    ## Generate plots of thrust, torque, etc. on propeller disk:
    ##-------------------------------------------------------------------           
    #run_plots_prop_disk(case, rotation, state.conditions,prop_radius)

    return

def cruise_conditions():
    # --------------------------------------------------------------------------------    
    #          Cruise conditions  
    # --------------------------------------------------------------------------------
    # Calculated for altitude of 4000 m (13000ft), typical altitude of a Cessna (max is 14000ft)
    rho                  = np.array([[0.8194]])
    mu                   = np.array([[0.00001661]])
    T                    = np.array([[262.17]])
    a                    = np.array([[np.sqrt(1.4*287.058*T[0][0])]])
    velocity_freestream  = np.array([[20]]) #m/s
    mach                 = velocity_freestream/a #np.array([[0.8]]) #
    pressure             = np.array([[57200.0]])
    re                   = rho*a*mach/mu
    
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
    
    velocity_vector                             = np.array([[mach[0][0]*a[0][0], 0. ,0.]])
    state.conditions.frames.inertial.velocity_vector  = np.tile(velocity_vector,(1,1))  
    state.conditions.frames.body.transform_to_inertial   = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])   
    return state



def baseline_propeller(prop, conditions):
    case  = 'uniform_freestream'
    prop  = include_prop_config(prop,case,prop.wing_ua,prop.wing_ut,prop.rotation) 
    
    F, Q, P, Cp, outputs, etap = prop.spin_simple_pusher(conditions)
        
    return F, Q, P, Cp, outputs, etap

def plots_v_prop_loc(prop, conditions):
    # Determine wing influence on propeller for each point along the wing:
    prop_y_locs = np.linspace(0.25,4,30)
    
    net                             = Battery_Propeller()    
    net.number_of_engines           = 2

    thrust_vals = np.zeros_like(prop_y_locs)
    torque_vals = np.zeros_like(prop_y_locs)
    power_vals  = np.zeros_like(prop_y_locs)
    Cp_vals     = np.zeros_like(prop_y_locs)   
    thrust_vals_2 = np.zeros_like(prop_y_locs)
    torque_vals_2 = np.zeros_like(prop_y_locs)
    power_vals_2 = np.zeros_like(prop_y_locs)
    Cp_vals_2 = np.zeros_like(prop_y_locs)  
    thrust_vals_3 = np.zeros_like(prop_y_locs)
    torque_vals_3 = np.zeros_like(prop_y_locs)
    power_vals_3 = np.zeros_like(prop_y_locs)
    Cp_vals_3 = np.zeros_like(prop_y_locs)      
    
    V1 = 15
    V2 = 20
    V3 = 22
    
    # Adjust wing influence on propeller for each possible placement on the wing. Generate results:
    for i in range(len(prop_y_locs)):
        prop_loc         = [2.5,prop_y_locs[i],0]
        ua_wing, ut_wing = wing_effect(prop_loc,prop.tip_radius)
        prop             = include_prop_config(prop,prop.freestream_case,ua_wing,ut_wing,prop.rotation)   
        net.propeller    = prop  
        
        # Run Propeller model for Disturbed Freestream:
        conditions.frames.inertial.velocity_vector[0]= V1
        F, Q, P, Cp , outputs , etap = prop.spin_simple_pusher(conditions) #prop.spin(conditions) # 
        conditions.frames.inertial.velocity_vector[0] = V2
        F_2, Q_2, P_2, Cp_2 , outputs , etap = prop.spin_simple_pusher(conditions) #prop.spin(conditions) #        
        conditions.frames.inertial.velocity_vector[0] = V3
        F_3, Q_3, P_3, Cp_3 , outputs , etap = prop.spin_simple_pusher(conditions) #prop.spin(conditions) #
        
        thrust_vals[i]=F
        torque_vals[i]=Q
        power_vals[i] =P
        Cp_vals[i]    =Cp
        thrust_vals_2[i]=F_2
        torque_vals_2[i]=Q_2
        power_vals_2[i] =P_2
        Cp_vals_2[i]    =Cp_2
        thrust_vals_3[i]=F_3
        torque_vals_3[i]=Q_3
        power_vals_3[i] =P_3
        Cp_vals_3[i]    =Cp_3       
        
    # Results from case with Uniform Flow are used as Baseline:
    conditions.frames.inertial.velocity_vector[0] = V1
    T_base, Q_base, P_base, Cp_base, outputs, etap = baseline_propeller(prop, conditions)    
    T_baseline = np.ones_like(thrust_vals)*T_base[0][0] #2591.77561215
    Q_baseline = np.ones_like(thrust_vals)*Q_base[0][0] #287.4145557
    P_baseline = np.ones_like(thrust_vals)*P_base[0][0] #93303.74385
    Cp_baseline = np.ones_like(thrust_vals)*Cp_base[0][0] #0.03811682
    
    conditions.frames.inertial.velocity_vector[0] = V2
    T_base2, Q_base2, P_base2, Cp_base2, outputs, etap = baseline_propeller(prop, conditions)    
    T_baseline_2 = np.ones_like(thrust_vals_2)*T_base2[0][0] #2591.77561215
    Q_baseline_2 = np.ones_like(thrust_vals_2)*Q_base2[0][0] #287.4145557
    P_baseline_2 = np.ones_like(thrust_vals_2)*P_base2[0][0] #93303.74385
    Cp_baseline_2 = np.ones_like(thrust_vals_2)*Cp_base2[0][0] #0.03811682    
    
    conditions.frames.inertial.velocity_vector[0] = V3
    T_base3, Q_base3, P_base3, Cp_base3, outputs, etap = baseline_propeller(prop, conditions)    
    T_baseline_3 = np.ones_like(thrust_vals_3)*T_base3[0][0] #2591.77561215
    Q_baseline_3 = np.ones_like(thrust_vals_3)*Q_base3[0][0] #287.4145557
    P_baseline_3 = np.ones_like(thrust_vals_3)*P_base3[0][0] #93303.74385
    Cp_baseline_3 = np.ones_like(thrust_vals_3)*Cp_base3[0][0] #0.03811682       
    
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
    eta_1 = thrust_vals*V1/power_vals
    eta_2 = thrust_vals_2*V2/power_vals_2
    eta_3 = thrust_vals_3*V3/power_vals_3

    eta_baseline_1 = T_baseline*V1/P_baseline
    eta_baseline_2 = T_baseline_2*V2/P_baseline_2
    eta_baseline_3 = T_baseline_3*V3/P_baseline_3    
    
    axisC = figC.add_subplot(1,1,1)
    axisC.plot(prop_y_locs/half_span,eta_1, label='V1_disturbed')#/(rho*n**2*D**5))
    axisC.plot(prop_y_locs/half_span,eta_baseline_1, label='V1_baseline')#/(rho*n**2*D**5))
    axisC.plot(prop_y_locs/half_span,eta_2, label='V2_disturbed')#/(rho*n**2*D**5))
    axisC.plot(prop_y_locs/half_span,eta_baseline_2, label='V2_baseline')#/(rho*n**2*D**5))
    axisC.plot(prop_y_locs/half_span,eta_3, label='V3_disturbed')#/(rho*n**2*D**5))
    axisC.plot(prop_y_locs/half_span,eta_baseline_3, label='V3_baseline')#/(rho*n**2*D**5))    
    axisC.set_ylabel('Propeller Efficiency')
    axisC.set_xlabel('Spanwise Center Propeller Location (y/0.5b)')
    axisC.set_title('Propeller Placement Effect on eta')  
    
    
    plt.legend()
    plt.show()
    
    return

def run_plots_prop_disk(case, rotation, conditions,R):
    #------------------------------------------------------------------
    # VEHICLE
    #------------------------------------------------------------------    
    vehicle                         = SUAVE.Vehicle()
    vehicle.mass_properties.takeoff = 1669. * Units.lb 
    rho                             = conditions.freestream.density
    
    net                    = Battery_Propeller()    
    net.number_of_engines  = 2
    prop_loc               = [2.5,3,0]
    ua_wing, ut_wing       = wing_effect(prop_loc,R)
    
    prop             = prop_1(conditions.freestream.velocity[0][0])
    prop             = include_prop_config(prop,case,ua_wing,ut_wing,rotation)  
    net.propeller    = prop  
     
    # Run Propeller model 
    F, Q, P, Cp , outputs , etap = prop.spin_simple_pusher(conditions) #prop.spin(conditions) #   
    
  
    # ----------------------------------------------------------------------------
    # DISC PLOTS   
    # ----------------------------------------------------------------------------
    theta = outputs.psi_2d[0,:,:]
    r     = outputs.r_2d[0,:,:] 
    
    # perpendicular velocity, up Plot 
    figT=plt.figure(0)
    axisT = figT.add_subplot(1,1,1)
    axisT.plot(r[0], outputs.blade_T_distribution[0])#,cmap=plt.cm.jet)  
    axisT.set_ylabel('Thrust Distribution')  
    
    plt.figure(1)
    fig0, axis0 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_0 = axis0.contourf(theta, r, (outputs.va_2d[0,:,:]-outputs.velocity[0][0])/outputs.velocity[0][0])#,cmap=plt.cm.jet)    
    cbar0 = plt.colorbar(CS_0, ax=axis0)
    cbar0.ax.set_ylabel('(Ua-Vinf)/Vinf , m/s')  
    
    # tangentIal velocity, ut Plot    
    plt.figure(2)
    fig1, axis1 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_1 = axis1.contourf(theta, r, outputs.vt_2d[0,:,:]/outputs.velocity[0][0])#,cmap=plt.cm.jet)    
    cbar1 = plt.colorbar(CS_1, ax=axis1)
    cbar1.ax.set_ylabel('Ut/Vinf , m/s')     
    
    # total velocity, U plot
    plt.figure(3)
    U = np.sqrt(outputs.va_2d**2 + outputs.vt_2d**2)
    fig2, axis2 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_2 = axis2.contourf(theta, r, U[0,:,:]/outputs.velocity[0][0])#,cmap=plt.cm.jet)  
    cbar2 = plt.colorbar(CS_2, ax=axis2)
    cbar2.ax.set_ylabel('Total U /Vinf, m/s') 
    
    # thrust distribution, blade_T_distribution_2d plot
    plt.figure(4)
    fig3, axis3 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_3 = axis3.contourf(theta, r, outputs.blade_T_distribution_2d[0,:,:])#,cmap=plt.cm.jet)  
    cbar3 = plt.colorbar(CS_3, ax=axis3)
    cbar3.ax.set_ylabel('Thrust Distribution')
    
    # torque distribution, blade_Q_distribution_2d plot
    plt.figure(5)
    fig4, axis4 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_4 = axis4.contourf(theta, r, outputs.blade_Q_distribution_2d[0,:,:])#,cmap=plt.cm.jet)  
    cbar4 = plt.colorbar(CS_4, ax=axis4)
    cbar4.ax.set_ylabel('Torque Distribution')   
    
    plt.figure(6)
    fig5, axis5 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_5 = axis5.contourf(theta, r, ua_wing)#,cmap=plt.cm.jet)  
    cbar5 = plt.colorbar(CS_5, ax=axis5)
    cbar5.ax.set_ylabel('Ua_wing Distribution') 
    
    plt.figure(7)
    fig6, axis6 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_6 = axis6.contourf(theta, r, ut_wing)#,cmap=plt.cm.jet)  
    cbar6 = plt.colorbar(CS_6, ax=axis6)
    cbar6.ax.set_ylabel('Ut_wing Distribution') 
    
    plt.show() 
    
    return



def prop_1(V_design):
    prop                            = SUAVE.Components.Energy.Converters.Propeller()
    prop.tag                        = 'Lift_Prop'    
  
    prop.tip_radius                 = 2.8 * Units.feet
    prop.hub_radius                 = 0.6 * Units.feet
    prop.number_blades              = 2
    prop.disc_area                  = np.pi*(prop.tip_radius**2)
    prop.induced_hover_velocity     = 0 
    prop.design_freestream_velocity = V_design
    prop.angular_velocity           = 3100. * Units['rpm']
    prop.design_Cl                  = 0.7
    prop.design_altitude            = 20 * Units.feet
    # Cessna 172 has a conservative L/D of 5:1, therefore, the design thrust is based on vehicle weight (7424.082 N)
    prop.design_thrust              = 800#1490 #2189.627212572062 #(vehicle.mass_properties.takeoff/net.number_of_engines)*Units.lb*9.81*contingency_factor
    prop.design_power               = 0.0
    prop.thrust_angle               = 0. * Units.degrees
    prop.inputs.omega               = np.ones((1,1)) *  3100. * Units['rpm']
     
    prop                            = propeller_design(prop)    
    return prop

def include_prop_config(prop,case,ua_wing,ut_wing,rotation):
    prop.prop_config                = 'pusher'
    prop.freestream_case            = case
    
    if case =='disturbed_freestream':
        # Wing influence at propeller location:
        prop.wing_ua                = ua_wing
        prop.wing_ut                = ut_wing
    else:
        prop.wing_ua                = np.zeros_like(ua_wing)
        prop.wing_ut                = np.zeros_like(ut_wing) 
        prop.rotation               = rotation
        
    return prop


def wing_effect(prop_loc,R):
    #-------------------------------------------------------------------------
    #          Choose wing setup:
    #-------------------------------------------------------------------------
    vehicle, xwing, ywing = flat_plate_wing() #simple_wing_textbook() #

    aoa                                            = np.array([[2 * Units.deg]])
    state                                          = cruise_conditions()    
    state.conditions.aerodynamics.angle_of_attack  = aoa
    mach                                           = state.conditions.freestream.mach_number

    radius     = 30.0 * Units.inches
    vortices   = 4
    n_sw       = 1#vortices **2
    n_cw       = vortices    

    # --------------------------------------------------------------------------------
    #          Settings and Calling for VLM:  
    # --------------------------------------------------------------------------------
    VLM_settings                          = Data()
    VLM_settings.number_panels_spanwise   = n_sw
    VLM_settings.number_panels_chordwise  = n_cw

    VD_geom, MCM, C_mn, gammaT, CL, CDi   = wing_VLM(vehicle, state, VLM_settings)
    VD                                    = copy.deepcopy(VD_geom)

    #------------------------------------------------------------------------------------
    #         Plot of velocity at blade element versus blade element angle:
    #------------------------------------------------------------------------------------
    #prop_x_desired = 2.5 #2.5944615384615393
    theta = np.linspace(0,2*np.pi,20)
    r2d = np.array([0.21428571, 0.25357143, 0.29285714, 0.33214286, 0.37142857,
                    0.41071429, 0.45      , 0.48928571, 0.52857143, 0.56785714,
         0.60714286, 0.64642857, 0.68571429, 0.725     , 0.76428571,
         0.80357143, 0.84285714, 0.88214286, 0.92142857, 0.96071429])*R
    #Output is the u- and w- velocity components corresponding to the sweep of r2d and theta: This will become input to propeller.py
    u_pts,w_pts = Blade_Element_Rotating_Velocity_Plot(vehicle, prop_loc,theta, r2d, VD, n_sw, n_cw, aoa, mach, C_mn, MCM, gammaT)
    return u_pts, w_pts

def flat_plate_wing():
    #-------------------------------------------------------------------------
    #          Variables:
    #-------------------------------------------------------------------------    
    span         = 8
    croot        = 2
    ctip         = 2
    sweep_le     = 0.0
    sref         = span*(croot+ctip)/2 #174.0*Units.feet **2      
    twist_root   = 0.0 * Units.degrees 
    twist_tip    = 0.0 * Units.degrees   
    dihedral     = 0.0 * Units.degrees
    AR           = span**2/sref
    # ------------------------------------------------------------------
    # Initialize the Vehicle and Main Wing
    # ------------------------------------------------------------------    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'simple_wing'           
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'        
    
    wing.aspect_ratio            = span/((croot+ctip)/2)
    wing.spans.projected         = span
    wing.chords.root             = croot
    wing.chords.tip              = ctip
    wing.areas.reference         = sref
    vehicle.reference_area       = wing.areas.reference 
    wing.twists.root             = twist_root
    wing.twists.tip              = twist_tip
    wing.sweeps.leading_edge     = sweep_le #45. * Units.degrees
    wing.dihedral                = dihedral #0. * Units.degrees
    
    wing.span_efficiency         = 0.98 
    wing.origin                  = [0.,0.,0.]
    wing.vertical                = False 
    wing.symmetric               = True
    
    a0 = 2*np.pi
    if (AR>=4 and sweep_le ==0):
        a_finite_wing = a0*(AR/(2+AR))
    elif (AR<4 and sweep_le ==0):
        a_finite_wing = a0/(np.sqrt(1+((a0/(np.pi*AR))**2)) + (a0/(np.pi*AR)))
    elif (AR<4 and sweep_le>0):
        a_finite_wing = a0*np.cos(sweep_le)/(np.sqrt(1+((a0*np.cos(sweep_le))/(np.pi*AR))**2) + (a0*np.cos(sweep_le))/(np.pi*AR))
    
    wing.a_finite_wing = a_finite_wing
    
    vehicle.append_component(wing) 
    
    xwing = [0, vehicle.wings.main_wing.chords.tip, vehicle.wings.main_wing.chords.root, vehicle.wings.main_wing.chords.tip,0,0]
    ywing = [-vehicle.wings.main_wing.spans.projected/2, -vehicle.wings.main_wing.spans.projected/2, 0, vehicle.wings.main_wing.spans.projected/2, vehicle.wings.main_wing.spans.projected/2, -vehicle.wings.main_wing.spans.projected/2]
    
    return vehicle, xwing, ywing



def wing_VLM(geometry,state, settings):
    # --------------------------------------------------------------------------------
    #          Get Vehicle Geometry and Unpaack Settings:  
    # --------------------------------------------------------------------------------    
    #geometry     = vehicle_setup(wing_parameters)
    Sref         = geometry.wings.main_wing.areas.reference
    n_sw         = settings.number_panels_spanwise
    n_cw         = settings.number_panels_chordwise
    aoa          = state.conditions.aerodynamics.angle_of_attack   # angle of attack  
    mach         = state.conditions.freestream.mach_number         # mach number
    
    ones         = np.atleast_2d(np.ones_like(aoa)) 
    
    # --------------------------------------------------------------------------------
    #          Generate Vortex Distribution and Build Induced Velocity Matrix:
    # --------------------------------------------------------------------------------     
    VD           = compute_vortex_distribution(geometry,settings)
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
    sur_flag = True # ignore propeller in front of wing
    RHS = compute_RHS_matrix(VD,n_sw,n_cw,delta,phi,state.conditions,geometry,sur_flag)

    # --------------------------------------------------------------------------------
    #          Compute Vortex Strength:
    # --------------------------------------------------------------------------------  
    n_cp                 = VD.n_cp  
    gamma                = np.linalg.solve(A,RHS)
    GAMMA                = np.repeat(np.atleast_3d(gamma), n_cp ,axis = 2 )
    gammaT               = np.transpose(gamma) 
    
    u           = (C_mn[:,:,:,0]*MCM[:,:,:,0]@gammaT)[:,:,0]
    v           = (C_mn[:,:,:,1]*MCM[:,:,:,1]@gammaT)[:,:,0]
    w           = (C_mn[:,:,:,2]*MCM[:,:,:,2]@gammaT)[:,:,0]  
    w_ind       = -np.sum(B*MCM[:,:,:,2]*GAMMA, axis = 2) 
     
    # ---------------------------------------------------------------------------------------
    #          Compute Aerodynamic Coefficients: 
    # ---------------------------------------------------------------------------------------  
    n_cppw               = n_sw*n_cw
    n_w                  = VD.n_w
    CS                   = VD.CS*ones
    wing_areas           = VD.wing_areas
    CL_wing              = np.zeros(n_w)
    CDi_wing             = np.zeros(n_w)
    Del_Y                = np.abs(VD.YB1 - VD.YA1)*ones
    
    # Linspace out where breaks are
    wing_space           = np.linspace(0,n_cppw*n_w,n_w+1)
    
    # Use split to divide u, w, gamma, and Del_y into more arrays
    u_n_w                = np.array(np.array_split(u,n_w,axis=1))
    u_n_w_sw             = np.array(np.array_split(u,n_w*n_sw,axis=1))
    
    w_ind_n_w            = np.array(np.array_split(w_ind,n_w,axis=1))
    w_ind_n_w_sw         = np.array(np.array_split(w_ind,n_w*n_sw,axis=1))    
    gamma_n_w            = np.array(np.array_split(gamma,n_w,axis=1))
    gamma_n_w_sw         = np.array(np.array_split(gamma,n_w*n_sw,axis=1))
    Del_Y_n_w            = np.array(np.array_split(Del_Y,n_w,axis=1))
    Del_Y_n_w_sw         = np.array(np.array_split(Del_Y,n_w*n_sw,axis=1))
    
    # Calculate the Coefficients on each wing individually
    L_wing               = np.sum(np.multiply(u_n_w+1,(gamma_n_w*Del_Y_n_w)),axis=2).T
    CL_wing              = L_wing/wing_areas
    Di_wing              = np.sum(np.multiply(-w_ind_n_w,(gamma_n_w*Del_Y_n_w)),axis=2).T
    CDi_wing             = Di_wing/wing_areas

    # total lift and lift coefficient
    L                    = np.atleast_2d(np.sum(np.multiply((1+u),gamma*Del_Y),axis=1)).T 
    CL                   = L/(0.5*Sref)   # validated form page 402-404, aerodynamics for engineers 
    
    CDi                  = np.sum(CDi_wing[0])
    # ---------------------------------------------------------------------------------------
    #          Comparisons to Expectations:
    # ---------------------------------------------------------------------------------------
    AR                   = geometry.wings.main_wing.spans.projected/geometry.wings.main_wing.chords.root
    CL_flat_plate        = 2*np.pi*aoa*(AR/(2+AR))
    
    elliptical_downwash  = CL/(np.pi*AR)
    actual_downwash      = np.arctan(w[0][0]/(state.conditions.freestream.velocity))*180/np.pi
    
    #y = np.linspace(-wing_parameters.span/2,wing_parameters.span/2, len(cl_y[0]))
    #l_elliptical         = (4*L/(np.pi*wing_parameters.span))*np.sqrt(1-(y/(wing_parameters.span/2))**2)
    #cl_elliptical        = l_elliptical/(0.5*Sref)
    
    ## ---------------------------------------------------------------------------------------
    ##          2D Results: Plot Resulting Lift Distribution: 
    ## ---------------------------------------------------------------------------------------
    ## Calculate each spanwise set of Cls and Cds, then split Cls and Cds for each wing
    #cl_y                 = np.sum(np.multiply(u_n_w_sw +1,(gamma_n_w_sw*Del_Y_n_w_sw)),axis=2).T/CS
    #cdi_y                = np.sum(np.multiply(-w_ind_n_w_sw,(gamma_n_w_sw*Del_Y_n_w_sw)),axis=2).T/CS
    #Cl_wings             = np.array(np.split(cl_y,n_w,axis=1))
    #Cd_wings             = np.array(np.split(cdi_y,n_w,axis=1))
    
    #dim = len(cl_y[0])
    #half_dim = int(dim/2)
    #fig = plt.figure()
    #axes = fig.add_subplot(1,1,1)
    #x1 = np.linspace(0, (wing_parameters.span/2),half_dim)
    #x2 = np.linspace(0, -(wing_parameters.span/2),half_dim)
    #axes.plot(x1, cl_y[0][0:half_dim],'b', label="Rectangular Distribution")
    #axes.plot(x2, cl_y[0][half_dim:dim],'b')
    #axes.plot(y, cl_elliptical[0], 'r',label="Elliptical Distribution")
    #axes.set_xlabel("Spanwise Location (y)")
    #axes.set_ylabel("Cl")
    #axes.set_title("Cl Distribution")
    #plt.legend()
    #plt.show()

    return VD, MCM, C_mn, gammaT, CL, CDi


def Blade_Element_Rotating_Velocity_Plot(vehicle, prop_loc,theta, radius, VD, n_sw, n_cw, aoa, mach, C_mn, MCM, gammaT):
    prop_x_center = np.array([vehicle.wings.main_wing.origin[0] + prop_loc[0]])
    prop_y_center = np.array([prop_loc[1]])
    prop_z_center = np.array([prop_loc[2]])

    #theta = np.linspace(0,6*np.pi,24*3)
    u_pts =  [[0 for j in range(len(theta))] for k in range(len(radius))]
    v_pts =  [[0 for j in range(len(theta))] for k in range(len(radius))]
    w_pts =  [[0 for j in range(len(theta))] for k in range(len(radius))]
    u_pts2 = [[0 for j in range(len(theta))] for k in range(len(radius))]
    v_pts2 = [[0 for j in range(len(theta))] for k in range(len(radius))]
    w_pts2 = [[0 for j in range(len(theta))] for k in range(len(radius))]

    for i in range(len(theta)):                                        
        #I need outputs from VLM_velocity_sweep to be ua_wing[[u(r1,phi1), ... u(rn,phi1)], ... [u(r1,phin), ... u(rn,phin)]]]
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
                cp_x[count]     = prop_location[0][x]
                cp_y[count]     = prop_location[1][y]
                cp_z[count]     = prop_location[2][z]
                prop_val[count] = [cp_x[count],cp_y[count],cp_z[count]]
                count           = count + 1

    max_val_per_loop    = len(C_mn[0])
    num_pts_of_interest = len(cp_x)

    u  = np.zeros(num_pts_of_interest)
    v  = np.zeros(num_pts_of_interest)
    w  = np.zeros(num_pts_of_interest)

    count  = 0
    num_loops_required = math.ceil(num_pts_of_interest/max_val_per_loop)
    for i in range(num_loops_required): # Loop through the number of the points and keep track of the count
        # Updating Vortex Distribution Matrix:
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

    # Check for abnormalaties in induced velocities
    check = (np.linspace(1,len(u)-1,len(u)-1))
    for val in check:
        j = int(val)
        if u[j] >= 4 or u[j] <= -4:
            u[j] = u[j-1]
        if v[j] >= 4 or v[j] <= -4:
            v[j] = v[j-1]       
        if w[j] >= 4 or w[j] <= -4:
            w[j] = w[j-1]        

    return C_mn, u, v, w, prop_val


if __name__ == '__main__': 
    main()    
    plt.show()   