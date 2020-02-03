import SUAVE
from SUAVE.Core import Units, Data
import copy
import math
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_mass
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_max_lift_coeff  
from SUAVE.Methods.Utilities.Chebyshev  import chebyshev_data
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_induced_velocity_matrix import compute_induced_velocity_matrix
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_induced_velocity_matrix import compute_mach_cone_matrix
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
    case = 'uniform_freestream'
    #case = 'disturbed_freestream'
    
    # Output ua_wing and ut_wing vectors from Research_Tester script:
    ua_wing, ut_wing = wing_effect()
    
    
    a                 = 343  
    Mach              = 0.05  
    density           = 1.225
    dynamic_viscosity = 1.78899787e-05        
    S                 = 1.2 
    
    #------------------------------------------------------------------
    # VEHICLE
    #------------------------------------------------------------------    
    vehicle                         = SUAVE.Vehicle()
    vehicle.mass_properties.takeoff = 3000. * Units.lb 
    lb_to_kg = 0.453592 
    rho      = 1.2
    contingency_factor = 1.3
    
    net                             = Battery_Propeller()    
    net.number_of_engines           = 12     
    prop                            = SUAVE.Components.Energy.Converters.Propeller()
    prop.tag                        = 'Lift_Prop'
    prop.prop_config                = 'pusher'
    prop.freestream_case            = case
    
    if case =='disturbed_freestream':
        # Wing influence at propeller location:
        prop.wing_ua                = ua_wing
        prop.wing_ut                = ut_wing        
    else:
        prop.wing_ua                = np.zeros_like(ua_wing)
        prop.wing_ut                = np.zeros_like(ut_wing)
        
                                    
    prop.tip_radius                 = 2.8 * Units.feet
    prop.hub_radius                 = 0.6 * Units.feet      
    prop.number_blades              = 2    
    vehicle_weight                  = vehicle.mass_properties.takeoff*lb_to_kg*9.81
    
    prop.disc_area                  = np.pi*(prop.tip_radius**2)    
    prop.induced_hover_velocity     = np.sqrt(vehicle_weight/(2*rho*prop.disc_area*net.number_of_engines)) 
    prop.freestream_velocity        = prop.induced_hover_velocity 
    prop.angular_velocity           = 3100. * Units['rpm']      
    prop.design_Cl                  = 0.7
    prop.design_altitude            = 20 * Units.feet                            
    prop.design_thrust              = (vehicle.mass_properties.takeoff/net.number_of_engines)*lb_to_kg*9.81*contingency_factor
    prop.design_power               = 0.0    
    prop.thrust_angle               = 0. * Units.degrees
    prop                            = propeller_design(prop) 
    
    # add propeller to network
    net.propeller            = prop    
     
    # Prepare Inputs for Propeller Model  
    T                 = 286.16889478 
    a                 = 343  
    Mach              = 0.1 
    density           = 1.225
    dynamic_viscosity = 1.78899787e-05      
      
    ctrl_pts = 1 
    prop.inputs.omega                           = np.ones((ctrl_pts,1)) *  3100. * Units['rpm']     
    conditions                                  = Data()
    conditions.freestream                       = Data()
    conditions.propulsion                       = Data()
    conditions.frames                           = Data()  
    conditions.frames.inertial                  = Data()  
    conditions.frames.body                      = Data() 
    conditions.freestream.density               = np.ones((ctrl_pts,1)) * density
    conditions.freestream.dynamic_viscosity     = np.ones((ctrl_pts,1)) * dynamic_viscosity   
    conditions.freestream.speed_of_sound        = np.ones((ctrl_pts,1)) * a 
    conditions.freestream.temperature           = np.ones((ctrl_pts,1)) * T
    velocity_vector                             = np.array([[Mach*a, 0. ,0.]])
    conditions.frames.inertial.velocity_vector  = np.tile(velocity_vector,(ctrl_pts,1))    
    conditions.frames.body.transform_to_inertial= np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])

    # Run Propeller model 
    F, Q, P, Cp , outputs , etap = prop.spin_simple_pusher(conditions)    
        
    # ----------------------------------------------------------------------------
    # DISC PLOTS   
    # ----------------------------------------------------------------------------
    theta = outputs.psi_2d[0,:,:]
    r     = outputs.r_2d[0,:,:] 
    
    # perpendicular velocity, up Plot 
    plt.figure(0)
    fig1, axis1 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_1 = axis1.contourf(theta, r, outputs.va_2d[0,:,:],cmap=plt.cm.jet)    
    cbar1 = plt.colorbar(CS_1, ax=axis1)
    cbar1.ax.set_ylabel('Ua , m/s')  
    
    # tangentIal velocity, ut Plot    
    plt.figure(1)
    fig2, axis2 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_2 = axis2.contourf(theta, r, outputs.vt_2d[0,:,:],cmap=plt.cm.jet)    
    cbar2 = plt.colorbar(CS_2, ax=axis2)
    cbar2.ax.set_ylabel('Ut , m/s')     
    
    # total velocity, U plot
    plt.figure(2)
    U = np.sqrt(outputs.va_2d**2 + outputs.vt_2d**2)
    fig3, axis3 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_3 = axis3.contourf(theta, r, U[0,:,:],cmap=plt.cm.jet)  
    cbar3 = plt.colorbar(CS_3, ax=axis3)
    cbar3.ax.set_ylabel('Total U , m/s') 
    
    # thrust distribution, blade_T_distribution_2d plot
    plt.figure(3)
    fig4, axis4 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_4 = axis4.contourf(theta, r, outputs.blade_T_distribution_2d[0,:,:],cmap=plt.cm.jet)  
    cbar4 = plt.colorbar(CS_4, ax=axis4)
    cbar4.ax.set_ylabel('Thrust Distribution')
    
    # torque distribution, blade_Q_distribution_2d plot
    plt.figure(4)
    fig4, axis4 = plt.subplots(subplot_kw=dict(projection='polar'))
    CS_4 = axis4.contourf(theta, r, outputs.blade_Q_distribution_2d[0,:,:],cmap=plt.cm.jet)  
    cbar4 = plt.colorbar(CS_4, ax=axis4)
    cbar4.ax.set_ylabel('Torque Distribution')     
    
    plt.show() 
    
    return 
    
def wing_effect():
    #-------------------------------------------------------------------------
    #          Choose wing setup:
    #-------------------------------------------------------------------------
    vehicle, xwing, ywing = flat_plate_wing() #simple_wing_textbook() #

    aoa     = np.array([[-2 * Units.deg]])
    state   = cruise_conditions(aoa)    
    mach    = state.conditions.freestream.mach_number

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
    prop_x_desired = 2.5944615384615393
    theta = np.linspace(0,2*np.pi,20)
    r2d = np.array([0.21428571, 0.25357143, 0.29285714, 0.33214286, 0.37142857,
                    0.41071429, 0.45      , 0.48928571, 0.52857143, 0.56785714,
         0.60714286, 0.64642857, 0.68571429, 0.725     , 0.76428571,
         0.80357143, 0.84285714, 0.88214286, 0.92142857, 0.96071429])
    #Output is the u- and w- velocity components corresponding to the sweep of r2d and theta: This will become input to propeller.py
    u_pts,w_pts = Blade_Element_Rotating_Velocity_Plot(vehicle, prop_x_desired,theta, r2d, VD, n_sw, n_cw, aoa, mach, C_mn, MCM, gammaT)
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


def cruise_conditions(aoa):
    # --------------------------------------------------------------------------------    
    #          Cruise conditions  
    # --------------------------------------------------------------------------------
    rho                  = np.array([[0.365184]])
    mu                   = np.array([[0.0000143326]])
    T                    = np.array([[258]])
    P                    = 57200.0
    a                    = 322.2
    velocity_freestream  = np.array([[16.11]]) #m/s
    mach                 = velocity_freestream/a #np.array([[0.8]]) #
    pressure             = np.array([57200.0])
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
    state.conditions.aerodynamics.angle_of_attack        = aoa
    state.conditions.frames.body.transform_to_inertial   = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])   
    return state


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


def Blade_Element_Rotating_Velocity_Plot(vehicle, prop_x_desired,theta, radius, VD, n_sw, n_cw, aoa, mach, C_mn, MCM, gammaT):
    prop_x_center = np.array([vehicle.wings.main_wing.origin[0] + prop_x_desired])
    prop_y_center = np.array([3.0])
    prop_z_center = np.array([0.0])

    #theta = np.linspace(0,6*np.pi,24*3)
    u_pts =  [[0 for j in range(len(theta))] for k in range(len(radius))]
    v_pts =  [[0 for j in range(len(theta))] for k in range(len(radius))]
    w_pts =  [[0 for j in range(len(theta))] for k in range(len(radius))]
    u_pts2 = [[0 for j in range(len(theta))] for k in range(len(radius))]
    v_pts2 = [[0 for j in range(len(theta))] for k in range(len(radius))]
    w_pts2 = [[0 for j in range(len(theta))] for k in range(len(radius))]


    for i in range(len(theta)):                                        
        #theta = 2*np.pi/discretizations
        #z_loc1 = np.array([prop_z_center + radius*np.sin(theta[i])])
        #y_loc1 = np.array([prop_y_center + radius*np.cos(theta[i])])
        #be_loc1 = [prop_x_center, y_loc1, z_loc1]

        #z_loc2 = np.array([prop_z_center + 0.5*radius*np.sin(theta[i])])
        #y_loc2 = np.array([prop_y_center + 0.5*radius*np.cos(theta[i])])
        #be_loc2 = [prop_x_center, y_loc2, z_loc2]

        #I need outputs from VLM_velocity_sweep to be ua_wing[[u(r1,phi1), ... u(rn,phi1)], ... [u(r1,phin), ... u(rn,phin)]]]
        for k in range(len(radius)):
            yloc = np.array([prop_y_center + radius[k]*np.cos(theta[i])])
            zloc = np.array([prop_z_center + radius[k]*np.sin(theta[i])])
            be_locx = [prop_x_center, yloc, zloc]
            Cmnx, uk, vk, wk, propvalk = VLM_velocity_sweep(VD,be_locx,n_sw,n_cw,aoa,mach,C_mn,MCM,gammaT)

            if wk>10 or wk<-10:
                w_pts[i][k] = w_pts[i-1]
                u_pts[i][k] = u_pts[i-1]
                v_pts[i][k] = v_pts[i-1]

            else:
                u_pts[i][k] = uk[0]
                v_pts[i][k] = vk[0]
                w_pts[i][k] = wk[0]


    ##Accessing the elements corresponding to the max radius in r2d:
    #rad_col_num = 19
    #u_pts = np.array(u_pts)
    #w_pts = np.array(w_pts)
    #u_single_radius = u_pts[ range(np.shape(u_pts)[0]), rad_col_num]
    #w_single_radius = w_pts[ range(np.shape(w_pts)[0]), rad_col_num]

    #fig_4 = plt.figure()
    #axes_4 = fig_4.add_subplot(1,1,1)
    #p4 = axes_4.plot(np.rad2deg(theta),u_single_radius, label='U for blade element at R')
    #p4b = axes_4.plot(np.rad2deg(theta),w_single_radius, label='W for blade element at R')
    ##p3 = axes_4.plot(np.rad2deg(theta),u_pts2, label='U for blade element at 0.5R')
    ##p3b = axes_4.plot(np.rad2deg(theta),w_pts2, label='W for blade element at 0.5R')    
    #axes_4.set_xlabel("Angle of Blade (deg)")
    #axes_4.set_ylabel("Spanwise Location(y)")   
    #axes_4.set_title("Velocity Sweep")    
    #plt.legend()
    #plt.show()
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

    return C_mn, u, v, w, prop_val #C_mn, u[0,0], v[0,0], w[0,0]

def compute_MCA(lamda,zeta,chi,V,R,Cl,B,T,x,a,nu):
    '''taken from propeller design model'''
    tanphit = lamda*(1.+zeta/2.)   # Tangent of the flow angle at the tip
    phit    = np.arctan(tanphit)   # Flow angle at the tip
    tanphi  = tanphit/chi          # Flow angle at every station
    f       = (B/2.)*(1.-chi)/np.sin(phit) 
    F       = (2./np.pi)*np.arccos(np.exp(-f)) #Prandtl momentum loss factor
    phi     = np.arctan(tanphi)  #Flow angle at every station
    
    #Step 3, determine the product Wc, and RE
    G       = F*x*np.cos(phi)*np.sin(phi) #Circulation function
    Wc      = 4.*np.pi*lamda*G*V*R*zeta/(Cl*B)
    Ma      = Wc/a
    RE      = Wc/nu

    #Step 4, determine epsilon and alpha from airfoil data
    
    #This is an atrocious fit of DAE51 data at RE=50k for Cd
    #There is also RE scaling
    Cdval   = (0.108*(Cl**4)-0.2612*(Cl**3)+0.181*(Cl**2)-0.0139*Cl+0.0278)*((50000./RE)**0.2)

    #More Cd scaling from Mach from AA241ab notes for turbulent skin friction
    Tw_Tinf = 1. + 1.78*(Ma**2)
    Tp_Tinf = 1. + 0.035*(Ma**2) + 0.45*(Tw_Tinf-1.)
    Tp      = Tp_Tinf*T
    Rp_Rinf = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4)
    
    Cd      = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval
    
    alpha   = Cl/(2.*np.pi)
    epsilon = Cd/Cl
    
    #Step 5, change Cl and repeat steps 3 and 4 until epsilon is minimized
    
    #Step 6, determine a and a', and W
    
    a       = (zeta/2.)*(np.cos(phi)**2.)*(1.-epsilon*np.tan(phi))
    aprime  = (zeta/(2.*x))*np.cos(phi)*np.sin(phi)*(1.+epsilon/np.tan(phi))
    W       = V*(1.+a)/np.sin(phi)
    
    #Step 7, compute the chord length and blade twist angle 
    c       = Wc/W
    MCA = c/4. - c[0]/4.   
    
    return MCA

if __name__ == '__main__': 
    main()    
    plt.show()   