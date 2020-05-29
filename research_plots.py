# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import scipy.optimize
import SUAVE
from SUAVE.Core import Units
import numpy as np
import pylab as plt
import math
import copy
from SUAVE.Core import Data
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_induced_velocity_matrix import compute_induced_velocity_matrix
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_induced_velocity_matrix import compute_mach_cone_matrix
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_vortex_distribution import compute_vortex_distribution
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_RHS_matrix              import compute_RHS_matrix
#from SUAVE.Plots.Vehicle_Plots import plot_vehicle_vlm_panelization
#from SUAVE.
import matplotlib.cm as cm 
    
def main():
    # This script shows results of the VLM for a given wing geometry. It plots CL v Alpha, as well as velocities induced in the xy and yz planes.
    #-------------------------------------------------------------------------
    #          Specify Flight Conditions:
    #-------------------------------------------------------------------------    
    cruise_speed = 40 # m/s
    aoa          = np.array([[2 * Units.deg]])
    state        = cruise_conditions(cruise_speed,aoa)
    
    #-------------------------------------------------------------------------
    #          Choose wing and propeller setup:
    #-------------------------------------------------------------------------
    vehicle  = vehicle_setup(state.conditions) 
        
    # --------------------------------------------------------------------------------
    #          Settings for VLM:  
    # --------------------------------------------------------------------------------
    vortices   = 10
    
    VLM_settings         = Data()
    VLM_settings.number_panels_spanwise    = 1#0*10#5 #vortices**2   # Number of spanwise panels
    VLM_settings.number_panels_chordwise   = vortices  #              # Number of chordwise panels
    
    
    ##-------------------------------------------------------------------------------------------
    ##          CL and Downwash Plots for Validating VLM:
    ##------------------------------------------------------------------------------------------- 
    CL_downwash_validation(vehicle,state,VLM_settings)
    
    
    # --------------------------------------------------------------------------------
    #          Calling the VLM:  
    # --------------------------------------------------------------------------------    
    VD_geom, MCM, C_mn, gammaT, cl_y, CL, CD   = wing_VLM(vehicle, state, VLM_settings)
    VD                                         = copy.deepcopy(VD_geom)
    
    #-------------------------------------------------------------------------------------------
    #          Updating Vortex Distribution Matrix to Solve for Arbitrary Control Point (x,y,z):
    #-------------------------------------------------------------------------------------------
    #xy_plane_results(vehicle, VD, VLM_settings, state, C_mn, MCM, gammaT)
    yz_plane_results(vehicle, VD, VLM_settings, state, C_mn, MCM, gammaT)
    xz_plane_results(vehicle, VD, VLM_settings, state, C_mn, MCM, gammaT)
    
    ##axes_1b = fig_1.add_subplot(1,2,2)
    ##c1b     = axes_1b.contourf(prop_x, prop_y, u_xy_plane.T)#, cmap=cm.plasma) #YlOrRd_r
    ##axes_1b.set_xlabel("Chordwise Location (x)")
    ##axes_1b.set_ylabel("Spanwise Location(y)")   
    ##axes_1b.set_title("U-velocities, z=%f" %prop_z[z_desired] + ", vortices=%i" %vortices)
    ##plt.colorbar(c1b)   
    
    #fig_2  = plt.figure()
    #axes_2a = fig_2.add_subplot(1,2,1)
    #c2a     = axes_2a.contourf(prop_y, prop_z, w_yz_plane.T,100)#, cmap=cm.plasma) #YlOrRd_r
    #axes_2a.set_xlabel("Spanwise Location (y)")
    #axes_2a.set_ylabel("Vertical Location (z)")   
    #axes_2a.set_title("W-velocities, x=%f" %prop_x[x_fixed] + ", vortices=%i" %vortices)
    #plt.colorbar(c2a)
    
    
    #x_stopLoc = (np.abs(prop_x - xstop)).argmin()
    #prop_x_plot = prop_x[0:x_stopLoc]
    #u_xz_plot   = u_xz_plane[0:x_stopLoc][:].T
    #Cp = 1-(1+u_xz_plane)**2
    
    #xwing = [0,2]
    #zwing = np.zeros(len(xwing))
    
    #fig_C2  = plt.figure()
    #axes_C2 = fig_C2.add_subplot(1,1,1)
    #cC2     = axes_C2.contourf(prop_x_plot, prop_z, Cp,100,cmap=cm.jet)#, cmap=cm.plasma) #YlOrRd_r
    #wingshape = axes_C2.plot(xwing,zwing,'k',linewidth=2) 
    #axes_C2.set_xlabel("Chordwise Location (x)")
    #axes_C2.set_ylabel("Vertical Location (z)")   
    #axes_C2.set_title("Pressure Coefficient Distribution at y=%dm" %prop_y[y_fixed])
    #plt.colorbar(cC2)   
    
    #plt.show()
    
    axes_2b = fig_2.add_subplot(1,2,2)
    c2b     = axes_2b.contourf(prop_y, prop_z, u_yz_plane.T, 100)#, cmap=cm.plasma) #YlOrRd_r
    axes_2b.set_xlabel("Spanwise Location (y)")
    axes_2b.set_ylabel("Vertical Location (z)")
    axes_2b.set_title("U-velocities, x=%f" %prop_x[x_fixed] + ", vortices=%i" %vortices)
    plt.colorbar(c2b)    
    
    
    fig_3 = plt.figure()
    axes_3a = fig_3.add_subplot(1,1,1)
    c3a = axes_3a.plot(prop_y,w2d)#/max(w2d))
    axes_3a.set_xlabel("Spanwise Location (y)")
    axes_3a.set_ylabel("W-velocity")
    axes_3a.set_title("W-velocities, x=%f" %prop_x[x_fixed] + ", z=%f" %prop_z[z_fixed] + ", vortices=%i" %vortices)
    

    wing_le = vehicle.wings.main_wing.origin[0]
    croot = vehicle.wings.main_wing.chords.root
    span = vehicle.wings.main_wing.spans.projected
    radius = vehicle.propulsors.prop_net.propeller.tip_radius
    n_sw   = VLM_settings.number_panels_spanwise
    n_cw   = VLM_settings.number_panels_chordwise
    
    n_w = 1 
    for i in range(n_w):
        x_pts = np.reshape(np.atleast_2d(VD_geom.XC[i*(n_sw*n_cw):(i+1)*(n_sw*n_cw)]).T, (n_sw,-1))
        y_pts = np.reshape(np.atleast_2d(VD_geom.YC[i*(n_sw*n_cw):(i+1)*(n_sw*n_cw)]).T, (n_sw,-1))
        #z_pts = np.reshape(np.atleast_2d(CP[6][i*(n_sw*n_cw):(i+1)*(n_sw*n_cw)]).T, (n_sw,-1))
        levals = np.linspace(-.05,0,50)
        #CS = axes_1a.plot(x_pts, y_pts,'ko')#, cmap = 'jet',levels=levals,extend='both')
        #CS2 = axes_1a.plot(x_pts, -y_pts,'ko')#, cmap = 'jet',levels=levals,extend='both')
        #wing_shape = axes_1a.plot([0+wing_le,croot+wing_le, croot+wing_le, 0+wing_le,0+wing_le], [-span/2,-span/2,span/2,span/2, -span/2],'k')  
        #CSb = axes_1b.plot(x_pts, y_pts,'ko')#, cmap = 'jet',levels=levals,extend='both')
        #CS2b = axes_1b.plot(x_pts, -y_pts,'ko')#, cmap = 'jet',levels=levals,extend='both')        
        #wing_shape2 = axes_1b.plot([0+wing_le,croot+wing_le, croot+wing_le, 0+wing_le,0+wing_le], [-span/2,-span/2,span/2,span/2, -span/2],'k')  
        wing_shape2a = axes_2a.plot([-span/2,span/2],[0+wing_le,0+wing_le],'r', linewidth=2)
        wing_shape2b = axes_2b.plot([-span/2,span/2],[0+wing_le,0+wing_le],'r', linewidth=2)
            
        # Display propeller at location [wing_te+0.5, 3m, 0]:
        prop_x_center = 2.5
        prop_y_center = 3.0
        prop_z_center = 0.0    
        theta = np.linspace(0,2*np.pi,30)
        yp1 = [radius*np.cos(theta[i]) + prop_y_center for i in range(len(theta))] 
        yp2 = [radius*np.cos(theta[i]) - prop_y_center for i in range(len(theta))] 
        zp = [radius*np.sin(theta[i]) + prop_z_center for i in range(len(theta))] 
        prop_shape_1a = axes_2a.plot(yp1,zp,'r',linewidth=2) 
        prop_shape_2a = axes_2a.plot(yp2,zp,'r',linewidth=2)  
        prop_shape_1b = axes_2b.plot(yp1,zp,'r',linewidth=2) 
        prop_shape_2b = axes_2b.plot(yp2,zp,'r',linewidth=2)         
    
    plt.show()
    n_w = 1  
    #------------------------------------------------------------------------------------
    #         Plot of velocity at blade element versus blade element angle:
    #------------------------------------------------------------------------------------
    prop_x_center = np.array([vehicle.wings.main_wing.origin[0] + prop_x[x_fixed]])
    prop_y_center = np.array([3.0])
    prop_z_center = np.array([0.0])

    theta = np.linspace(0,6*np.pi,24*3)
    u_pts = [0 for j in range(len(theta))]
    v_pts = [0 for j in range(len(theta))]
    w_pts = [0 for j in range(len(theta))]
    u_pts2 = [0 for j in range(len(theta))]
    v_pts2 = [0 for j in range(len(theta))]
    w_pts2 = [0 for j in range(len(theta))]    

        
    for i in range(len(theta)):
        #theta = 2*np.pi/discretizations
        z_loc1 = np.array([prop_z_center + radius*np.sin(theta[i])])
        y_loc1 = np.array([prop_y_center + radius*np.cos(theta[i])])
        be_loc1 = [prop_x_center, y_loc1, z_loc1]
        
        z_loc2 = np.array([prop_z_center + 0.5*radius*np.sin(theta[i])])
        y_loc2 = np.array([prop_y_center + 0.5*radius*np.cos(theta[i])])
        be_loc2 = [prop_x_center, y_loc2, z_loc2]
        
        C_mn_1, u, v, w, prop_val   = VLM_velocity_sweep(VD, be_loc1, n_sw, n_cw, aoa, mach, C_mn, MCM, gammaT)
        C_mn_2, u2,v2,w2, prop_val2 = VLM_velocity_sweep(VD, be_loc2, n_sw, n_cw, aoa, mach, C_mn, MCM, gammaT)
    
        if w>10 or w<-10:
            w_pts[i] = w_pts[i-1]
            u_pts[i] = u_pts[i-1]
            v_pts[i] = v_pts[i-1]
            w_pts2[i] = w_pts2[i-1]
            u_pts2[i] = u_pts2[i-1]
            v_pts2[i] = v_pts2[i-1]            
        else:
            u_pts[i] = u
            v_pts[i] = v
            w_pts[i] = w
            u_pts2[i] = u2
            v_pts2[i] = v2
            w_pts2[i] = w2         
            
    fig_4 = plt.figure()
    axes_4 = fig_4.add_subplot(1,1,1)
    p4 = axes_4.plot(np.rad2deg(theta),u_pts, label='U for blade element at R')
    p4b = axes_4.plot(np.rad2deg(theta),w_pts, label='W for blade element at R')
    p3 = axes_4.plot(np.rad2deg(theta),u_pts2, label='U for blade element at 0.5R')
    p3b = axes_4.plot(np.rad2deg(theta),w_pts2, label='W for blade element at 0.5R')    
    axes_4.set_xlabel("Angle of Blade (deg)")
    axes_4.set_ylabel("Spanwise Location(y)")   
    axes_4.set_title("Velocity Sweep")    
    plt.legend()
    plt.show()
    
    print(w_pts)
    

      
# ----------------------------------------------------------------------
#   Supporting Functions:
# ----------------------------------------------------------------------
def cruise_conditions(cruise_speed,aoa):
    # --------------------------------------------------------------------------------    
    #          Cruise conditions  
    # --------------------------------------------------------------------------------
    rho                  = np.array([[0.81934611]])
    mu                   = np.array([[1.66118986e-05]])
    T                    = np.array([[262.16631373]])
    a                    = np.array([[np.sqrt(1.4*287.058*T[0][0])]])
    velocity_freestream  = np.array([[cruise_speed]]) #m/s this is the velocity at which the propeller is designed for based on the CLwing value being near 0.7
    mach                 = velocity_freestream/a #np.array([[0.8]]) #
    pressure             = np.array([[61660.37789389]])
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


def vehicle_setup(conditions):
    #-----------------------------------------------------------------    
    # Vehicle Initialization:
    #-----------------------------------------------------------------
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'simple_Cessna'
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    vehicle.mass_properties.takeoff    = 2550. * Units.lbf #gross weight of Cessna 172
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
    
    wing.span_efficiency         = 0.8 
    wing.origin                  = [0.,0.,0.]
    wing.vertical                = False 
    wing.symmetric               = True
  
    return wing

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
    
    prop.tip_radius                 = 2.8 * Units.feet
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
     
    prop                            = propeller_design(prop,20)    
    return prop



def wing_VLM(vehicle, state, VLM_settings):
    # --------------------------------------------------------------------------------
    #          Get Vehicle Geometry and Unpaack Settings:  
    # --------------------------------------------------------------------------------    
    #vehicle     = vehicle_setup(wing_parameters)
    Sref         = vehicle.wings.main_wing.areas.reference
    n_sw         = VLM_settings.number_panels_spanwise
    n_cw         = VLM_settings.number_panels_chordwise

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
    vehicle.vortex_distribution= VD
    RHS = compute_RHS_matrix(n_sw,n_cw,delta,phi,state.conditions,vehicle,True,False)

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







def simple_wing_textbook():
    # ------------------------------------------------------------------
    # Initialize the Vehicle
    # ------------------------------------------------------------------    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'simple_wing'     
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'  
    wing.aspect_ratio            = 5.
    wing.thickness_to_chord      = 0.08 
    wing.taper                   = 1.
    wing.span_efficiency         = 0.98 
    wing.spans.projected         = 1.
    wing.chords.root             = 0.2
    wing.chords.tip              = 0.2
    wing.chords.mean_aerodynamic = 0.2
    wing.areas.reference         = 0.2  
    vehicle.reference_area       = wing.areas.reference 
    wing.twists.root             = 0.0 * Units.degrees 
    wing.twists.tip              = 0.0 * Units.degrees 
    wing.sweeps.leading_edge     = 45. * Units.degrees
    wing.origin                  = [0.,0.,0.]
    wing.aerodynamic_center      = [0,0,0]
    wing.dihedral                = 0. * Units.degrees
    wing.vertical                = False 
    wing.symmetric               = True
    wing.dynamic_pressure_ratio  = 0.9 
    
    segment = SUAVE.Components.Wings.Segment() 
    segment.tag                   = 'root_segment'
    segment.percent_span_location = 0.0
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 1.0
    segment.dihedral_outboard     = 0.  * Units.degrees
    segment.sweeps.leading_edge   = 45. * Units.degrees
    wing.Segments.append(segment)         
    
    segment = SUAVE.Components.Wings.Segment() 
    segment.tag                   = 'tip_segment'
    segment.percent_span_location = 1
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 1.
    segment.dihedral_outboard     = 0 * Units.degrees
    segment.sweeps.quarter_chord  = 0 * Units.degrees
    wing.Segments.append(segment) 

    # add to vehicle
    vehicle.append_component(wing) 
    return vehicle
    


def Cessna_wing():
    #-------------------------------------------------------------------------
    #          Variables:
    #-------------------------------------------------------------------------    
    span         = 10.9
    croot        = 1.6256
    ctip         = 1.1303
    sweep_le     = 0.0
    sref         = 174.0*Units.feet **2      
    twist_root   = 0.0 * Units.degrees 
    twist_tip    = 0.0 * Units.degrees   
    dihedral     = 0.0 * Units.degrees
    # ------------------------------------------------------------------
    # Initialize the Vehicle and Main Wing
    # ------------------------------------------------------------------    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'simple_wing'           
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'        
    
    wing.aspect_ratio            = 7.52
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
    
    vehicle.append_component(wing) 
    return vehicle



def CL_downwash_validation(vehicle, state, VLM_settings):
    #-------------------------------------------------------------------------------------------
    #          Range of Angle of Attack:
    #------------------------------------------------------------------------------------------- 
    aoa_vec         = np.linspace(-2,8,20) * Units.deg
    N               = 16
    prop_x          = np.array([2.0])
    prop_y          = np.linspace(0,4,N)
    prop_y2          = np.linspace(0.01,3.4,N)
    prop_z          = np.array([0.0])
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
            C_mn_2, u,v,w,prop_val  = VLM_velocity_sweep(VD,prop_loc,VLM_settings,state_new,C_mn,MCM,gammaT)            
            w_vlm[i,b]              = w
        
    w_momentum_theory = (2*CL_flat_plate*vehicle.wings.main_wing.areas.reference)/(np.pi*vehicle.wings.main_wing.spans.projected**2)
    aoa_vec           = aoa_vec*180/np.pi
    w_vlm_avg         = np.mean(-w_vlm, axis=1)
    CL_vec            = np.mean(CL_vec, axis=1) #CL_vec[:,0]
    CDi_vec           = np.mean(CDi_vec, axis=1)
    w_momentum_theory = (2*CL_vec*vehicle.wings.main_wing.areas.reference)/(np.pi*vehicle.wings.main_wing.spans.projected**2)
    
    span_y = 
    fig_val1  = plt.figure()
    axes_val1 = fig_val1.add_subplot(1,1,1)
    axes_val1.plot(span_y, span_w,'--', linewidth=4)
    axes_val1.set_xlabel("Angle of Attack (deg)")
    axes_val1.set_ylabel("CL")
    axes_val1.set_title("CL v Alpha, Flat Plate")
    plt.legend()
    plt.grid()
    plt.show()
    
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
    
def VLM_velocity_sweep(VD,prop_location,VLM_settings,state,C_mn,MCM,gammaT):
    aoa  = state.conditions.aerodynamics.angle_of_attack
    mach = state.conditions.freestream.mach_number
    
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
        C_mn, DW_mn = compute_induced_velocity_matrix(VD,VLM_settings.number_panels_spanwise,VLM_settings.number_panels_chordwise,aoa,mach)
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


def parse_velocities_2d(u,v,w,prop_loc,prop_loc_values,vehicle, desired_cross_section_value):
    
    prop_x = prop_loc[0]
    prop_y = prop_loc[1]
    prop_z = prop_loc[2] 
    
    u2d = np.zeros(len(prop_y))
    v2d = np.zeros(len(prop_y))
    w2d = np.zeros(len(prop_y))

    w_avg = 0
    z_desired_val = desired_cross_section_value[2]
    z_fixed = (np.abs(prop_z - z_desired_val)).argmin()   
    x_desired_val = desired_cross_section_value[0]
    x_fixed = (np.abs(prop_x - x_desired_val)).argmin()      

    
    # Generate results along y:
    for y in range(len(prop_y)):
        # Velocities for plotting the x-y plane results with z=0:
       
        prop_loc_2d_ysweep = [prop_x[x_fixed], prop_y[y], prop_z[z_fixed]]
        loc_2d_ysweep = np.where((prop_loc_values == prop_loc_2d_ysweep).all(axis=1))
        u2d[y] = u[loc_2d_ysweep[0][0]]
        v2d[y] = v[loc_2d_ysweep[0][0]]
        w2d[y] = w[loc_2d_ysweep[0][0]]
    return u2d, v2d, w2d, z_fixed, x_fixed


def parse_velocities(u,v,w,prop_loc,prop_loc_values,vehicle, desired_cross_section_value):
    
    prop_x = prop_loc[0]
    prop_y = prop_loc[1]
    prop_z = prop_loc[2]
    
    u_xy_plane = np.array([[0.0 for j in range(len(prop_y))] for k in range(len(prop_x))])
    v_xy_plane = np.array([[0.0 for j in range(len(prop_y))] for k in range(len(prop_x))])
    w_xy_plane = np.array([[0.0 for j in range(len(prop_y))] for k in range(len(prop_x))])
    u_yz_plane = np.array([[0.0 for j in range(len(prop_z))] for k in range(len(prop_y))])
    v_yz_plane = np.array([[0.0 for j in range(len(prop_z))] for k in range(len(prop_y))])
    w_yz_plane = np.array([[0.0 for j in range(len(prop_z))] for k in range(len(prop_y))])
    u_xz_plane = np.array([[0.0 for j in range(len(prop_z))] for k in range(len(prop_x))])
    v_xz_plane = np.array([[0.0 for j in range(len(prop_z))] for k in range(len(prop_x))])
    w_xz_plane = np.array([[0.0 for j in range(len(prop_z))] for k in range(len(prop_x))])    
    
    u2d = np.zeros(len(prop_y))
    v2d = np.zeros(len(prop_y))
    w2d = np.zeros(len(prop_y))

    w_avg = 0
    
    if len(prop_z)==1:
        z_desired_val = desired_cross_section_value
        z_fixed = (np.abs(prop_z - z_desired_val)).argmin()
        
        # Generate results for the xy-plane:
        for x in range(len(prop_x)):
            for y in range(len(prop_y)):
                # Velocities for plotting the x-y plane results with z=0:
                prop_loc_xy_plane = [prop_x[x], prop_y[y], prop_z[z_fixed]]
                
                # Find location of prop_loc_values that matches prop_loc:
                loc_xy_plane = np.where((prop_loc_values == prop_loc_xy_plane).all(axis=1))
                u_xy_plane[x][y] = u[loc_xy_plane]
                v_xy_plane[x][y] = v[loc_xy_plane]
                w_xy_plane[x][y] = w[loc_xy_plane]
                
                #prop_loc_2d_ysweep = [prop_x[x_fixed], prop_y[y], prop_z[z_fixed]]
                #loc_2d_ysweep = np.where((prop_loc_values == prop_loc_2d_ysweep).all(axis=1))
                #u2d[y] = u[loc_2d_ysweep[0][0]]
                #v2d[y] = v[loc_2d_ysweep[0][0]]
                #w2d[y] = w[loc_2d_ysweep[0][0]]
                #if prop_y[y]<vehicle.wings.main_wing.spans.projected/2 and 0.01<prop_y[y] and x==len(prop_x)/2:
                    #w_avg += w2d[y]
        return u_xy_plane, w_xy_plane, z_fixed
            
                
    if len(prop_x)==1:
        x_desired_val = desired_cross_section_value
        x_fixed = (np.abs(prop_x - x_desired_val)).argmin()
        
        # Generate results for the yz-plane:
        for y in range(len(prop_y)):
                for z in range(len(prop_z)):
                    # Velocities for plotting the y-z plane results with x=1m behind TE:
                    prop_loc_desired = [prop_x[x_fixed], prop_y[y], prop_z[z]]
                    # Find location of prop_loc_values that matches prop_loc:
                    loc_yz = np.where((prop_loc_values == prop_loc_desired).all(axis=1))
                    
                    u_yz_plane[y][z] = u[loc_yz]
                    v_yz_plane[y][z] = v[loc_yz]
                    w_yz_plane[y][z] = w[loc_yz]
        return u_yz_plane, w_yz_plane, x_fixed
    
    # Generate results for the xz-plane:
    if len(prop_y)==1:
        y_desired_val = desired_cross_section_value
        y_fixed = (np.abs(prop_y - y_desired_val)).argmin()
        
        for x in range(len(prop_x)):
            for z in range(len(prop_z)):
                    prop_loc_desired = [prop_x[x], prop_y[y_fixed], prop_z[z]]
                    # Find location of prop_loc_values that matches prop_loc:
                    loc_xz = np.where((prop_loc_values == prop_loc_desired).all(axis=1))

                    u_xz_plane[x][z] = u[loc_xz] [0]
                    v_xz_plane[x][z] = v[loc_xz] [0]
                    w_xz_plane[x][z] = w[loc_xz] [0]
        return u_xz_plane, w_xz_plane, y_fixed
    
    
    return #w_xy_plane, u_xy_plane, w_yz_plane, u_yz_plane, u_xz_plane, fixed_plane_points, xstop, w_avg

def xy_plane_results(vehicle, VD, VLM_settings, state, C_mn, MCM, gammaT):
    aoa = state.conditions.aerodynamics.angle_of_attack
    #N_eval = 2 # number of spanwise evaluation points per n_sw
    #n               = np.linspace(VLM_settings.n_sw+1,0,(VLM_settings.n_sw*N_eval+1))         # vectorize
    #thetan          = n*(np.pi/2)/(VLM_settings.n_sw+1)                 # angular stations
    #y_coords_eval   = 0.5*vehicle.wings.main_wing.spans.projected*np.cos(thetan)                  # y locations based on the angular spacing 
    #y_outer = np.arange(-1.5*(vehicle.wings.main_wing.spans.projected/2), -(vehicle.wings.main_wing.spans.projected/2),0.2)
    #y_eval = np.concatenate((-y_outer,-y_coords_eval, y_coords_eval,y_outer), axis=None)
   
    y_step_size = 0.30769230999999975 # ehicle.wings.main_wing.spans.projected/41 # 
    x_step_size = vehicle.wings.main_wing.chords.root/(VLM_settings.n_cw*4)
    #radius     = 30.0 * Units.inches
    #prop_x = np.linspace(-1.5, 13*radius, 40) #40
    #prop_y = np.linspace(-1.5*(vehicle.wings.main_wing.spans.projected/2), 1.5*(vehicle.wings.main_wing.spans.projected/2),40) #
    #prop_z = np.array([0]) #np.linspace(-5*radius, 5*radius, 31) #31
    
    prop_x = np.arange(-1.5,8.0,x_step_size) #np.linspace(-1.5, 8, 40) #40
    prop_y = np.arange(-1.5*(vehicle.wings.main_wing.spans.projected/2), 1.5*(vehicle.wings.main_wing.spans.projected/2),y_step_size)#1.55*(vehicle.wings.main_wing.spans.projected/2), step_size*2) #np.linspace(-1.5*(vehicle.wings.main_wing.spans.projected/2), 1.5*(vehicle.wings.main_wing.spans.projected/2),40) #
    prop_z = np.array([0.0])#np.arange(-1,1,step_size/30) #np.linspace(-2, 2, 31) #31
    prop_loc = [prop_x, prop_y, prop_z]
    
    # Compute velocities:
    # u,v,w:velocities corresponding to triple for loop with x on outer loop, then y then z
    C_mn, u, v, w, prop_loc_values = VLM_velocity_sweep(VD, prop_loc, VLM_settings, state, C_mn, MCM, gammaT)    
    
    #--------------------------------------------------------------------------------------------------------------
    #   PARSE THE DATA OUTPUT FROM VLM_velocity_sweep, and PLOT velocity field in all three planes
    #   At a fixed z- or x- location, determine the induced velocities from the wing in the x-y and y-z planes:
    #--------------------------------------------------------------------------------------------------------------
    # Cross section value of interest:
    z_desired_val = 0.0

    u_xy_plane, w_xy_plane, z_fixed = \
        parse_velocities(u,v,w,prop_loc,prop_loc_values,vehicle,z_desired_val)
    
    x_desired_val = 2
    y_desired_val = 0 #not used
    desired_cross_section_value = np.array([x_desired_val,y_desired_val,z_desired_val])
    u2d, v2d, w2d, z_fixed, x_fixed = parse_velocities_2d(u,v,w,prop_loc,prop_loc_values,vehicle, desired_cross_section_value)
    
    #-------------------------------------------------------------------------------------------
    #      Plot the induced velocities in the x-y and y-z planes:
    #-------------------------------------------------------------------------------------------  
    #V-Vinf/Vinf
    u_wing_tot_xz = np.sqrt(np.square(u_xy_plane+np.ones_like(u_xy_plane))+np.square(w_xy_plane))-np.ones_like(u_xy_plane)

    fig_1  = plt.figure()
    axes_1a = fig_1.add_subplot(1,1,1)
    c1a     = axes_1a.contourf(prop_x, prop_y, u_xy_plane.T,100)#, cmap=cm.jet) #YlOrRd_r
    axes_1a.set_xlabel("Chordwise Location (x)")
    axes_1a.set_ylabel("Spanwise Location(y)")   
    axes_1a.set_title("$u_{a,wing}$, z=%1.2f" %prop_z[z_fixed] + ", AoA=%i" %(aoa[0][0]/Units.deg))
    plt.colorbar(c1a)
    
    fig_2  = plt.figure()
    axes_2a = fig_2.add_subplot(1,1,1)
    c2a     = axes_2a.contourf(prop_x, prop_y, w_xy_plane.T,100)#, cmap=cm.jet) #YlOrRd_r
    axes_2a.set_xlabel("Chordwise Location (x)")
    axes_2a.set_ylabel("Spanwise Location(y)")   
    axes_2a.set_title("$u_{t,wing}$, z=%1.2f" %prop_z[z_fixed] + ", AoA=%i" %(aoa[0][0]/Units.deg))
    plt.colorbar(c2a)       
    
    fig_3 = plt.figure()
    axes_3a = fig_3.add_subplot(1,1,1)
    c3a     = axes_3a.plot(prop_y, w2d)#, cmap=cm.jet) #YlOrRd_r
    axes_3a.set_xlabel("Spanwise Location (y)")
    axes_3a.set_ylabel("$\dfrac{w}{V_\infty}$")   
    axes_3a.set_title("$u_{t,wing}$, z=%1.2f" %prop_z[z_fixed] + ", x=%1.2f" %prop_x[x_fixed] + ", AoA=%i" %(aoa[0][0]/Units.deg))
    plt.grid()
    plt.show()
    
    return

def xz_plane_results(vehicle, VD, VLM_settings, state, C_mn, MCM, gammaT):
    aoa = state.conditions.aerodynamics.angle_of_attack
    
    Nz = 80
    # Sinusoidal Spacing:
    thetaz_A = 1-np.sin(np.linspace(np.pi/2,np.pi,Nz))
    thetaz_B = np.flip(1+np.sin(np.linspace(3*np.pi/2,2*np.pi,Nz)))
    
    thetaz = 3*np.concatenate((-thetaz_B,thetaz_A[10:]),axis=None)
    
    
    step_size = 0.2
    x_step_size = vehicle.wings.main_wing.chords.root/(VLM_settings.n_cw*4)
    prop_x = np.arange(1.9,3.0,x_step_size)#step_size/2) #np.linspace(-1.5, 8, 40) #40
    prop_y = np.array([2])#np.arange(-1.5*(vehicle.wings.main_wing.spans.projected/2), 0, step_size*2)#1.55*(vehicle.wings.main_wing.spans.projected/2), step_size*2) #np.linspace(-1.5*(vehicle.wings.main_wing.spans.projected/2), 1.5*(vehicle.wings.main_wing.spans.projected/2),40) #
    prop_z = (thetaz) #np.arange(-1.0,1.1,step_size/160) #np.linspace(-2, 2, 31) #31
    prop_loc = [prop_x, prop_y, prop_z]
    
    # Compute velocities:
    # u,v,w:velocities corresponding to triple for loop with x on outer loop, then y then z
    C_mn, u, v, w, prop_loc_values = VLM_velocity_sweep(VD, prop_loc, VLM_settings, state, C_mn, MCM, gammaT)    
    
    #--------------------------------------------------------------------------------------------------------------
    #   PARSE THE DATA OUTPUT FROM VLM_velocity_sweep, and PLOT velocity field in all three planes
    #   At a fixed z- or x- location, determine the induced velocities from the wing in the x-y and y-z planes:
    #--------------------------------------------------------------------------------------------------------------
    # Cross section value of interest:
    y_desired_val = 2.0

    u_xz_plane, w_xz_plane, y_fixed = \
        parse_velocities(u,v,w,prop_loc,prop_loc_values,vehicle,y_desired_val)

    
    #-------------------------------------------------------------------------------------------
    #      Plot the induced velocities in the x-y and y-z planes:
    #-------------------------------------------------------------------------------------------  
    velocity = state.conditions.freestream.velocity
    #V/Vinf
    u_wing_tot_xz = np.sqrt(np.square(velocity*(1+u_xz_plane))+np.square(velocity*w_xz_plane))/velocity
    
    xwing = [0,2]
    zwing = np.zeros(len(xwing))    

    fig_1  = plt.figure()
    axes_1a = fig_1.add_subplot(1,1,1)
    c1a     = axes_1a.contourf(prop_x, prop_z, u_xz_plane.T,100, cmap=cm.jet) #YlOrRd_r
    wingshape = axes_1a.plot(xwing,zwing,'k',linewidth=2) 
    axes_1a.set_xlabel("Chordwise Location (x)")
    axes_1a.set_ylabel("Vertical Location(z)")   
    axes_1a.set_title("$u_{a,wing}$, y=%d" %prop_y[y_fixed] + ", AoA=%i" %(aoa[0][0]/Units.deg))
    plt.colorbar(c1a)
    
    fig_2  = plt.figure()
    axes_2a = fig_2.add_subplot(1,1,1)
    c2a     = axes_2a.contourf(prop_x, prop_z, w_xz_plane.T,100, cmap=cm.jet) #YlOrRd_r
    wingshape = axes_2a.plot(xwing,zwing,'k',linewidth=2) 
    axes_2a.set_xlabel("Chordwise Location (x)")
    axes_2a.set_ylabel("Vertical Location(z)")   
    axes_2a.set_title("Downwash, y=%d" %prop_y[y_fixed] + ", AoA=%i" %(aoa[0][0]/Units.deg))
    plt.colorbar(c2a)    
    
    Cp = 1-(u_wing_tot_xz)**2#1-((1+u_xz_plane)*np.cos(aoa[0][0])+w_xz_plane*np.sin(aoa[0][0]))**2 #1-(1+u_wing_tot_xz)**2 #
   
    
    fig_3  = plt.figure()
    axes_3 = fig_3.add_subplot(1,1,1)
    cC3    = axes_3.contourf(prop_x, prop_z, Cp.T,100,cmap=cm.jet)#, cmap=cm.plasma) #YlOrRd_r
    wingshape = axes_3.plot(xwing,zwing,'k',linewidth=2) 
    axes_3.set_xlabel("Chordwise Location (x)")
    axes_3.set_ylabel("Vertical Location (z)")   
    axes_3.set_title("Pressure Coefficient Distribution at y=%dm" %prop_y[y_fixed])
    plt.colorbar(cC3)       
    
    plt.show()
    
    return

def yz_plane_results(vehicle, VD, VLM_settings, state, C_mn, MCM, gammaT):
    aoa = state.conditions.aerodynamics.angle_of_attack
    Nz = 80*3
    # Sinusoidal Spacing:
    thetaz_A = 1-np.sin(np.linspace(np.pi/2,np.pi,Nz))
    thetaz_B = np.flip(1+np.sin(np.linspace(3*np.pi/2,2*np.pi,Nz)))
    
    thetaz = 3*np.concatenate((-thetaz_B,thetaz_A[1:]),axis=None)
    
    y_step_size = 0.30769230999999975
    prop_x = np.array([2.25]) #np.array([vehicle.wings.main_wing.chords.root+1])#.25])#np.arange(-1.5,8.0,step_size/2) #np.linspace(-1.5, 8, 40) #40
    prop_y = np.arange(-1.5*(vehicle.wings.main_wing.spans.projected/2), 1.5*(vehicle.wings.main_wing.spans.projected/2), y_step_size)#1.55*(vehicle.wings.main_wing.spans.projected/2), step_size*2) #np.linspace(-1.5*(vehicle.wings.main_wing.spans.projected/2), 1.5*(vehicle.wings.main_wing.spans.projected/2),40) #
    prop_z = thetaz #np.arange(-3.5,3.5,step_size/30) #np.linspace(-2, 2, 31) #31
    prop_loc = [prop_x, prop_y, prop_z]
    
    # Compute velocities:
    # u,v,w:velocities corresponding to triple for loop with x on outer loop, then y then z
    C_mn, u, v, w, prop_loc_values = VLM_velocity_sweep(VD, prop_loc, VLM_settings, state, C_mn, MCM, gammaT)    
    
    #--------------------------------------------------------------------------------------------------------------
    #   PARSE THE DATA OUTPUT FROM VLM_velocity_sweep, and PLOT velocity field in all three planes
    #   At a fixed z- or x- location, determine the induced velocities from the wing in the x-y and y-z planes:
    #--------------------------------------------------------------------------------------------------------------
    # Cross section values of interest:
    x_desired_val = 2.2#2.25 #1+ vehicle.wings.main_wing.chords.root

    u_yz_plane, w_yz_plane, x_fixed = \
        parse_velocities(u,v,w,prop_loc,prop_loc_values,vehicle,x_desired_val)
    
    
    #-------------------------------------------------------------------------------------------
    #      Plot the induced velocities in the x-y and y-z planes:
    #-------------------------------------------------------------------------------------------  
    #V-Vinf/Vinf
    u_wing_tot_yz = np.sqrt(np.square(u_yz_plane+np.ones_like(u_yz_plane))+np.square(w_yz_plane))-np.ones_like(u_yz_plane)
       
    fig_1  = plt.figure()
    axes_1a = fig_1.add_subplot(1,1,1)
    c1a     = axes_1a.contourf(prop_y, prop_z, u_yz_plane.T,100)#, cmap=cm.jet) #YlOrRd_r
    axes_1a.set_xlabel("Spanwise Location (y)")
    axes_1a.set_ylabel("Vertical Location(z)")   
    axes_1a.set_title("$u_{a,wing}$, x=%1.2f" %prop_x[x_fixed] + ", AoA=%i" %(aoa[0][0]/Units.deg))
    plt.colorbar(c1a)
    
    fig_2  = plt.figure()
    axes_2a = fig_2.add_subplot(1,1,1)
    c2a     = axes_2a.contourf(prop_y, prop_z, w_yz_plane.T,100)#, cmap=cm.jet) #YlOrRd_r
    axes_2a.set_xlabel("Spanwise Location (y)")
    axes_2a.set_ylabel("Vertical Location(z)")   
    axes_2a.set_title("Downwash, x=%1.2f" %prop_x[x_fixed] + ", AoA=%i" %(aoa[0][0]/Units.deg))
    plt.colorbar(c2a)    
    
    plt.show()
    
    return

if __name__ == '__main__':    
    main()