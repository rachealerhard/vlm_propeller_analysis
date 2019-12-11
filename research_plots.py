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
from SUAVE.Plots.Vehicle_Plots import plot_vehicle_vlm_panelization
import matplotlib.cm as cm 
    
def main():
    #-------------------------------------------------------------------------
    #          Choose wing setup:
    #-------------------------------------------------------------------------
    vehicle = flat_plate_wing() #simple_wing_textbook() #
    aoa     = np.array([[2.0 * Units.deg]])
    state   = cruise_conditions(aoa)    
    mach    = state.conditions.freestream.mach_number
    
    radius  = 30.0 * Units.inches
    vortices   = 7
    n_sw       = 1#vortices **2
    n_cw       = vortices    
        
    # --------------------------------------------------------------------------------
    #          Settings and Calling for VLM:  
    # --------------------------------------------------------------------------------
    VLM_settings                           = Data()
    VLM_settings.number_panels_spanwise    = n_sw
    VLM_settings.number_panels_chordwise   = n_cw
    
    VD_geom, MCM, C_mn, GAMMA, gammaT, B, cl_y, CL   = wing_VLM(vehicle, state, VLM_settings)
    VD                                               = copy.deepcopy(VD_geom)
    
    #-------------------------------------------------------------------------------------------
    #          CL and Downwash Plots for Validating VLM:
    #------------------------------------------------------------------------------------------- 
    aoa_vec                                       = np.linspace(-2,8,101)
    prop_loc                                      = [[2.0], [0.2], [0.0]]
    CL_vec, CL_flat_plate_theory, w_theory, w_VLM = CL_downwash_validation(vehicle,VLM_settings,aoa_vec,mach,prop_loc,n_sw,n_cw)
    
    ex_loc = np.where(aoa_vec==2.0)
    w_t_ex = 2*CL_flat_plate_theory[ex_loc]*vehicle.wings.main_wing.areas.reference/(np.pi*vehicle.wings.main_wing.spans.projected**2)
    w_vlm_ex = w_VLM[ex_loc]
    
    #w_elliptical = CL_flat_plate_theory/(np.pi*4)
    
    #Plotting CL distribution of VLM v. Theory:
    fig_val1  = plt.figure()
    axes_val1 = fig_val1.add_subplot(1,1,1)
    axes_val1.plot(aoa_vec, CL_flat_plate_theory,'b', label="Flat Plate Theory")
    axes_val1.plot(aoa_vec, CL_vec,'r', label="Flat Plate VLM Result")
    axes_val1.set_xlabel("Angle of Attack (deg)")
    axes_val1.set_ylabel("CL")
    axes_val1.set_title("CL v Alpha, Flat Plate")
    plt.legend()
    
    #Plotting downwash of VLM v. Momentum Theory: 
    fig_val2 = plt.figure()
    axes_val2 = fig_val2.add_subplot(1,1,1)
    axes_val2.plot(aoa_vec, w_theory,'b', label="Momentum Theory")
    #axes_val2.plot(aoa_vec, -w_elliptical,'k', label="Elliptical Theory")
    axes_val2.plot(aoa_vec, -w_VLM,'r', label="Flat Plate VLM Result")
    axes_val2.set_xlabel("Angle of Attack (deg)")
    axes_val2.set_ylabel("Downwash Velocity")
    axes_val2.set_title("Downwash v Alpha, Flat Plate")  
    
    plt.legend()
    plt.show()     
    
    #-------------------------------------------------------------------------------------------
    #          Updating Vortex Distribution Matrix to Solve for Arbitrary Control Point (x,y,z):
    #-------------------------------------------------------------------------------------------
    prop_x = np.linspace(-1.5, 13*radius, 40) #40
    prop_y = np.linspace(-1.5*(vehicle.wings.main_wing.spans.projected/2), 1.5*(vehicle.wings.main_wing.spans.projected/2),40) #
    prop_z = np.linspace(-5*radius, 5*radius, 31) #31
    
    #--------------------------------------------------------------------------------------------------------------
    #      At a fixed z- or x- location, determine the induced velocities from the wing in the x-y and y-z planes:
    #--------------------------------------------------------------------------------------------------------------
    u_xy_plane = np.array([[0.0 for j in range(len(prop_y))] for k in range(len(prop_x))])
    v_xy_plane = np.array([[0.0 for j in range(len(prop_y))] for k in range(len(prop_x))])
    w_xy_plane = np.array([[0.0 for j in range(len(prop_y))] for k in range(len(prop_x))])
    u_yz_plane = np.array([[0.0 for j in range(len(prop_z))] for k in range(len(prop_y))])
    v_yz_plane = np.array([[0.0 for j in range(len(prop_z))] for k in range(len(prop_y))])
    w_yz_plane = np.array([[0.0 for j in range(len(prop_z))] for k in range(len(prop_y))])    
    
    u2d = np.zeros(len(prop_y))
    v2d = np.zeros(len(prop_y))
    w2d = np.zeros(len(prop_y))
    
    prop_loc = [prop_x, prop_y, prop_z]
    
    # u,v,w:velocities corresponding to tripple for loop with x on outer loop, then y then z
    C_mn, u, v, w, prop_loc_values = VLM_velocity_sweep(VD, prop_loc, n_sw, n_cw, aoa,mach, C_mn, MCM, GAMMA, gammaT, B)
    
    # Cross section values of interest:
    x_desired_val = 2.5 #1+ vehicle.wings.main_wing.chords.root
    x_desired = (np.abs(prop_x - x_desired_val)).argmin()
    z_desired = math.floor(len(prop_z)/2)
    
    for x in range(len(prop_x)):
        for y in range(len(prop_y)):
            # Velocities for plotting the x-y plane results with z=0:
            prop_loc_xy_plane = [prop_x[x], prop_y[y], prop_z[z_desired]]
            # Find location of prop_loc_values that matches prop_loc:
            loc_xy_plane = np.where((prop_loc_values == prop_loc_xy_plane).all(axis=1))
            u_xy_plane[x][y] = u[loc_xy_plane]
            v_xy_plane[x][y] = v[loc_xy_plane]
            w_xy_plane[x][y] = w[loc_xy_plane]
            
            prop_loc_2d_ysweep = [prop_x[x_desired], prop_y[y], prop_z[z_desired]]
            loc_2d_ysweep = np.where((prop_loc_values == prop_loc_2d_ysweep).all(axis=1))
            u2d[y] = u[loc_2d_ysweep[0][0]]
            v2d[y] = v[loc_2d_ysweep[0][0]]
            w2d[y] = w[loc_2d_ysweep[0][0]]
            
            for z in range(len(prop_z)):
                # Velocities for plotting the y-z plane results with x=1m behind TE:
                prop_loc_desired = [prop_x[x_desired], prop_y[y], prop_z[z]]
                # Find location of prop_loc_values that matches prop_loc:
                loc = np.where((prop_loc_values == prop_loc_desired).all(axis=1))
                
                u_yz_plane[y][z] = u[loc]
                v_yz_plane[y][z] = v[loc]
                w_yz_plane[y][z] = w[loc]
    
    
    #-------------------------------------------------------------------------------------------
    #      Plot the induced velocities in the x-y and y-z planes:
    #-------------------------------------------------------------------------------------------    
    fig_1  = plt.figure()
    axes_1a = fig_1.add_subplot(1,1,1)
    c1a     = axes_1a.contourf(prop_x, prop_y, w_xy_plane.T)#, cmap=cm.plasma) #YlOrRd_r
    axes_1a.set_xlabel("Chordwise Location (x)")
    axes_1a.set_ylabel("Spanwise Location(y)")   
    axes_1a.set_title("W-velocities, z=%d" %prop_z[z_desired] + ", AoA=%i" %(aoa[0][0]*180/np.pi))
    plt.colorbar(c1a)
    
    #axes_1b = fig_1.add_subplot(1,2,2)
    #c1b     = axes_1b.contourf(prop_x, prop_y, u_xy_plane.T)#, cmap=cm.plasma) #YlOrRd_r
    #axes_1b.set_xlabel("Chordwise Location (x)")
    #axes_1b.set_ylabel("Spanwise Location(y)")   
    #axes_1b.set_title("U-velocities, z=%f" %prop_z[z_desired] + ", vortices=%i" %vortices)
    #plt.colorbar(c1b)    
    
    
    fig_2  = plt.figure()
    axes_2a = fig_2.add_subplot(1,2,1)
    c2a     = axes_2a.contourf(prop_y, prop_z, w_yz_plane.T,40)#, cmap=cm.plasma) #YlOrRd_r
    axes_2a.set_xlabel("Spanwise Location (y)")
    axes_2a.set_ylabel("Vertical Location (z)")   
    axes_2a.set_title("W-velocities, x=%f" %prop_x[x_desired] + ", vortices=%i" %vortices)
    plt.colorbar(c2a)
    
    axes_2b = fig_2.add_subplot(1,2,2)
    c2b     = axes_2b.contourf(prop_y, prop_z, u_yz_plane.T, 40)#, cmap=cm.plasma) #YlOrRd_r
    axes_2b.set_xlabel("Spanwise Location (y)")
    axes_2b.set_ylabel("Vertical Location (z)")
    axes_2b.set_title("U-velocities, x=%f" %prop_x[x_desired] + ", vortices=%i" %vortices)
    plt.colorbar(c2b)    
    
    
    fig_3 = plt.figure()
    axes_3a = fig_3.add_subplot(1,1,1)
    c3a = axes_3a.plot(prop_y,w2d)#/max(w2d))
    axes_3a.set_xlabel("Spanwise Location (y)")
    axes_3a.set_ylabel("W-velocity")
    axes_3a.set_title("W-velocities, x=%f" %prop_x[x_desired] + ", z=%f" %prop_z[z_desired] + ", vortices=%i" %vortices)
    

    wing_le = vehicle.wings.main_wing.origin[0]
    croot = vehicle.wings.main_wing.chords.root
    span = vehicle.wings.main_wing.spans.projected
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
    prop_x_center = np.array([vehicle.wings.main_wing.origin[0] + prop_x[x_desired]])
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
        
        C_mn_1, u, v, w, prop_val   = VLM_velocity_sweep(VD, be_loc1, n_sw, n_cw, aoa, mach, C_mn, MCM, GAMMA, gammaT, B)
        C_mn_2, u2,v2,w2, prop_val2 = VLM_velocity_sweep(VD, be_loc2, n_sw, n_cw, aoa, mach, C_mn, MCM, GAMMA, gammaT, B)
    
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
    
    wing.span_efficiency         = 0.98 
    wing.origin                  = [0.,0.,0.]
    wing.vertical                = False 
    wing.symmetric               = True
    
    vehicle.append_component(wing) 
    return vehicle
    
def cruise_conditions(aoa):
    # --------------------------------------------------------------------------------    
    #          Cruise conditions  
    # --------------------------------------------------------------------------------
    rho                  = np.array([[0.365184]])
    mu                   = np.array([[0.0000143326]])
    T                    = np.array([[258]])
    P                    = 57200.0
    a                    = 322.2
    velocity_freestream  = np.array([[1.0]]) #m/s
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
  
def CL_downwash_validation(vehicle, VLM_settings, aoa_vec,mach,prop_loc,n_sw, n_cw):
    #-------------------------------------------------------------------------------------------
    #          CL Plots for Validating VLM:
    #------------------------------------------------------------------------------------------- 
    AR              = vehicle.wings.main_wing.aspect_ratio
    CL_vec          = np.zeros(len(aoa_vec))
    w_vlm           = np.zeros(len(aoa_vec))
    CL_flat_plate   = 2*np.pi*aoa_vec*(np.pi/180)*(AR/(2+AR))
    
    for i in range(len(aoa_vec)):
        aoa_val = np.array([[aoa_vec[i] * Units.deg]])
        state_test = cruise_conditions(aoa_val)
        VD, MCM, C_mn, GAMMA, gammaT, B, cl_y_out, CL_out= wing_VLM(vehicle,state_test,VLM_settings)
        CL_vec[i] = CL_out
        C_mn_2, u,v,w,prop_val = VLM_velocity_sweep(VD,prop_loc,n_sw,n_cw,np.array([aoa_vec[i]]),mach,C_mn,MCM,GAMMA,gammaT,B)
        w_vlm[i] = w
        
    w_momentum_theory = (2*CL_vec*vehicle.wings.main_wing.areas.reference)/(np.pi*vehicle.wings.main_wing.spans.projected**2)
    
    return CL_vec, CL_flat_plate, w_momentum_theory, w_vlm
    
def VLM_velocity_sweep(VD,prop_location,n_sw,n_cw,aoa,mach,C_mn,MCM,GAMMA,gammaT,B):
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
    for i in range(num_loops_required):
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
        if u[j] >= 4 or u[j] <= -4:
            u[j] = u[j-1]
        if v[j] >= 4 or v[j] <= -4:
            v[j] = v[j-1]       
        if w[j] >= 4 or w[j] <= -4:
            w[j] = w[j-1]        
            
    return C_mn, u, v, w, prop_val #C_mn, u[0,0], v[0,0], w[0,0]
    

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
    RHS = compute_RHS_matrix(VD,n_sw,n_cw,delta,phi,state.conditions,geometry)

    # --------------------------------------------------------------------------------
    #          Compute Vortex Strength:
    # --------------------------------------------------------------------------------  
    n_cp                 = VD.n_cp  
    gamma                = np.linalg.solve(A,RHS)
    GAMMA                = np.repeat(np.atleast_3d(gamma), n_cp ,axis = 2 )
    gammaT                = np.transpose(gamma) 
    
    u = (C_mn[:,:,:,0]*MCM[:,:,:,0]@gammaT)[:,:,0]
    v = (C_mn[:,:,:,1]*MCM[:,:,:,1]@gammaT)[:,:,0]
    w = (C_mn[:,:,:,2]*MCM[:,:,:,2]@gammaT)[:,:,0]  
    w_ind                = -np.sum(B*MCM[:,:,:,2]*GAMMA, axis = 2) 
     
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
    w_n_w                = np.array(np.array_split(w,n_w,axis=1))
    w_n_w_sw             = np.array(np.array_split(w,n_w*n_sw,axis=1))
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
    
    # Calculate each spanwise set of Cls and Cds, then split Cls and Cds for each wing
    cl_y                 = np.sum(np.multiply(u_n_w_sw +1,(gamma_n_w_sw*Del_Y_n_w_sw)),axis=2).T/CS
    cdi_y                = np.sum(np.multiply(-w_ind_n_w_sw,(gamma_n_w_sw*Del_Y_n_w_sw)),axis=2).T/CS
    Cl_wings             = np.array(np.split(cl_y,n_w,axis=1))
    Cd_wings             = np.array(np.split(cdi_y,n_w,axis=1))

    # total lift and lift coefficient
    L                    = np.atleast_2d(np.sum(np.multiply((1+u),gamma*Del_Y),axis=1)).T 
    CL                   = L/(0.5*Sref)   # validated form page 402-404, aerodynamics for engineers 
    
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
    #axes.plot(y, cl_elliptical[0], 'r',label="Elliptical Distribution")
    #axes.set_xlabel("Spanwise Location (y)")
    #axes.set_ylabel("Cl")
    #axes.set_title("Cl Distribution")
    #plt.legend()
    #plt.show()

    return VD, MCM, C_mn, GAMMA, gammaT, B, cl_y, CL


if __name__ == '__main__':    
    main()
