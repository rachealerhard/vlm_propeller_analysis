# Cessna_172.py
# 
# Created:  Mar 2019, M. Clarke

""" setup file for a mission with a twin prop modified C172 SP NAV III with 
propeller interaction 
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

from SUAVE.Core import Data
from SUAVE.Plots.Mission_Plots import *
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift import  VLM
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power, initialize_from_mass
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_kv
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main(): 
    #  Define Vehicle 
    vehicle =  vehicle_setup()

    # Set Up conditions 
    state = SUAVE.Analyses.Mission.Segments.Conditions.State()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()    
        
    ctrl_pts   = 4 
    AoA        = 2.14*np.ones((ctrl_pts,1))*Units.degrees
    Mach       = 0.2
    rho        = 0.365184   
    mu         = 0.0000143326   
    T          = 216.827
    P          = 22729.3    
    a          = 295.190 
    re         = rho*a*Mach/mu    
    state.conditions.freestream.mach_number             = Mach  * np.ones_like(AoA) 
    state.conditions.freestream.density                 = rho * np.ones_like(AoA) 
    state.conditions.freestream.dynamic_viscosity       = mu  * np.ones_like(AoA) 
    state.conditions.freestream.temperature             = T   * np.ones_like(AoA) 
    state.conditions.freestream.pressure                = P   * np.ones_like(AoA) 
    state.conditions.freestream.reynolds_number         = re  * np.ones_like(AoA)
    state.conditions.freestream.speed_of_sound          = a   * np.ones_like(AoA)
    state.conditions.freestream.velocity                = 59.00928* np.ones_like(AoA) 
    state.conditions.aerodynamics.angle_of_attack       = AoA  
    state.conditions.frames                             = Data()  
    state.conditions.frames.inertial                    = Data()  
    state.conditions.frames.body                        = Data()  
    state.conditions.frames.body.transform_to_inertial  = np.array([[[1., 0., 0],[0., 1., 0.],[0., 0., 1.]]]) 
    state.conditions.propulsion.throttle                = np.ones((ctrl_pts,1))
    velocity_vector                                     = np.array([[59.00928, 0. ,0.]])
    state.conditions.frames.inertial.velocity_vector    = np.tile(velocity_vector,(ctrl_pts,1))       

    propeller = vehicle.propulsors.propulsor.propeller    
    propeller.inputs.omega                              = np.ones((ctrl_pts,1)) * 1260 * Units['rpm']  
  
    settings = Data()
    settings.use_surrogate = False 
    vortices = 5
    settings.number_panels_spanwise  = vortices **2 
    settings.number_panels_chordwise = vortices   
    
    # Run Propeller model 
    F, Q, P, Cp ,  outputs , etap = propeller.spin(state.conditions)
    propeller.outputs = outputs
    
    # Run VLM With No Surrogate                
    CL, CDi, CM, CL_wing, CDi_wing, cl_y , cdi_y , CP  =  VLM(state.conditions,settings,vehicle)
    
    # Plot Results   	
    VD         = vehicle.vortex_distribution	 	
    n_sw       = VD.n_sw 
    n_cw       = VD.n_cw 
    n_w        = VD.n_w
       	 
    for ti in range(ctrl_pts):    
        line = ['-b','-b','-r','-r','-k']
        fig  = plt.figure()
        fig.set_size_inches(12, 12)       
        axes = fig.add_subplot(1,1,1)
        for i in range(n_w): 
            y_pts = VD.Y_SW[i*(n_sw):(i+1)*(n_sw)]
            z_pts = cl_y[ti][i*(n_sw):(i+1)*(n_sw)]
            axes.plot(y_pts, z_pts, line[i] ) 
        axes.set_xlabel("Spanwise Location (m)")
        axes.set_ylabel("Sectional CL")
        axes.set_title('$C_{Ly}$')
        
   
        fig        = plt.figure()	
        axes       = fig.add_subplot(1, 1, 1)  
        x_max      = max(VD.XC) + 2
        y_max      = max(VD.YC) + 2
        axes.set_ylim(x_max, 0)
        axes.set_xlim(-y_max, y_max)            
        fig.set_size_inches(12, 12)         	 
        for i in range(n_w):
            x_pts = np.reshape(np.atleast_2d(VD.XC[i*(n_sw*n_cw):(i+1)*(n_sw*n_cw)]).T, (n_sw,-1))
            y_pts = np.reshape(np.atleast_2d(VD.YC[i*(n_sw*n_cw):(i+1)*(n_sw*n_cw)]).T, (n_sw,-1))
            z_pts = np.reshape(np.atleast_2d(CP[ti][i*(n_sw*n_cw):(i+1)*(n_sw*n_cw)]).T, (n_sw,-1))  
            levals = np.linspace(-.02,.003,500)
            CS = axes.contourf( y_pts, x_pts,z_pts, cmap = 'jet',levels=levals,extend='both')  
            
        # Set Color bar	
        cbar = fig.colorbar(CS, ax=axes)
        cbar.ax.set_ylabel('$C_{P}$', rotation =  0)  # angle   
        plt.axis('off')	
        plt.grid(None)     
            
            
    return  

def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Cessna_172_SP'    


    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff   = 2550. * Units.pounds
    vehicle.mass_properties.takeoff       = 2550. * Units.pounds
    vehicle.mass_properties.max_zero_fuel = 2550. * Units.pounds
    vehicle.mass_properties.cargo         = 0. 

    # envelope properties
    vehicle.envelope.ultimate_load = 5.7
    vehicle.envelope.limit_load    = 3.8

    # basic parameters
    vehicle.reference_area         = 15.45  
    vehicle.passengers             = 4

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.span_efficiency         = 0.9
    wing.areas.reference         = 15.45 * Units['meters**2']  
    wing.spans.projected         = 11. * Units.meter  

    wing.chords.root             = 1.67 * Units.meter  
    wing.chords.tip              = 1.14 * Units.meter  
    wing.chords.mean_aerodynamic = 1.47 * Units.meter   
    wing.taper                   = wing.chords.root/wing.chords.tip

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 3.0 * Units.degrees
    wing.twists.tip              = 1.5 * Units.degrees

    wing.origin                  = [2.032, 0., 0.] 
    wing.aerodynamic_center      = [0.558, 0., 0.]

    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True

    wing.dynamic_pressure_ratio  = 1.0

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'

    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.span_efficiency         = 0.95
    wing.areas.reference         = 3.74 * Units['meters**2']  
    wing.spans.projected         = 3.454  * Units.meter 
    wing.sweeps.quarter_chord    = 12.5 * Units.deg
    
    wing.chords.root             = 1.397 * Units.meter 
    wing.chords.tip              = 0.762 * Units.meter 
    wing.chords.mean_aerodynamic = 1.09 * Units.meter 
    wing.taper                   = wing.chords.root/wing.chords.tip

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [6.248, 0., 0.] 
    wing.aerodynamic_center      = [0.508, 0., 0.]
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False

    wing.dynamic_pressure_ratio  = 0.9

    # add to vehicle
    vehicle.append_component(wing)


    ## ------------------------------------------------------------------
    ##   Vertical Stabilizer
    ## ------------------------------------------------------------------

    #wing = SUAVE.Components.Wings.Wing()
    #wing.tag = 'vertical_stabilizer'    

    #wing.sweeps.quarter_chord    = 25. * Units.deg
    #wing.thickness_to_chord      = 0.12
    #wing.span_efficiency         = 0.9
    #wing.areas.reference         = 2.258 * Units['meters**2']  
    #wing.spans.projected         = 1.854   * Units.meter 

    #wing.chords.root             = 1.6764 * Units.meter 
    #wing.chords.tip              = 0.6858 * Units.meter 
    #wing.chords.mean_aerodynamic = 1.21 * Units.meter 
    #wing.taper                   = wing.chords.root/wing.chords.tip

    #wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    #wing.twists.root             = 0.0 * Units.degrees
    #wing.twists.tip              = 0.0 * Units.degrees

    #wing.origin                  = [6.01 ,0,  0.623] 
    #wing.aerodynamic_center      = [0.508 ,0,0] 

    #wing.vertical                = True 
    #wing.symmetric               = False
    #wing.t_tail                  = False

    #wing.dynamic_pressure_ratio  = 1.0

    ## add to vehicle
    #vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------

    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'

    fuselage.seats_abreast         = 2.

    fuselage.fineness.nose         = 1.6
    fuselage.fineness.tail         = 2.

    fuselage.lengths.nose          = 60.  * Units.inches
    fuselage.lengths.tail          = 161. * Units.inches
    fuselage.lengths.cabin         = 105. * Units.inches
    fuselage.lengths.total         = 326. * Units.inches
    fuselage.lengths.fore_space    = 0.
    fuselage.lengths.aft_space     = 0.    

    fuselage.width                 = 42. * Units.inches

    fuselage.heights.maximum       = 62. * Units.inches
    fuselage.heights.at_quarter_length          = 62. * Units.inches
    fuselage.heights.at_three_quarters_length   = 62. * Units.inches
    fuselage.heights.at_wing_root_quarter_chord = 23. * Units.inches

    fuselage.areas.side_projected  = 8000.  * Units.inches**2.
    fuselage.areas.wetted          = 30000. * Units.inches**2.
    fuselage.areas.front_projected = 42.* 62. * Units.inches**2.
    fuselage.effective_diameter    = 50. * Units.inches


    # add to vehicle
    vehicle.append_component(fuselage)
    
    #---------------------------------------------------------------------------------------------
    # DEFINE PROPELLER
    #---------------------------------------------------------------------------------------------

    # build network    
    net = Battery_Propeller()
    net.number_of_engines = 2.
    net.nacelle_diameter  = 42 * Units.inches
    net.engine_length     = 0.01 * Units.inches
    net.areas             = Data()
    net.areas.wetted      = 0.01*(2*np.pi*0.01/2)    
    net.voltage           = 500.

    # Component 1 the ESC
    esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency = 0.95 # Gundlach for brushless motors
    net.esc        = esc

    # Component 2 the Propeller
    # Design the Propeller
    prop = SUAVE.Components.Energy.Converters.Propeller() 
    prop.prop_config         = 'pusher'
    prop.number_blades       = 2.0
    prop.freestream_velocity = 135.*Units['mph']    
    prop.angular_velocity    = 1600.  * Units.rpm # 2400
    prop.tip_radius          = 76./2. * Units.inches
    prop.hub_radius          = 8.     * Units.inches
    prop.design_Cl           = 0.8
    prop.design_altitude     = 12000. * Units.feet
    prop.design_thrust       = 820. 
    prop.design_power        = 0. * Units.watts 
    prop.origin              = [[2.,2.5,0.],[2.,-2.5,0.]]  #  prop influence    
    prop.rotation            = [1,-1]
    prop                     = propeller_design(prop)    
    net.propeller            = prop    
 
    # Component 6 the Payload
    payload = SUAVE.Components.Energy.Peripherals.Payload()
    payload.power_draw           = 0. #Watts 
    payload.mass_properties.mass = 1.0 * Units.kg
    net.payload                  = payload
    
    # Component 7 the Avionics
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 20. #Watts  
    net.avionics        = avionics      

    # Component 8 the Battery
    bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.mass_properties.mass = 500. * Units.kg  
    bat.specific_energy      = 350. * Units.Wh/Units.kg
    bat.resistance           = 0.006
    bat.max_voltage          = 500.
    
    initialize_from_mass(bat,bat.mass_properties.mass)
    net.battery              = bat
     
    
    # Propeller  motor 
    motor                              = SUAVE.Components.Energy.Converters.Motor()
    etam                               = 0.95
    v                                  = bat.max_voltage *3/4
    omeg                               = prop.angular_velocity  
    io                                 = 4.0 
    start_kv                           = 1
    end_kv                             = 25
    # do optimization to find kv or just do a linspace then remove all negative values, take smallest one use 0.05 change
    # essentially you are sizing the motor for a particular rpm which is sized for a design tip mach 
    # this reduces the bookkeeping errors     
    possible_kv_vals                   = np.linspace(start_kv,end_kv,(end_kv-start_kv)*20 +1 , endpoint = True) * Units.rpm
    res_kv_vals                        = ((v-omeg/possible_kv_vals)*(1.-etam*v*possible_kv_vals/omeg))/io  
    positive_res_vals                  = np.extract(res_kv_vals > 0 ,res_kv_vals) 
    kv_idx                             = np.where(res_kv_vals == min(positive_res_vals))[0][0]   
    kv                                 = possible_kv_vals[kv_idx]  
    res                                = min(positive_res_vals) 
    
    motor.mass_properties.mass         = 10. * Units.kg
    motor.origin                       = prop.origin  
    motor.propeller_radius             = prop.tip_radius   
    motor.speed_constant               = kv
    motor.resistance                   = res
    motor.no_load_current              = io 
    motor.gear_ratio                   = 1. 
    motor.gearbox_efficiency           = 1. # Gear box efficiency     
    net.motor                          = motor 
    
    # add the solar network to the vehicle
    vehicle.append_component(net)          

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    return vehicle


if __name__ == '__main__': 
    main()    
    plt.show()