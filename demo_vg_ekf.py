# -*- coding: utf-8 -*-
# Filename: demo_vg_ekf.py

"""
Created on 2019-12-26
@author: Ocean
"""

import os
import math
import numpy as np
from gnss_ins_sim.sim import imu_model
from gnss_ins_sim.sim import ins_sim

# globals
D2R = math.pi/180

motion_def_path = os.path.abspath('.//demo_motion_def_files//')
fs = 100.0          # IMU sample frequency

def vg_ekf_test():
    '''
    vg_ekf_test
    '''
    #### IMU model, typical for IMU381
    imu_err = {'gyro_b': np.array([0.0, 0.0, 0.0]),
               'gyro_arw': np.array([0.25, 0.25, 0.25]) * 1.0,
               'gyro_b_stability': np.array([3.5, 3.5, 3.5]) * 1.0,
               'gyro_b_corr': np.array([100.0, 100.0, 100.0]),
               'accel_b': np.array([0.0e-3, 0.0e-3, 0.0e-3]),
               'accel_vrw': np.array([0.03119, 0.03009, 0.04779]) * 1.0,
               'accel_b_stability': np.array([4.29e-5, 5.72e-5, 8.02e-5]) * 1.0,
               'accel_b_corr': np.array([200.0, 200.0, 200.0]),
               'mag_std': np.array([0.2, 0.2, 0.2]) * 1.0
              }
    # do not generate mag and GPS data
    imu = imu_model.IMU(accuracy=imu_err, axis=6, gps=False)

    #### Algorithm
    # mahony filter in a virtual inertial frame.
    from demo_algorithms import vg_ekf
    '''
    calculate pitch, roll by EKF.
    '''
    # create the algorith object
    algo = vg_ekf.VGEKFTest()

    #### start simulation
    sim = ins_sim.Sim([fs, 0.0, 0.0],
                        # '/Users/songyang/project/code/github/gnss-ins-sim_learn/demo_data_files/bosch',
                      motion_def_path+"//motion_def-90deg_turn.csv", #motion_def.csv   motion_def-90deg_turn.csv
                      ref_frame=1,
                      imu=imu,
                      mode=None,
                      env=None,
                      algorithm=[algo])
    # run the simulation for 1000 times
    sim.run(1)
    # generate simulation results, summary
    # do not save data since the simulation runs for 1000 times and generates too many results
    sim.results(err_stats_start=-1, gen_kml=False)
    # plot attitude error
    sim.plot(['att_euler'])

if __name__ == '__main__':
    vg_ekf_test()
