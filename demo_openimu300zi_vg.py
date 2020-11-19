# -*- coding: utf-8 -*-
# Filename: demo_openimu300zi_vg.py

"""
simulate OpenIMU300ZI VG algo.
Created on 2020-02-13
@author: Ocean
"""

import os
import math
import numpy as np
from gnss_ins_sim.sim import imu_model
from gnss_ins_sim.sim import ins_sim

# globals
D2R = math.pi/180.0

motion_def_path = os.path.abspath('.//demo_motion_def_files//')
fs = 200.0          # IMU sample frequency

def test_openimu300zi_sim():
    '''
    simulate OpenIMU300ZI VG algo.
    '''
    #### choose a built-in IMU model, typical for IMU380
    n = 1.0 # 1.0: add noise; 0.0: without noise.
    imu_err = {'gyro_b': np.array([1.0, -1.0, 0.5]) * n,
               'gyro_arw': np.array([0.25, 0.25, 0.25]) * n,
               'gyro_b_stability': np.array([3.5, 3.5, 3.5]) * n,
               'gyro_b_corr': np.array([100.0, 100.0, 100.0]),
               'accel_b': np.array([50.0e-3, 50.0e-3, 50.0e-3]) * 0.0,
               'accel_vrw': np.array([0.03119, 0.03009, 0.04779]) * n,
               'accel_b_stability': np.array([4.29e-5, 5.72e-5, 8.02e-5]) * n,
               'accel_b_corr': np.array([200.0, 200.0, 200.0]),
               'mag_std': np.array([0.2, 0.2, 0.2]) * n
              }

    imu_err_IMU381 = {'gyro_b': np.array([0.0, 0.0, 0.0]) * n,
               'gyro_arw': np.array([0.3, 0.3, 0.3]) * n,
               'gyro_b_stability': np.array([6, 6, 6]) * n,
               'gyro_b_corr': np.array([100.0, 100.0, 100.0]),
               'accel_b': np.array([50.0e-3, 50.0e-3, 50.0e-3]) * 0.0,
               'accel_vrw': np.array([0.05, 0.05, 0.05]) * n,
               'accel_b_stability': np.array([2e-5 * 9.8, 2e-5 * 9.8, 2e-5 * 9.8]) * n,
               'accel_b_corr': np.array([200.0, 200.0, 200.0]),
               'mag_std': np.array([0.2, 0.2, 0.2]) * n
              }

    imu_err_IMU330 = {'gyro_b': np.array([0.0, 0.0, 0.0]) * n,
               'gyro_arw': np.array([0.2, 0.2, 0.2]) * n,
               'gyro_b_stability': np.array([1.5, 1.5, 1.5]) * n,
               'gyro_b_corr': np.array([100.0, 100.0, 100.0]),
               'accel_b': np.array([50.0e-3, 50.0e-3, 50.0e-3]) * 0.0,
               'accel_vrw': np.array([0.04, 0.04, 0.04]) * n,
               'accel_b_stability': np.array([2e-5 * 9.8, 2e-5 * 9.8, 2e-5 * 9.8]) * n,
               'accel_b_corr': np.array([200.0, 200.0, 200.0]),
               'mag_std': np.array([0.2, 0.2, 0.2]) * n
              }

    # do not generate GPS and magnetometer data
    imu = imu_model.IMU(accuracy=imu_err_IMU381, axis=9, gps=False)

    #### Algorithm
    # OpenIMU300ZI VG algorithm
    from demo_algorithms import openimu300zi_vg
    cfg_file = os.path.abspath('.//demo_algorithms//dmu380_sim_lib//ekfSim_tilt.cfg')
    algo = openimu300zi_vg.OpenIMU300ZISim(cfg_file)

    #### start simulation
    sim = ins_sim.Sim([fs, 0.0, fs],
                      motion_def_path+"//motion_def.csv", # motion_def motion_def-90deg_turn   motion_def-static
                    #   ".//demo_saved_data//car_test_20180929//",
                      ref_frame=1,
                      imu=imu,
                      # mode=None,
                      # env=None,
                      # env='[0.1 0.01 0.11]g-random',
                      # env='[0.05 0.005 0.055]g-random',
                      env='[0.5 0.4 0.65]-random',
                      # env='[0.3 0.3 0.5]-random',
                      # env='[0.1 0.1 0.2]-random',
                      # env='[0.1 0.2 0.3]g-10Hz-sinusoidal',
                      algorithm=algo)
    sim.run(1)
    # generate simulation results, summary, and save data to files
    sim.results()  # do not save data
    # sim.results('.//demo_saved_data//tmp', gen_kml=True)

    # plot data
    # sim.plot(['att_euler'])
    sim.plot(['att_euler','wb'], opt={'att_euler':'error'})

if __name__ == '__main__':
    test_openimu300zi_sim()
